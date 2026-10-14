"""L4 window weighting experiment: benchmark, extract Train, nested selection, locked evaluation."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import shutil
import time
import zipfile

import numpy as np
import torch

from features import (MODEL, REVISION, WINDOW, INPUT_SHA, Encoder, atomic, benchmark_jobs,
                      caller_samples, digest, extract, load_jobs, read, read_block,
                      read_partition, require, save, sha, waveform, write_block)
from heads import MAX_EPOCHS, PATIENCE, BATCH, develop, fit_lr, load, metrics, predict, train_head, torch_save


def environment(config):
    os.environ['HF_HOME'] = config['hf_cache']
    import scipy, sklearn, transformers, soundfile
    require(torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(), 'This experiment expects an L4 GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    return dict(model=MODEL, revision=REVISION, input_sha256=INPUT_SHA,
                source={n: sha(Path(__file__).with_name(n)) for n in ('run.py', 'features.py', 'heads.py')},
                config=config, window_seconds=WINDOW, feature='last_hidden_state mean per balanced window',
                gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, precision='bf16 extraction / fp32 head',
                python=platform.python_version(), packages={k: v.__version__ for k, v in [('torch', torch), ('numpy', np), ('scipy', scipy), ('sklearn', sklearn), ('transformers', transformers), ('soundfile', soundfile)]},
                design=dict(folds=3, inner_stop_fraction=.15, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                            batch=BATCH, head_lr=.001, weight_decay=.001, dropout=.1, prior_KL=.01,
                            candidates=['lr', 'mean', 'attention'], seed=42))


def benchmark(jobs, config, out, ih):
    path = out/'benchmark.json'
    if path.exists():
        report = read(path); require(report['identity'] == ih, 'Benchmark identity')
        print(json.dumps(report, indent=2)); return
    subset = benchmark_jobs(jobs)
    missing = [j for j in subset if not (Path(config['data_root'])/j['wav_relative_path']).is_file()]
    require(not missing, 'Audio DATA_ROOT incorrect or missing WAV: ' + (missing[0]['wav_relative_path'] if missing else ''))
    t = time.monotonic(); encoder = Encoder(); model_load = time.monotonic()-t
    # Warm up separately; never train the frozen backbone.
    encoder.encode(waveform(subset[len(subset)//2], config['data_root']))
    records = []
    for i, j in enumerate(subset):
        p = out/'benchmark_features'/(j['call_id']+'.npz')
        timing = p.with_suffix('.json')
        if p.exists() and timing.exists():
            read_block(p, [j], ih); r = read(timing)
            require(r['identity'] == ih, 'Benchmark timing identity')
        else:
            t = time.monotonic()
            value = encoder.encode(waveform(j, config['data_root']))
            write_block(p, [j], [value], ih)
            r = dict(identity=ih, call_id=j['call_id'], seconds=time.monotonic()-t, samples=caller_samples(j))
            save(timing, r)
        records.append(r)
        if (i+1) % 20 == 0: print(f'Benchmark: {i+1}/200', flush=True)
    ratio = sum(r['seconds'] for r in records) / sum(r['samples'] for r in records)
    estimates = {p: ratio*sum(caller_samples(j) for j in jobs if j['partition'] == p)/3600 for p in ('train', 'internal_validation')}
    report = dict(identity=ih, measured_calls=200, model_load_seconds=model_load,
                  measured_seconds=sum(r['seconds'] for r in records),
                  projected_extraction_hours=estimates,
                  total_extraction_hours=sum(estimates.values()),
                  note='Estimate, not deadline: includes WAV reads, BF16 forward and Drive writes. Drive throttling may dominate. Benchmark uses per-call files; full extraction writes 64-call blocks. Head CV time is extra and measured during development. Benchmark calls are reused.')
    save(path, report); print(json.dumps(report, indent=2), flush=True)


def export_results(out):
    with zipfile.ZipFile(out/'window_attention_results.zip.tmp', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('identity.json', 'benchmark.json', 'selection.lock.json', 'final_models.json',
                     'validation_started.json', 'completion.json', 'REPORT.md', 'validation_predictions.csv', 'events.jsonl'):
            z.write(out/name, name)
        for path in sorted((out/'logs').glob('*.log')):
            z.write(path, 'logs/'+path.name)
    os.replace(out/'window_attention_results.zip.tmp', out/'window_attention_results.zip')


def final_fit(jobs, config, out, ih):
    lock = read(out/'selection.lock.json')
    require(lock['identity'] == ih, 'Selection identity mismatch')
    if (out/'final_models.json').exists():
        final = read(out/'final_models.json')
        require(final['selection_sha256'] == sha(out/'selection.lock.json') and final['checkpoint_sha256'] == sha(out/'model.pt'), 'Final checkpoint changed')
        return
    require(not (out/'validation_started.json').exists(), 'Cannot refit after evaluation starts')
    train, values = read_partition(jobs, out, ih, 'train')
    y = np.array([j['gender'] == 'M' for j in train], dtype=np.int64)
    lr = fit_lr(values, y)
    selected = lr if lock['selected'] == 'lr' else train_head(values, y, lock['selected'], out/'final_head_resume.pt', 'cuda', epochs=lock['final_epochs'])
    checkpoint = dict(identity=ih, selected=selected, baseline=lr, class_order=['F', 'M'],
                      model=MODEL, revision=REVISION, window_seconds=WINDOW,
                      selection_sha256=sha(out/'selection.lock.json'))
    torch_save(out/'model.pt', checkpoint)
    save(out/'final_models.json', dict(identity=ih, selection_sha256=checkpoint['selection_sha256'], checkpoint_sha256=sha(out/'model.pt')))


def evaluate(jobs, config, out, ih):
    final = read(out/'final_models.json'); lock = read(out/'selection.lock.json')
    require(final['identity'] == lock['identity'] == ih, 'Final identity changed')
    require(final['selection_sha256'] == sha(out/'selection.lock.json') and final['checkpoint_sha256'] == sha(out/'model.pt'), 'Selection/model hash mismatch')
    marker = dict(identity=ih, final_models_sha256=sha(out/'final_models.json'))
    if (out/'validation_started.json').exists(): require(read(out/'validation_started.json') == marker, 'Evaluation identity changed')
    else: save(out/'validation_started.json', marker)
    extract(jobs, config['data_root'], out, ih, 'internal_validation')
    val, values = read_partition(jobs, out, ih, 'internal_validation')
    y = np.array([j['gender'] == 'M' for j in val], dtype=np.int64)
    checkpoint = load(out/'model.pt')
    p = predict(checkpoint['selected'], values, 'cuda'); baseline = predict(checkpoint['baseline'], values, 'cuda')
    selected_metrics, base_metrics = metrics(y, p), metrics(y, baseline)
    slices = {}
    durations = np.array([caller_samples(j)/16000 for j in val])
    for name, mask in [('under_12s', durations < 12), ('12_to_36s', (durations >= 12) & (durations < 36)), ('36s_plus', durations >= 36)]:
        if mask.any():
            slices[name] = dict(n=int(mask.sum()), selected_accuracy=float((p[mask].argmax(1)==y[mask]).mean()), baseline_accuracy=float((baseline[mask].argmax(1)==y[mask]).mean()))
    new_ok, old_ok = p.argmax(1) == y, baseline.argmax(1) == y
    result = dict(status='complete', identity=ih, selected=lock['selected'], metrics=selected_metrics,
                  same_pipeline_LR=base_metrics, versus_LR=dict(corrected=int((new_ok & ~old_ok).sum()), regressed=int((~new_ok & old_ok).sum())),
                  duration_slices=slices, reached_99=selected_metrics['correct'] >= 5542,
                  historical_results=dict(wavlm_correct=5490, ensemble_correct=5497),
                  historical_baseline_count_reproduced=base_metrics['correct'] == 5490,
                  scope='Previously observed Internal Validation; not independent test. Official Validation unused. No speaker-disjoint guarantee. Historical comparisons are descriptive, not candidate selection.')
    s = io.StringIO(); w = csv.writer(s, lineterminator='\n')
    w.writerow(['call_id', 'gender', 'prediction', 'p_F', 'p_M', 'baseline_prediction', 'caller_seconds'])
    for j, probs, b, seconds in zip(val, p, baseline, durations):
        w.writerow([j['call_id'], j['gender'], 'FM'[int(probs.argmax())], *probs, 'FM'[int(b.argmax())], seconds])
    atomic(out/'validation_predictions.csv', s.getvalue().encode())
    report = f"# Frozen WavLM window weighting\n\nSelected: {lock['selected']}\n\nAccuracy: {selected_metrics['accuracy']:.6%} ({selected_metrics['correct']}/5597), wrong: {selected_metrics['wrong']}\n\nSame-pipeline LR: {base_metrics['accuracy']:.6%}\n\nHistorical ensemble: 98.213329% (5497/5597).\n\n{result['scope']}\n"
    atomic(out/'REPORT.md', report.encode()); save(out/'completion.json', result)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['benchmark', 'extract-train', 'develop', 'final-fit', 'evaluate'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = read(args.config)
    # Import Transformers only after disabling unrelated optional vision/audio modules.
    import sys
    sys.modules['torchvision'] = None; sys.modules['librosa'] = None
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    require(out.resolve() != Path(config['data_root']).resolve(), 'Separate output/data directories required')
    import fcntl
    mutex = open('/tmp/aigo-wavlm-window-gpu.lock', 'w')
    fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Same VM lock only: one runtime per shared Drive output is required.
    jobs = load_jobs(config['input_json']); ident = environment(config); ih = digest(ident)
    if (out/'identity.json').exists(): require(read(out/'identity.json') == ident, 'Environment/source/config changed. Use a new output directory.')
    else:
        require(not any(p.name != 'logs' for p in out.iterdir()), 'Nonempty output without identity')
        save(out/'identity.json', ident)
    started = time.monotonic(); event = dict(stage=args.stage, identity=ih, started=time.time())
    try:
        if args.stage == 'benchmark':
            require(shutil.disk_usage(out).free > 4*2**30, 'Need 4 GiB free output space')
            benchmark(jobs, config, out, ih)
        else:
            require(read(out/'benchmark.json')['identity'] == ih, 'Benchmark first')
            if args.stage == 'extract-train':
                require(not (out/'selection.lock.json').exists(), 'Selection already locked; proceed to final-fit/evaluate')
                extract(jobs, config['data_root'], out, ih, 'train')
            elif args.stage == 'develop':
                if (out/'selection.lock.json').exists():
                    print(json.dumps(read(out/'selection.lock.json'), indent=2))
                else:
                    require(not (out/'validation_started.json').exists(), 'No development after evaluation')
                    train, values = read_partition(jobs, out, ih, 'train')
                    result = develop(train, values, out, 'cuda'); result['identity'] = ih
                    save(out/'selection.lock.json', result); print(json.dumps(result, indent=2), flush=True)
            elif args.stage == 'final-fit': final_fit(jobs, config, out, ih)
            elif args.stage == 'evaluate':
                if not (out/'completion.json').exists(): evaluate(jobs, config, out, ih)
                else: print((out/'REPORT.md').read_text())
        event.update(status='complete', session_seconds=time.monotonic()-started)
    except BaseException as exc:
        event.update(status='failed', error=repr(exc), session_seconds=time.monotonic()-started)
        raise
    finally:
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n'); f.flush(); os.fsync(f.fileno())
    if args.stage == 'evaluate': export_results(out)


if __name__ == '__main__': main()
