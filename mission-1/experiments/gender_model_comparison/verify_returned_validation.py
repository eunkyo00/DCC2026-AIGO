"""Verify the original execution bundle, including the known diagnostic-only revision.

Never edits the current bundle or returned results. Reconstructs the previous runner
only if every reconstructed file and all manifest metadata match the returned hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
EXPECTED_ZIP = '2a26d6d23a15d182c9e7fbf7085a0eb6a4bf409f0bb2e6b7b7aad67bc835271e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    archive_hash = hashlib.sha256(args.results.read_bytes()).hexdigest()
    if archive_hash != EXPECTED_ZIP:
        raise RuntimeError('ZIP differs from the user-reported Colab SHA256')
    current = json.loads((ROOT/'validation_bundle_manifest.json').read_text())
    with zipfile.ZipFile(args.results) as z:
        manifest_bytes = z.read('validation_bundle_manifest.json')
        progress = json.loads(z.read('progress.json'))
    executed = json.loads(manifest_bytes)
    if {k:v for k,v in current.items() if k!='files'} != {k:v for k,v in executed.items() if k!='files'}:
        raise RuntimeError('Execution metadata differs from the prepared bundle')
    if set(current['files']) != set(executed['files']):
        raise RuntimeError('Execution file set differs')
    files = {}
    for name, expected in executed['files'].items():
        data = (ROOT/name).read_bytes()
        if name == 'ecapa_validation.py' and hashlib.sha256(data).hexdigest() != expected:
            detailed = """    differences = [f'{p}: Train={expected!r}, current={versions[p]!r}'
                   for p, expected in old['versions'].items() if versions[p] != expected]
    require(not differences, 'Packages differ from Train screen; do not mix caches:\\n' +
            '\\n'.join(differences))"""
            original = "    require(versions == old['versions'], 'Packages differ from Train screen; do not mix caches')"
            text = data.decode()
            if text.count(detailed) != 1:
                raise RuntimeError('Cannot reconstruct known diagnostic-only change')
            data = text.replace(detailed, original).encode()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f'Executed code hash mismatch: {name}')
        files[name] = data
    output = ROOT/'validation_results'
    output.mkdir(exist_ok=True)
    reference = ROOT/'artifacts/executed_ecapa_validation_bundle.zip'
    with zipfile.ZipFile(reference,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            z.writestr(name,data)
        z.writestr('validation_bundle_manifest.json',manifest_bytes)
    with tempfile.TemporaryDirectory() as tmp:
        mission = Path(tmp)/'mission-1'
        work = mission/'experiments/gender_model_comparison'
        work.mkdir(parents=True)
        for name,data in files.items():
            p = work/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(data)
        (work/'validation_bundle_manifest.json').write_bytes(manifest_bytes)
        # Read-only analysis of the original baseline files; no training or GPU.
        for name in ('baseline','ssl'):
            (mission/name).symlink_to(ROOT.parents[1]/name, target_is_directory=True)
        subprocess.run([sys.executable,str(work/'analyze_ecapa_validation.py'),str(args.results.resolve()),
                        '--output',str(output)],check=True)
    metrics = json.loads((output/'metrics.json').read_text())
    metrics['runtime'] = {'gpu': 'NVIDIA L4', 'calls_in_session': progress['session_calls'],
                          'inference_and_drive_wall_seconds': progress['session_wall_seconds_including_drive_flush']}
    (output/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    with (output/'REPORT.md').open('a') as f:
        f.write('\n실측: NVIDIA L4, 추론·Drive 저장 7,720.671초 (2시간 8분 41초). 모델 로딩·최종 ZIP 생성 제외.\n')
        f.write('Wav2Vec2 대비 0.428801%p 낮고 오답이 24개 많다. 기존 최고 baseline은 Wav2Vec2로 유지한다.\n')
        f.write('실행 후 변경된 패키지 오류 표시 문구는 당시 버전으로 복원해 모든 bundle 파일 hash를 검증했다.\n')
    provenance = dict(result_zip_sha256=archive_hash,
                      executed_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                      executed_runner_sha256=executed['files']['ecapa_validation.py'],
                      all_executed_file_hashes_verified=True,
                      change='Only package mismatch error detail changed after upload; exact prior runner reconstructed and hash-verified.',
                      reference_bundle=str(reference.relative_to(ROOT)))
    (output/'verification_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    main()
