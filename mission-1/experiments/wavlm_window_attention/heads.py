"""Small mean/attention heads; outer OOF never participates in early stopping."""
import copy
import io
from pathlib import Path
import warnings
import time

import numpy as np
import torch
from torch import nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from features import atomic, read, require, save

MAX_EPOCHS = 12
PATIENCE = 3
BATCH = 64
SEED = 42


def pooled(values):
    return np.stack([(x * (w / w.sum())[:, None]).sum(0) for x, w in values]).astype(np.float32)


def metrics(y, p):
    require(p.shape == (len(y), 2) and np.isfinite(p).all(), 'Invalid probabilities')
    require((p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, atol=1e-6), 'Probability sum')
    pred = p.argmax(1)
    return dict(n=len(y), correct=int((pred == y).sum()), wrong=int((pred != y).sum()),
                accuracy=float((pred == y).mean()),
                gender_accuracy={g: float((pred[y == i] == i).mean()) for i, g in enumerate('FM')},
                confusion_F_M=[[int(((y == i) & (pred == j)).sum()) for j in range(2)] for i in range(2)])


class Head(nn.Module):
    def __init__(self, kind, dim=1024):
        super().__init__(); self.kind = kind
        # Same seed gives both candidates the exact same initial classifier.
        self.dropout = nn.Dropout(.1)
        self.classifier = nn.Linear(dim, 2)
        if kind == 'attention':
            self.attention = nn.Sequential(nn.Linear(dim, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
            nn.init.zeros_(self.attention[-1].weight)
        else:
            require(kind == 'mean', 'Unknown head')

    def forward(self, x, weights):
        prior = weights / weights.sum(1, keepdim=True)
        if self.kind == 'attention':
            logits = self.attention(x).squeeze(-1) + prior.clamp_min(1e-20).log()
            logits = logits.masked_fill(weights == 0, -torch.inf)
            alpha = logits.softmax(1)
        else:
            alpha = prior
        pooled_x = (x * alpha.unsqueeze(-1)).sum(1)
        kl = (alpha * (alpha.clamp_min(1e-20).log() - prior.clamp_min(1e-20).log())).sum(1).mean()
        return self.classifier(self.dropout(pooled_x)), kl


def batch(values, ix, mean, scale, device):
    length = max(len(values[i][0]) for i in ix)
    dim = len(mean)
    x = np.zeros((len(ix), length, dim), np.float32)
    w = np.zeros((len(ix), length), np.float32)
    for row, i in enumerate(ix):
        a, b = values[i]; n = len(a)
        x[row, :n] = (a-mean)/scale
        w[row, :n] = b / b.sum()
    return torch.as_tensor(x, device=device), torch.as_tensor(w, device=device)


def torch_save(path, obj):
    b = io.BytesIO(); torch.save(obj, b); atomic(path, b.getvalue())


def load(path):
    # Only load checkpoints generated in this run's identity-locked output.
    return torch.load(path, map_location='cpu', weights_only=False)


def predict_head(record, values, device):
    model = Head(record['kind'], len(record['mean'])).to(device)
    model.load_state_dict(record['state']); model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(values), BATCH):
            x, w = batch(values, range(start, min(start+BATCH, len(values))), record['mean'], record['scale'], device)
            predictions.append(model(x, w)[0].softmax(1).cpu().numpy())
    return np.concatenate(predictions)


def train_head(values, y, kind, path, device, stop_values=None, stop_y=None, epochs=None):
    """Tune on an inner stop set; refit from scratch for the selected epoch count."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    scaler = StandardScaler().fit(pooled(values))
    mean, scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    model = Head(kind, len(mean)).to(device)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    limit = epochs if epochs is not None else MAX_EPOCHS
    best_loss = float('inf'); best_epoch = 0; best_state = None; stale = 0; start = 0
    history = []
    if path.exists():
        checkpoint = load(path)
        require(checkpoint['kind'] == kind and checkpoint['limit'] == limit and checkpoint['train_count'] == len(y), 'Head resume mismatch')
        if checkpoint['done']: return checkpoint['record']
        model.load_state_dict(checkpoint['state']); optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss, best_epoch, best_state, stale = (checkpoint[k] for k in ('best_loss', 'best_epoch', 'best_state', 'stale'))
        start, history = checkpoint['epoch'], checkpoint['history']
        torch.set_rng_state(checkpoint['rng'])
        if device.startswith('cuda'): torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    for epoch in range(start, limit):
        model.train(); order = np.random.default_rng(SEED+epoch).permutation(len(y)); total_loss = 0.
        for begin in range(0, len(order), BATCH):
            ix = order[begin:begin+BATCH]
            x, w = batch(values, ix, mean, scale, device)
            target = torch.as_tensor(y[ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, kl = model(x, w)
            loss = nn.functional.cross_entropy(logits, target) + .01 * kl
            require(torch.isfinite(loss).item(), 'Non-finite head loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            total_loss += loss.item() * len(ix)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = dict(kind=kind, state=state, mean=mean, scale=scale, epochs=epoch+1)
        if stop_values is not None:
            p = predict_head(record, stop_values, device)
            val_loss = float(-np.log(p[np.arange(len(stop_y)), stop_y].clip(1e-12, 1)).mean())
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, best_state, stale = val_loss, epoch+1, state, 0
            else:
                stale += 1
        else:
            val_loss = None; best_epoch, best_state = epoch+1, state
        history.append(dict(epoch=epoch+1, training_loss=total_loss/len(y), inner_stop_loss=val_loss))
        done = epoch+1 == limit or (stop_values is not None and stale >= PATIENCE)
        record.update(state=best_state, epochs=best_epoch, history=history)
        torch_save(path, dict(kind=kind, limit=limit, train_count=len(y), epoch=epoch+1,
                             state=state, optimizer=optimizer.state_dict(), best_state=best_state,
                             best_loss=best_loss, best_epoch=best_epoch, stale=stale, history=history,
                             rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                             done=done, record=record))
        print(f'{path.stem}: epoch {epoch+1}/{limit}, train loss {total_loss/len(y):.4f}, stop {val_loss}', flush=True)
        if done: break
    return record


def fit_lr(values, y):
    x = pooled(values); scaler = StandardScaler().fit(x)
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        lr = LogisticRegression(C=.1, max_iter=2000, random_state=SEED).fit(scaler.transform(x), y)
    require(list(lr.classes_) == [0, 1], 'Class order')
    return dict(kind='lr', mean=scaler.mean_, scale=scaler.scale_, coef=lr.coef_, intercept=lr.intercept_)


def predict(record, values, device):
    if record['kind'] != 'lr': return predict_head(record, values, device)
    from scipy.special import expit
    x = (pooled(values)-record['mean'])/record['scale']
    male = expit((x @ record['coef'].T + record['intercept']).ravel())
    return np.column_stack([1-male, male])


def split_plan(y):
    result = []
    for k, (train, test) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        fit, stop = train_test_split(train, test_size=.15, stratify=y[train], random_state=SEED+k)
        require(not set(fit) & set(stop) and not set(train) & set(test), 'Split leakage')
        result.append((train, test, fit, stop))
    return result


def select(y, predictions, folds):
    baseline = metrics(y, predictions['lr']); rows = []
    for kind in ('mean', 'attention'):
        p = predictions[kind]; m = metrics(y, p)
        deltas = [metrics(y[folds == k], p[folds == k])['accuracy'] - metrics(y[folds == k], predictions['lr'][folds == k])['accuracy'] for k in range(3)]
        eligible = min(deltas) >= 0 and m['accuracy'] >= baseline['accuracy'] + .001
        if kind == 'attention':
            # Require a benefit over the matched mean head before attributing benefit to weighting.
            eligible = eligible and m['correct'] > metrics(y, predictions['mean'])['correct']
        rows.append(dict(kind=kind, metrics=m, fold_deltas=deltas, eligible=bool(eligible)))
    good = [r for r in rows if r['eligible']]
    chosen = max(good, key=lambda r: (r['metrics']['correct'], r['kind'] == 'mean'))['kind'] if good else 'lr'
    return dict(selected=chosen, baseline=baseline, candidates=rows,
                rule='No outer fold below LR; pooled gain >=0.1 percentage point; attention must beat mean; ties prefer mean; else LR',
                scope='Train OOF model selection, not unbiased test. Calls are disjoint; speakers may overlap.')


def develop(jobs, values, out, device):
    y = np.array([j['gender'] == 'M' for j in jobs], dtype=np.int64)
    ids = np.array([j['call_id'] for j in jobs]); plans = split_plan(y)
    ps = {k: np.zeros((len(y), 2), dtype=np.float64) for k in ('lr', 'mean', 'attention')}
    folds = np.full(len(y), -1); epochs = {k: [] for k in ('mean', 'attention')}
    measured_fold_seconds = []
    for k, (tr, test, inner, stop) in enumerate(plans):
        fold_path = out / 'development' / f'fold_{k}.pt'
        if fold_path.exists():
            result = load(fold_path)
            require(np.array_equal(result['ids'], ids[test]), 'OOF ID mismatch')
        else:
            fold_start = time.monotonic()
            subset = lambda ix: [values[i] for i in ix]
            p = {'lr': predict(fit_lr(subset(tr), y[tr]), subset(test), device)}; ep = {}
            for kind in ('mean', 'attention'):
                tune = train_head(subset(inner), y[inner], kind, out/'development'/f'fold_{k}_{kind}_tune.pt', device, subset(stop), y[stop])
                ep[kind] = tune['epochs']
                refit = train_head(subset(tr), y[tr], kind, out/'development'/f'fold_{k}_{kind}_refit.pt', device, epochs=ep[kind])
                p[kind] = predict(refit, subset(test), device)
            result = dict(ids=ids[test], predictions=p, epochs=ep)
            torch_save(fold_path, result)
            measured_fold_seconds.append(time.monotonic()-fold_start)
        for kind in ps:
            metrics(y[test], result['predictions'][kind])
            ps[kind][test] = result['predictions'][kind]
        for kind in epochs: epochs[kind].append(result['epochs'][kind])
        folds[test] = k
        print(f'Outer fold {k+1}/3 complete', flush=True)
        if measured_fold_seconds:
            print(f'Remaining fold ETA: {np.mean(measured_fold_seconds)*(2-k)/60:.1f} min (final refit/evaluation extra)', flush=True)
    require((folds >= 0).all(), 'OOF coverage')
    selection = select(y, ps, folds)
    selection['epoch_counts'] = epochs
    selection['final_epochs'] = int(np.median(epochs[selection['selected']])) if selection['selected'] != 'lr' else None
    torch_save(out/'development/oof.pt', dict(ids=ids, y=y, folds=folds, predictions=ps))
    return selection
