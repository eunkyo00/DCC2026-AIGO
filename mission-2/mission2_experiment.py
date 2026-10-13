# Mission 2: 단일 파일 실험. 사용법과 지정할 경로는 파일 하단 주석 참조.
from __future__ import annotations
import csv, gc, hashlib, io, json, math, os, random, shutil, sys, tempfile, time, unicodedata, zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'm2-resume-1'
# Budget-only update: training/data/checkpoint semantics are unchanged.
COMPATIBLE_RUNNER_HASHES = {'5e4f258c051308b426be61e160c9025adad032858ff8c5f02962ac4e3cfdbbaa', 'ee778e7d15510ad7e71ac2ac86aa594f52fe330c6f83b694c21968f0518357ad'}
VARIANTS = {'base': dict(n_fft=2048, win_length=2048, hop_length=512, frames=24),
            'fine': dict(n_fft=512, win_length=200, hop_length=80, frames=151)}
CONDITIONS = [('A', 'base', 'mean'), ('B', 'fine', 'mean'),
              ('C', 'base', 'attention'), ('D', 'fine', 'attention')]

class PauseRun(RuntimeError): pass

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

class Store:
    """Two committed generations. A torn write never destroys both copies."""
    def __init__(self, root, signature):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.signature = signature; self.seq = -1
    def load(self):
        found = []; invalid = []
        for slot in (0, 1):
            meta = self.root / f'slot{slot}.json'; data = self.root / f'slot{slot}.pt'
            if not meta.exists(): continue
            try:
                m = read_json(meta)
                if m['signature'] != self.signature:
                    raise RuntimeError(f'설정이 다른 체크포인트입니다: {self.root}. 새 RUN_NAME을 사용하세요.')
                if sha(data) != m['sha256']: raise ValueError('checksum')
                found.append((m['seq'], data))
            except RuntimeError: raise
            except (OSError, ValueError, KeyError): invalid.append(slot)
        if invalid: print(f'미완성/손상 저장본 제외: {self.root.name} {invalid}', flush=True)
        for seq, data in sorted(found, reverse=True):
            try:
                # Only our own resume files contain Python RNG/optimizer objects.
                result = torch.load(data, map_location='cpu', weights_only=False)
                self.seq = seq; return result
            except Exception as e: print(f'체크포인트 읽기 실패, 이전 세대 확인: {type(e).__name__}', flush=True)
        if any(self.root.glob('slot*.json')):
            raise RuntimeError(f'복구 가능한 체크포인트가 없습니다: {self.root}. 원본을 보존하고 확인하세요.')
        return None
    def save(self, value):
        seq = self.seq + 1; slot = seq % 2
        with tempfile.TemporaryDirectory(prefix='m2_save_') as tmpdir:
            local = Path(tmpdir) / 'state.pt'; torch.save(value, local)
            checksum = sha(local)
            partial = self.root / f'slot{slot}.pt.partial'
            shutil.copyfile(local, partial)
            if sha(partial) != checksum: raise IOError('Drive checkpoint copy verification failed')
            os.replace(partial, self.root / f'slot{slot}.pt')
            atomic_json(self.root / f'slot{slot}.json', dict(seq=seq, signature=self.signature, sha256=checksum))
        self.seq = seq

class Clock:
    def __init__(self, root, minutes):
        self.store = Store(Path(root) / 'time_state', 'time-v1')
        old = self.store.load() or {'seconds': 0}
        self.previous = old['seconds']; self.start = time.monotonic(); self.limit = minutes * 60
        self.last = self.start
    @property
    def elapsed(self): return self.previous + time.monotonic() - self.start
    @property
    def remaining(self): return max(0, self.limit - self.elapsed)
    def save(self): self.store.save({'seconds': self.elapsed}); self.last = time.monotonic()
    def check(self):
        if time.monotonic() - self.last > 60: self.save()
        if self.remaining < 90:
            self.save(); raise PauseRun('누적 시간 예산 도달. 저장 완료. EXTRA_MINUTES를 늘리고 동일 RUN_NAME으로 재실행하면 이어집니다.')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['cpu'])
    if torch.cuda.is_available() and s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])

def norm(s): return unicodedata.normalize('NFC', str(s))

def read_ref(ref):
    if '::' in ref:
        archive, member = ref.split('::', 1)
        with zipfile.ZipFile(archive) as z: return z.read(member)
    return Path(ref).read_bytes()

def inventory(root, clock):
    """Index extracted files or ZIP members without changing shared source data."""
    root = Path(root)
    if not root.exists(): raise FileNotFoundError(f'접근할 수 없는 데이터 경로: {root}')
    records = {'wav': {}, 'json': {}}
    files = [root] if root.is_file() else (Path(d) / f for d, _, names in os.walk(root) for f in names)
    count = 0
    def add(name, ref, stamp):
        p = Path(name)
        if '__MACOSX' in p.parts or p.name.startswith('._'): return
        ext = p.suffix.lower().lstrip('.')
        if ext not in records: return
        key = norm(p.stem)
        if key in records[ext]: raise ValueError(f'중복 {ext} stem: {key}; 데이터 경로를 더 좁혀 주세요.')
        records[ext][key] = {'ref': ref, 'stamp': stamp}
    for p in files:
        if p.suffix.lower() in ('.wav', '.json'):
            st = p.stat(); add(p.name, str(p), [st.st_size, st.st_mtime_ns])
        elif p.suffix.lower() == '.zip':
            with zipfile.ZipFile(p) as z:
                for inf in z.infolist():
                    if not inf.is_dir(): add(inf.filename, str(p) + '::' + inf.filename, [inf.file_size, inf.CRC])
        count += 1
        if count % 256 == 0: clock.check()
    missing = set(records['wav']) ^ set(records['json'])
    if missing: raise ValueError(f'WAV/JSON 매칭 누락 {len(missing)}개. 예: {sorted(missing)[:3]}')
    if not records['wav']: raise ValueError(f'WAV/JSON이 없습니다: {root}. Mel 특징 ZIP은 원본 음성이 아닙니다.')
    return records

def address_of(label):
    if isinstance(label, dict):
        for key in ('address', 'Address', 'addr', '주소'):
            if isinstance(label.get(key), str): return label[key]
        for v in label.values():
            if isinstance(v, (dict, list)):
                a = address_of(v)
                if a is not None: return a
    elif isinstance(label, list):
        for v in label:
            a = address_of(v)
            if a is not None: return a
    return None

def parse_call(stem, wav, label):
    # Unlike broad string matching, only an address field can authorize Seoul training.
    address = address_of(label)
    if address is None: raise ValueError(f'{stem}: 주소 필드를 확인할 수 없어 서울 필터 적용 불가')
    if '서울' not in address: return None
    utterances = label.get('utterances')
    if not isinstance(utterances, list): raise ValueError(f'{stem}: utterances 스키마 불일치')
    rows = []
    for i, u in enumerate(utterances):
        if not isinstance(u, dict): raise ValueError(f'{stem}: 발화 형식 오류')
        raw = u.get('speaker', u.get('Speaker'))
        if str(raw).strip() not in ('0', '1'): raise ValueError(f'{stem}: speaker 0/1 필요')
        a, b = float(u['startAt']), float(u['endAt'])
        if not np.isfinite([a, b]).all() or a < 0 or b <= a: raise ValueError(f'{stem}: 발화 경계 오류')
        rows.append(dict(uid=f'{stem}:{i}', stem=stem, start=a, end=b, label=int(raw), duration=(b-a)/1000))
    if not rows: raise ValueError(f'{stem}: 빈 발화 목록')
    return dict(stem=stem, wav=wav, rows=rows)

def make_manifest(root, split, out, clock):
    index = inventory(root, clock); signature = digest(index)
    target = Path(out) / 'manifests' / split; target.mkdir(parents=True, exist_ok=True)
    store = Store(target / 'progress', signature)
    state = store.load() or {'next': 0, 'parts': [], 'excluded': 0}
    stems = sorted(index['wav']); step = 128
    for first in range(state['next'], len(stems), step):
        calls = []
        for stem in stems[first:first+step]:
            label = json.loads(read_ref(index['json'][stem]['ref']).decode('utf-8-sig'))
            call = parse_call(stem, index['wav'][stem]['ref'], label)
            if call: calls.append(call)
            else: state['excluded'] += 1
        name = f'part_{first:06d}.json'; atomic_json(target / name, calls)
        state['parts'].append({'name': name, 'sha256': sha(target / name)})
        state['next'] = min(first + step, len(stems)); store.save(state); clock.check()
        if first % 1024 == 0: print(f'{split} 라벨 확인 {state["next"]}/{len(stems)}', flush=True)
    calls = []
    for part in state['parts']:
        path = target / part['name']
        if sha(path) != part['sha256']: raise IOError(f'매니페스트 손상: {path}')
        calls.extend(read_json(path))
    print(f'{split}: 서울 {len(calls):,}통화 / {sum(len(c["rows"]) for c in calls):,}발화; 비서울 제외 {state["excluded"]}', flush=True)
    if not calls: raise ValueError('서울 통화가 없습니다. 주소/데이터 폴더 확인 필요')
    return calls, signature

def make_groups(calls, size=16): return [calls[i:i+size] for i in range(0, len(calls), size)]

def select_groups(groups, target, seed):
    order = np.random.default_rng(seed).permutation(len(groups)); chosen=[]; n=0
    for i in order:
        chosen.append(int(i)); n += sum(len(c['rows']) for c in groups[i])
        if n >= target: break
    return sorted(chosen)

def feature(segment, variant):
    import librosa
    spec = VARIANTS[variant]; frames = spec['frames']
    def mel(y):
        m = librosa.feature.melspectrogram(y=y, sr=8000, n_mels=64,
                n_fft=spec['n_fft'], win_length=spec['win_length'], hop_length=spec['hop_length'],
                center=True, pad_mode='constant', power=2.0)
        return np.clip((librosa.power_to_db(m, ref=np.max) + 80) / 80, 0, 1)
    first = np.pad(segment[:12000], (0, max(0, 12000-len(segment))))
    a = mel(first)
    if len(segment) <= 12000: b = a.copy()
    else:
        whole = mel(segment); edges=np.rint(np.linspace(0, whole.shape[1], frames+1)).astype(int)
        # Match the original implementation exactly, including float32 cumulative sums.
        cumulative=np.pad(np.cumsum(whole,axis=1,dtype=np.float32),((0,0),(1,0)))
        b=(cumulative[:,edges[1:]]-cumulative[:,edges[:-1]])/np.diff(edges)
    if a.shape != (64, frames) or b.shape != a.shape: raise ValueError('Unexpected Mel shape')
    return np.rint(np.concatenate([a,b], axis=1)*255).astype(np.uint8)

def call_features(call, variants):
    waveform, sr = sf.read(io.BytesIO(read_ref(call['wav'])), dtype='float32', always_2d=False)
    if sr != 8000 or waveform.ndim != 1: raise ValueError(f'{call["stem"]}: 8kHz mono WAV가 필요합니다')
    if not np.isfinite(waveform).all(): raise ValueError('비유한 음성 샘플')
    result = {v: [] for v in variants}
    for r in call['rows']:
        start = round(r['start']*8); end = round(r['end']*8)
        if end > len(waveform)+400: raise ValueError(f'{r["uid"]}: WAV 범위를 넘는 발화')
        segment = waveform[start:min(end,len(waveform))]
        if not len(segment): raise ValueError(f'{r["uid"]}: 빈 음성 구간')
        for v in variants: result[v].append(feature(segment,v))
    return {v:np.stack(x) for v,x in result.items()}

class Cache:
    def __init__(self, out, data_signature, local, max_gb=28):
        self.root=Path(out)/'features'; self.signature=data_signature
        self.local=Path(local)/data_signature[:12]; self.local.mkdir(parents=True,exist_ok=True)
        self.max_bytes=max_gb*1024**3
        self.groups = {}
    def path(self, split, variant, group): return self.root/split/variant/f'{group:05d}.npy'
    def valid(self, split, variant, group):
        p=self.path(split,variant,group); m=p.with_suffix('.json')
        return p.exists() and m.exists() and read_json(m).get('signature')==self.signature and p.stat().st_size==read_json(m)['size']
    def put(self, split, variant, group, x):
        p=self.path(split,variant,group); p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix('.npy.partial')
        with open(tmp,'wb') as f: np.save(f,x,allow_pickle=False)
        h=sha(tmp); os.replace(tmp,p)
        atomic_json(p.with_suffix('.json'),dict(signature=self.signature,sha256=h,size=p.stat().st_size,shape=list(x.shape)))
    def get(self, split, variant, group):
        try:
            return self._get(split, variant, group)
        except (OSError, ValueError, KeyError) as error:
            if split not in self.groups:
                raise
            print(f'특징 캐시 복구: {split}/{variant}/{group:05d} ({type(error).__name__})', flush=True)
            outputs = [call_features(call, [variant])[variant] for call in self.groups[split][group]]
            self.put(split, variant, group, np.concatenate(outputs))
            key = f'{split}_{variant}_{group:05d}'
            (self.local / (key + '.npy')).unlink(missing_ok=True)
            (self.local / (key + '.sha')).unlink(missing_ok=True)
            return self._get(split, variant, group)

    def _get(self, split, variant, group):
        p=self.path(split,variant,group); m=read_json(p.with_suffix('.json'))
        key=f'{split}_{variant}_{group:05d}'; q=self.local/(key+'.npy'); tag=q.with_suffix('.sha')
        if not q.exists() or not tag.exists() or tag.read_text()!=m['sha256'] or q.stat().st_size!=m['size']:
            files=sorted(self.local.glob('*.npy'),key=lambda f:f.stat().st_mtime)
            size=sum(f.stat().st_size for f in files)
            while files and size+m['size']>self.max_bytes:
                old=files.pop(0); size-=old.stat().st_size; old.unlink(); old.with_suffix('.sha').unlink(missing_ok=True)
            if shutil.disk_usage(self.local).free < m['size']+1024**3: raise PauseRun('로컬 캐시 디스크 공간 부족. 캐시 경로/디스크 여유 공간 확인 필요')
            tmp=q.with_suffix('.partial'); shutil.copyfile(p,tmp)
            if sha(tmp)!=m['sha256']: raise IOError(f'특징 파일 손상: {p}; 해당 shard 복구 필요')
            os.replace(tmp,q); tag.write_text(m['sha256'])
        os.utime(q,None)
        return np.load(q,allow_pickle=False)

def prepare(cache, split, groups, selected, variants, workers, clock):
    start=time.monotonic(); count=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pos,i in enumerate(selected):
            missing=[v for v in variants if not cache.valid(split,v,i)]
            if missing:
                outputs=list(pool.map(lambda c:call_features(c,missing),groups[i]))
                for v in missing: cache.put(split,v,i,np.concatenate([x[v] for x in outputs]))
                count+=sum(len(c['rows']) for c in groups[i])
            clock.check()
            if missing and (pos%8==0 or pos==len(selected)-1):
                print(f'특징 {split} {pos+1}/{len(selected)}, 이번 실행 {count:,}발화 / {(time.monotonic()-start)/60:.1f}분',flush=True)
    return dict(seconds=time.monotonic()-start, samples=count)

class RoleResNet(nn.Module):
    def __init__(self,pooling='mean',pretrained=False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        old=self.backbone.conv1.weight.detach()
        self.backbone.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        if pretrained:
            with torch.no_grad(): self.backbone.conv1.weight.copy_(F.interpolate(old.reshape(-1,1,7,7),size=3,mode='bilinear',align_corners=False).reshape(64,3,3,3))
        self.backbone.maxpool=nn.Identity()
        self.backbone.fc=nn.Sequential(nn.Dropout(.25),nn.Linear(512,2))
        self.pooling=pooling
        if pooling=='attention':
            self.attention=nn.Sequential(nn.Conv1d(512,128,1),nn.Tanh(),nn.Conv1d(128,1,1))
            nn.init.zeros_(self.attention[-1].weight); nn.init.zeros_(self.attention[-1].bias)
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer('image_std',torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)
    def forward(self,x):
        if self.training:
            x=x.clone(); b,_,f,t=x.shape
            for axis,limit in ((2,6),(3,max(1,round(3*t/48)))):
                length=x.shape[axis]; coord=torch.arange(length,device=x.device)[None,:]
                for _ in range(2):
                    widths=torch.randint(limit+1,(b,1),device=x.device)
                    starts=(torch.rand(b,1,device=x.device)*(length-widths+1)).long()
                    mask=(coord>=starts)&(coord<starts+widths)
                    x=x.masked_fill(mask[:,None,:,None] if axis==2 else mask[:,None,None,:],0)
        x=(x.repeat(1,3,1,1)-self.image_mean)/self.image_std
        bb=self.backbone; x=bb.relu(bb.bn1(bb.conv1(x)))
        x=bb.layer4(bb.layer3(bb.layer2(bb.layer1(x))))
        if self.pooling=='mean': x=x.mean((2,3))
        else:
            x=x.mean(2); w=self.attention(x).softmax(-1); x=(x*w).sum(-1)
        return bb.fc(x)

def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def load_shared(model, state):
    missing,unexpected=model.load_state_dict(state,strict=False)
    if unexpected or any(not k.startswith('attention.') for k in missing):
        raise ValueError(f'초기 체크포인트 구조 불일치: missing={missing}, unexpected={unexpected}')

def weight_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rows_for(group): return [r for c in group for r in c['rows']]

def batches(cache, split, variant, groups, selected, batch_size, seed, epoch=None, start_batch=0):
    order=list(selected); rng=np.random.default_rng(seed+(epoch or 0)*1009)
    if epoch is not None: rng.shuffle(order)
    batch_no=0
    for i in order:
        rows=rows_for(groups[i]); indices=np.arange(len(rows))
        if epoch is not None: rng.shuffle(indices)
        count=math.ceil(len(rows)/batch_size)
        if batch_no+count<=start_batch: batch_no+=count; continue
        x=cache.get(split,variant,i)
        if len(x)!=len(rows): raise ValueError('특징/매니페스트 행 불일치')
        for first in range(0,len(rows),batch_size):
            ids=indices[first:first+batch_size]; number=batch_no; batch_no+=1
            if number<start_batch: continue
            yield number,torch.from_numpy(x[ids].astype(np.float32)/255).unsqueeze(1),torch.tensor([rows[j]['label'] for j in ids]),[rows[j] for j in ids]

def metric(rows, probs):
    y=np.array([r['label'] for r in rows]); pred=np.asarray(probs)>=.5
    cm=[[int(((y==a)&(pred==b)).sum()) for b in (0,1)] for a in (0,1)]
    def acc(mask): return float((pred[mask]==y[mask]).mean()) if mask.any() else None
    duration=np.array([r['duration'] for r in rows]); groups=[]
    for a,b in [(0,.5),(.5,1),(1,1.5),(1.5,3),(3,6),(6,float('inf'))]:
        for role in (None,0,1):
            m=(duration>=a)&(duration<b)
            if role is not None: m&=y==role
            groups.append(dict(seconds=f'{a:g}-{b:g}',role=role,n=int(m.sum()),errors=int((pred[m]!=y[m]).sum()),accuracy=acc(m)))
    return dict(samples=len(y),accuracy=float((pred==y).mean()),threshold=.5,confusion=cm,
                operator_recall=acc(y==0),caller_recall=acc(y==1),by_duration_role=groups)

@torch.inference_mode()
def evaluate(model, cache, groups, selected, variant, batch_size, root, clock, device):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    signature=digest([weight_hash(model),cache.signature,variant,selected]); model.eval()
    all_rows=[]; probs=[]
    for i in selected:
        rows=rows_for(groups[i]); p=root/f'{i:05d}.json'
        old=read_json(p) if p.exists() else None
        if old and old.get('signature')==signature and len(old.get('p',[]))==len(rows): values=old['p']
        else:
            values=[]
            for _,x,_,_ in batches(cache,'Validation',variant,groups,[i],batch_size,0):
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                    v=model(x.to(device)).float().softmax(1)[:,1]
                values.extend(v.cpu().tolist())
            if not np.isfinite(values).all(): raise ValueError('비유한 예측 확률')
            atomic_json(p,dict(signature=signature,p=values))
        all_rows.extend(rows); probs.extend(values); clock.check()
    result=metric(all_rows,probs); atomic_json(root/'metrics.json',result)
    return result,all_rows,np.array(probs)

def train_run(name,variant,pooling,initial,train_groups,valid_groups,train_ids,valid_ids,
              cache,plan,out,cfg,clock,device):
    root=Path(out)/'runs'/name
    signature=digest([cache.signature,name,variant,pooling,plan,train_ids,valid_ids,cfg['seed'],cfg['lr'],VERSION])
    store=Store(root/'resume',signature); state=store.load()
    if state and state['done']: return state
    seed_all(cfg['seed']); model=RoleResNet(pooling); load_shared(model,initial); model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    steps=sum(math.ceil(len(rows_for(train_groups[i]))/plan['batch']) for i in train_ids)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,steps*plan['epochs']),eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    if state:
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler']); restore_rng(state['rng'])
    else:
        state=dict(epoch=0,next_batch=0,done=False,history=[],best_accuracy=-1,best_model=None,updates=0)
    last_save=time.monotonic()
    def save():
        nonlocal last_save
        state.update(model=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state())
        store.save(state); clock.save(); last_save=time.monotonic()
    if store.seq < 0:
        save()  # A safe rollback point even if the first optimizer step is interrupted.
    try:
        for epoch in range(state['epoch'],plan['epochs']):
            model.train()
            for number,x,y,_ in batches(cache,'Training',variant,train_groups,train_ids,plan['batch'],cfg['seed'],epoch,state['next_batch']):
                clock.check(); x=x.to(device); y=y.to(device)
                lam=float(np.random.beta(.1,.1)); permutation=torch.randperm(len(y),device=device)
                mixed=lam*x+(1-lam)*x[permutation]; optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                    logits=model(mixed)
                    loss=lam*F.cross_entropy(logits,y,label_smoothing=.03)+(1-lam)*F.cross_entropy(logits,y[permutation],label_smoothing=.03)
                if not torch.isfinite(loss): raise FloatingPointError('non-finite loss')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),5)
                scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=scale: scheduler.step()
                state['updates']+=1; state['next_batch']=number+1
                if time.monotonic()-last_save>=cfg['checkpoint_seconds']: save()
                if number%100==0: print(f'{name} epoch {epoch+1}/{plan["epochs"]} batch {number+1}/{steps} loss={loss.item():.4f}, 남은 예산 {clock.remaining/60:.1f}분',flush=True)
            save()  # Commit the exact weights before resumable validation.
            result,_,_=evaluate(model,cache,valid_groups,valid_ids,variant,plan['batch'],root/f'eval_{epoch:02d}',clock,device)
            state['history'].append(dict(epoch=epoch+1,**result))
            if result['accuracy']>state['best_accuracy']:
                state['best_accuracy']=result['accuracy']; state['best_model']=cpu_state(model)
            state['epoch']=epoch+1; state['next_batch']=0; save()
            print(f'{name}: Accuracy@0.5={result["accuracy"]:.6f}',flush=True)
        state['done']=True; save(); return state
    except PauseRun:
        save(); raise
    except KeyboardInterrupt:
        # An interrupt may land inside optimizer.step(); do not commit partial updates.
        raise PauseRun('사용자 중단. 마지막 안전한 주기 저장본부터 재개합니다. 같은 설정으로 다시 실행하세요.')
    finally:
        del model,optimizer,scheduler,scaler; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def export_predictions(path,rows,probs):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.partial')
    with open(tmp,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(['uid','stem','startAt','endAt','duration','speaker','p_caller','prediction','correct'])
        for r,p in zip(rows,probs): writer.writerow([r['uid'],r['stem'],r['start'],r['end'],r['duration'],r['label'],float(p),int(p>=.5),int((p>=.5)==r['label'])])
    os.replace(tmp,path)
    tmp=path.with_suffix('.npz.partial')
    with open(tmp,'wb') as f:
        np.savez_compressed(f,uid=np.array([r['uid'] for r in rows]),stem=np.array([r['stem'] for r in rows]),
            start_ms=np.array([r['start'] for r in rows]),end_ms=np.array([r['end'] for r in rows]),
            duration=np.array([r['duration'] for r in rows]),y_true=np.array([r['label'] for r in rows]),p_caller=probs)
    os.replace(tmp,path.with_suffix('.npz'))

def audit_samples(path,rows,probs,seed):
    # Listening requires a human; never invent a cause from confidence or duration.
    rng=np.random.default_rng(seed); chosen=[]
    for role in (0,1):
        for a,b in [(0,.5),(.5,1.5),(1.5,3),(3,float('inf'))]:
            for correct in (False,True):
                ids=[i for i,r in enumerate(rows) if r['label']==role and a<=r['duration']<b and ((probs[i]>=.5)==r['label'])==correct]
                if ids: chosen.extend(rng.choice(ids,min(5,len(ids)),replace=False).tolist())
    export_predictions(path,[rows[i] for i in chosen],probs[chosen])
    # These blank fields are intentionally left for human listening annotations.
    with open(path,encoding='utf-8-sig') as f: records=list(csv.DictReader(f))
    fields=list(records[0]) if records else ['uid']
    fields+=['boundary_issue','overlap','noise_or_silence','ambiguous_role','notes']
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)

def paired_report(rows,new,old,seed):
    y=np.array([r['label'] for r in rows]); good=(new>=.5)==y; prev=(old>=.5)==y
    stems,idx=np.unique([r['stem'] for r in rows],return_inverse=True)
    sizes=np.bincount(idx); delta=np.bincount(idx,weights=good.astype(int)-prev.astype(int))
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(1000):
        take=rng.integers(len(stems),size=len(stems)); samples.append(delta[take].sum()/sizes[take].sum())
    return dict(fixed=int((good&~prev).sum()),regressed=int((~good&prev).sum()),both_wrong=int((~good&~prev).sum()),
                accuracy_delta=float(good.mean()-prev.mean()),call_bootstrap_95pct=np.quantile(samples,[.025,.975]).tolist(),
                note='통화 단위 paired bootstrap, seed 변동/Validation 모델 선택 편향을 포함하지 않는 탐색적 구간')

def benchmark(initial,cache,train_groups,train_ids,cfg,device,clock):
    timing={}; batch=cfg.get('batch_size',128)
    # Highest resolution + attention first; all conditions use the resulting common batch size.
    while True:
        model=None
        try:
            seed_all(cfg['seed']); model=RoleResNet('attention'); load_shared(model,initial); model.to(device).train()
            sample=torch.from_numpy(cache.get('Training','fine',train_ids[0])[:batch].astype(np.float32)/255).unsqueeze(1)
            if len(sample)<batch: sample=sample.repeat(math.ceil(batch/len(sample)),1,1,1)[:batch]
            sample=sample.to(device); y=torch.zeros(batch,dtype=torch.long,device=device)
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16): loss=F.cross_entropy(model(sample),y)
                loss.backward()
            del sample,y,loss,model; model=None; torch.cuda.empty_cache(); break
        except torch.cuda.OutOfMemoryError:
            del model; gc.collect(); torch.cuda.empty_cache(); batch//=2
            if batch<16: raise PauseRun('L4 메모리 부족. 다른 GPU 작업/런타임 상태를 확인하세요.')
    for name,variant,pool in CONDITIONS:
        seed_all(cfg['seed']); model=RoleResNet(pool); load_shared(model,initial); model.to(device).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
        sample=torch.from_numpy(cache.get('Training',variant,train_ids[0])[:batch].astype(np.float32)/255).unsqueeze(1)
        if len(sample)<batch: sample=sample.repeat(math.ceil(batch/len(sample)),1,1,1)[:batch]
        x=sample.to(device); y=torch.zeros(batch,dtype=torch.long,device=device)
        seconds=[]
        for j in range(5):
            torch.cuda.synchronize(); started=time.monotonic(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): loss=F.cross_entropy(model(x),y)
            loss.backward(); optimizer.step(); torch.cuda.synchronize()
            if j: seconds.append(time.monotonic()-started)
        timing[name]=float(np.median(seconds)); clock.check()
        del model,optimizer,x,y,loss,sample; gc.collect(); torch.cuda.empty_cache()
    return batch,timing

def resolve_roots(cfg):
    base=Path(cfg['data_root']) if cfg.get('data_root') else None
    answer={}
    for split,key in [('Training','train_root'),('Validation','valid_root')]:
        if cfg.get(key): answer[split]=str(Path(cfg[key])); continue
        if base is None or not base.exists():
            raise ValueError('--data-root 또는 --train-root/--valid-root로 원본 WAV+JSON 경로를 지정하세요.')
        if (base/split).exists(): answer[split]=str(base/split); continue
        matches=[]; frontier=[base]
        for _ in range(3):
            nxt=[]
            for p in frontier:
                for c in p.iterdir():
                    if c.is_dir():
                        if c.name.lower()==split.lower(): matches.append(c)
                        else: nxt.append(c)
            frontier=nxt
        if len(matches)!=1: raise ValueError(f'{split} 폴더를 하나로 결정할 수 없습니다. TRAIN_ROOT / VALID_ROOT를 직접 지정하세요.')
        answer[split]=str(matches[0])
    if answer['Training']==answer['Validation']: raise ValueError('Training과 Validation 경로가 같습니다')
    return answer

def budget_epochs(remaining, one_epoch, preferred_minutes, reserve_minutes, max_epochs):
    """A stage target is soft; never reject a feasible first epoch solely on its target."""
    available=max(0.0, remaining-reserve_minutes*60)
    if not math.isfinite(one_epoch) or one_epoch<=0:
        raise ValueError('처리 시간 측정이 유효하지 않습니다')
    if available<one_epoch: return 0,available
    target=min(preferred_minutes*60,available)
    return min(max_epochs,max(1,int(target/one_epoch))),available

def compatible_budget_update(saved,current):
    old=dict(saved); new=dict(current)
    old_hash=old.pop('runner_sha256',None); new.pop('runner_sha256',None)
    return old==new and old_hash in COMPATIBLE_RUNNER_HASHES

def runtime_device():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU가 필요합니다. NVIDIA 드라이버와 CUDA용 PyTorch 설치를 확인하세요.')
    return torch.device('cuda')

def run(cfg):
    out=Path(cfg['output_root'])/cfg['run_name']; out.mkdir(parents=True,exist_ok=True)
    # User may extend only the time budget when resuming. Model/data changes need a new run name.
    immutable={k:v for k,v in cfg.items() if k not in ('max_minutes','extra_minutes','local_cache','checkpoint_seconds')}
    immutable['runner_sha256']=sha(__file__)
    lock=out/'config.json'
    if lock.exists() and read_json(lock)!=immutable:
        saved=read_json(lock)
        if not compatible_budget_update(saved,immutable):
            raise ValueError('기존 RUN_NAME의 설정/코드와 다릅니다. 원래 설정으로 복원하거나 새 RUN_NAME을 사용하세요.')
        atomic_json(out/'budget_update.json',dict(previous_runner=saved['runner_sha256'],current_runner=immutable['runner_sha256'],
            note='시간 배정·특징 캐시 복구만 갱신. 데이터/가중치/학습 조건/기존 고정 계획은 유지.'))
        print('재개·캐시 복구 수정본 적용: 기존 체크포인트와 특징을 유지합니다.',flush=True)
    atomic_json(lock,immutable)
    if (out/'COMPLETE.json').exists():
        print('이미 완료된 실험입니다. 결과:',out/'final_report.json'); return read_json(out/'final_report.json')
    clock=Clock(out,cfg['max_minutes']+cfg.get('extra_minutes',0))
    try:
        clock.check()
        device=runtime_device(); torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=True
        hardware=torch.cuda.get_device_name() if device.type=='cuda' else str(device)
        print('GPU:',hardware,'/ 남은 누적 예산:',round(clock.remaining/60,1),'분',flush=True)
        versions={'python':sys.version,'torch':torch.__version__,'gpu':hardware}
        import librosa, torchvision
        versions.update(librosa=librosa.__version__,torchvision=torchvision.__version__,numpy=np.__version__)
        environment=out/'environment.json'
        if environment.exists():
            oldenv=read_json(environment)
            for k in ('torch','torchvision','librosa','numpy'):
                if oldenv.get(k)!=versions[k]: raise RuntimeError(f'재개 환경의 {k} 버전이 달라졌습니다. environment.json에 기록된 버전으로 복원하세요.')
        else: atomic_json(environment,versions)
        roots=resolve_roots(cfg)
        train,ts=make_manifest(roots['Training'],'Training',out,clock)
        valid,vs=make_manifest(roots['Validation'],'Validation',out,clock)
        if {c['stem'] for c in train}&{c['stem'] for c in valid}: raise ValueError('Training/Validation 통화 ID 중복 발견')
        data_signature=digest([ts,vs,VARIANTS,VERSION]); groups={'Training':make_groups(train),'Validation':make_groups(valid)}
        tids=select_groups(groups['Training'],cfg['screen_train_samples'],cfg['seed'])
        vids=select_groups(groups['Validation'],cfg['screen_valid_samples'],cfg['seed']+1)
        for split,ids in [('Training',tids),('Validation',vids)]:
            labels=[r['label'] for i in ids for r in rows_for(groups[split][i])]
            if set(labels)!={0,1}: raise ValueError(f'{split} 표본에 두 클래스가 모두 있어야 합니다')
        atomic_json(out/'selection.json',dict(data_signature=data_signature,train_groups=tids,valid_groups=vids,
            train_samples=sum(len(rows_for(groups['Training'][i])) for i in tids),
            valid_samples=sum(len(rows_for(groups['Validation'][i])) for i in vids)))
        cache=Cache(out,data_signature,cfg['local_cache'])
        cache.groups=groups
        seed_all(cfg['seed'])
        init_store=Store(out/'shared_initial',digest([data_signature,cfg.get('initial_checkpoint'),cfg['seed'],VERSION]))
        init=init_store.load()
        if init is None:
            path=cfg.get('initial_checkpoint','')
            if path:
                p=Path(path)
                if not p.is_file(): raise FileNotFoundError(f'기존 ResNet18 체크포인트를 찾지 못했습니다: {p}')
                c=torch.load(p,map_location='cpu',weights_only=True)
                if c.get('model_name')!='resnet18_smallstem' or c.get('feature_variant')!='first_plus_whole':
                    raise ValueError('초기 가중치는 resnet18_smallstem + first_plus_whole 체크포인트여야 합니다. 초기 first_1p5/ResNet34 파일과 구분하세요.')
                m=RoleResNet(); load_shared(m,c['model_state_dict'])
                init={'model':cpu_state(m),'source':'existing_resnet18','source_sha256':sha(p)}
            else:
                m=RoleResNet(pretrained=True); init={'model':cpu_state(m),'source':'ImageNet; short screening may undertrain','source_sha256':None}
                print('기존 가중치 미지정: ImageNet에서 시작합니다. 5시간 내 충분한 수렴은 보장되지 않습니다.',flush=True)
            init_store.save(init); del m
        # Warm up librosa JIT before parallel feature extraction.
        feature(np.zeros(12000,dtype=np.float32),'base'); feature(np.zeros(12000,dtype=np.float32),'fine')
        plan_path=out/'screen_plan.json'
        if not plan_path.exists():
            pilot_ids=tids[:min(2,len(tids))]
            pilot=prepare(cache,'Training',groups['Training'],pilot_ids,['base','fine'],cfg['workers'],clock)
            batch,timing=benchmark(init['model'],cache,groups['Training'],tids,cfg,device,clock)
            steps=sum(math.ceil(len(rows_for(groups['Training'][i]))/batch) for i in tids)
            valid_steps=sum(math.ceil(len(rows_for(groups['Validation'][i]))/batch) for i in vids)
            one_epoch=sum(timing.values())*(steps+valid_steps*.6)*1.4
            epochs,available=budget_epochs(clock.remaining,one_epoch,cfg['screen_minutes'],110,cfg['screen_max_epochs'])
            print(f'4조건 각각 1 epoch 예상 {one_epoch/60:.1f}분 / 비교에 사용 가능 {available/60:.1f}분 / 후속 예약 110분',flush=True)
            if epochs<1:
                shortage=math.ceil((one_epoch-available)/60)+10
                raise PauseRun(f'4조건 각각 1 epoch에 약 {one_epoch/60:.1f}분이 필요하지만 비교 예산은 {available/60:.1f}분입니다. '
                    f'현재 EXTRA_MINUTES 값에 최소 {shortage}분을 더하세요. 준비 단계 시간을 제외한 추정이며 저장된 데이터는 유지됩니다.')
            feature_rate=pilot['samples']/max(pilot['seconds'],1e-6) if pilot['samples'] else None
            screen_n=sum(len(rows_for(groups['Training'][i])) for i in tids)+sum(len(rows_for(groups['Validation'][i])) for i in vids)
            if feature_rate:
                print(f'두 Mel의 표본 전처리 예상: 약 {screen_n/feature_rate/60:.1f}분 (Drive I/O 변동 가능)',flush=True)
            plan=dict(batch=batch,epochs=epochs,step_seconds=timing,estimated_screen_minutes=one_epoch*epochs/60,
                      pilot_feature_seconds=pilot['seconds'],pilot_feature_samples=pilot['samples'])
            atomic_json(plan_path,plan)
        plan=read_json(plan_path); print('고정 비교 계획:',plan,flush=True)
        prepare(cache,'Training',groups['Training'],tids,['base','fine'],cfg['workers'],clock)
        prepare(cache,'Validation',groups['Validation'],vids,['base','fine'],cfg['workers'],clock)
        states={}; comparison=[]
        for name,variant,pool in CONDITIONS:
            clock.check()
            result=train_run(name,variant,pool,init['model'],groups['Training'],groups['Validation'],tids,vids,cache,plan,out,cfg,clock,device)
            states[name]=result
            comparison.append(dict(condition=name,feature=variant,pooling=pool,accuracy=result['best_accuracy'],epochs=result['epoch']))
            atomic_json(out/'screen_comparison.json',comparison)
        # A tie retains the simpler A baseline, rather than claiming an improvement.
        winner=max(comparison,key=lambda x:x['accuracy']); name=winner['condition']
        effects={r['condition']:r['accuracy'] for r in comparison}
        atomic_json(out/'screen_effects.json',dict(frontend_with_mean=effects['B']-effects['A'],
            attention_with_base=effects['C']-effects['A'],interaction=effects['D']-effects['B']-effects['C']+effects['A'],
            note='동일 seed·표본·epoch의 짧은 선별 결과. 통계적 유의성 또는 최종 성능 증거가 아님'))
        all_train=list(range(len(groups['Training']))); all_valid=list(range(len(groups['Validation'])))
        final_plan_path=out/'final_plan.json'; variant=winner['feature']; pool=winner['pooling']
        if name=='A' and init['source']=='existing_resnet18':
            print('개선 후보가 대조군을 넘지 못했습니다. 추가 학습을 생략하고 보관된 기존 ResNet18을 전체 평가합니다.',flush=True)
            final_model=init['model']; variant='base'; pool='mean'; final_status='no_screen_improvement'
        else:
            prepare(cache,'Training',groups['Training'],all_train,[variant],cfg['workers'],clock)
            prepare(cache,'Validation',groups['Validation'],all_valid,[variant],cfg['workers'],clock)
            if not final_plan_path.exists():
                full_steps=sum(math.ceil(len(rows_for(groups['Training'][i]))/plan['batch']) for i in all_train)
                eval_steps=sum(math.ceil(len(rows_for(groups['Validation'][i]))/plan['batch']) for i in all_valid)
                epoch_seconds=plan['step_seconds'][name]*(full_steps+.6*eval_steps)*1.4
                epochs,available=budget_epochs(clock.remaining,epoch_seconds,cfg['final_minutes'],25,cfg['final_max_epochs'])
                print(f'전체 학습 1 epoch 예상 {epoch_seconds/60:.1f}분 / 학습에 사용 가능 {available/60:.1f}분 / 평가 예약 25분',flush=True)
                if epochs<1:
                    shortage=math.ceil((epoch_seconds-available)/60)+10
                    raise PauseRun(f'전체 학습 1 epoch 예상 {epoch_seconds/60:.1f}분, 사용 가능 {available/60:.1f}분. '
                        f'현재 EXTRA_MINUTES 값에 최소 {shortage}분을 더하고 재개하세요. 특징은 보존됐습니다.')
                atomic_json(final_plan_path,dict(batch=plan['batch'],epochs=epochs,winner=name))
            fp=read_json(final_plan_path)
            result=train_run('final_'+name,variant,pool,states[name]['best_model'],groups['Training'],groups['Validation'],all_train,all_valid,cache,fp,out,cfg,clock,device)
            final_model=result['best_model']; final_status='trained_candidate'
        prepare(cache,'Validation',groups['Validation'],all_valid,[variant],cfg['workers'],clock)
        model=RoleResNet(pool); model.load_state_dict(final_model); model.to(device)
        metrics,rows,probs=evaluate(model,cache,groups['Validation'],all_valid,variant,plan['batch'],out/'final_eval',clock,device)
        export_predictions(out/'predictions.csv',rows,probs); audit_samples(out/'listening_audit.csv',rows,probs,cfg['seed'])
        del model; gc.collect(); torch.cuda.empty_cache()
        baseline=None; paired=None
        if init['source']=='existing_resnet18':
            prepare(cache,'Validation',groups['Validation'],all_valid,['base'],cfg['workers'],clock)
            model=RoleResNet(); model.load_state_dict(init['model']); model.to(device)
            baseline,brows,bprobs=evaluate(model,cache,groups['Validation'],all_valid,'base',plan['batch'],out/'baseline_eval',clock,device)
            if [r['uid'] for r in brows]!=[r['uid'] for r in rows]: raise ValueError('최종/기준 모델 평가 행 순서 불일치')
            export_predictions(out/'baseline_predictions.csv',brows,bprobs); paired=paired_report(rows,probs,bprobs,cfg['seed'])
            del model; gc.collect(); torch.cuda.empty_cache()
        deploy=Store(out/'best_model',digest([data_signature,variant,pool,VERSION]))
        deploy.load()
        deploy.save(dict(model_state_dict=final_model,model_name='resnet18_smallstem_resume',feature_variant=variant,
            pooling=pool,preprocessing=VARIANTS[variant],threshold=.5,metrics=metrics,source=VERSION))
        report=dict(status=final_status,winner=winner,screening=comparison,final=metrics,measured_resnet18_baseline=baseline,
            paired=paired,reported_ensemble_reference=dict(accuracy=.91419154,samples=111947,independently_rerun=False),
            reference_comparison_validated=False,
            recommendation=('retain_existing_resnet18' if baseline and metrics['accuracy']<=baseline['accuracy'] else
                'improved_over_measured_single_resnet18_only' if baseline else 'no_measured_trained_baseline'),
            seconds=clock.elapsed,notes=['원래 ResNet18+34 앙상블의 직접 재평가는 포함하지 않음',
                '기존 111947개 평가와 정확히 같은 데이터인지는 행 ID 대조 전까지 보장하지 않음',
                '동일 Validation으로 선별한 결과이며 독립 Test 성능이 아님',
                '음성 경계/겹침 원인은 listening_audit.csv의 사람 청취가 필요함'])
        atomic_json(out/'final_report.json',report)
        atomic_json(out/'COMPLETE.json',dict(report='final_report.json',seconds=clock.elapsed))
        clock.save(); print('완료. 결과 폴더:',out,flush=True)
        print('최종 Accuracy@0.5:',metrics['accuracy'],'/ 샘플:',metrics['samples'],flush=True)
        return report
    except PauseRun as e:
        clock.save(); atomic_json(out/'PAUSED.json',dict(reason=str(e),seconds=clock.elapsed))
        print('\n[저장 후 일시 중지]',e,'\n결과 폴더:',out,flush=True)
        return {'status':'paused','reason':str(e)}
    except KeyboardInterrupt:
        clock.save(); print('중단됨. 마지막 저장본부터 재개합니다:',out,flush=True)
        return {'status':'interrupted'}
    finally:
        clock.save()

# 실행 전: CUDA 버전에 맞는 torch/torchvision을 설치하고 아래 패키지를 설치하세요.
# python -m pip install numpy soundfile==0.13.1 librosa==0.11.0
# 실행 예시 (Training/Validation 하위 폴더가 있는 원본 WAV+JSON 폴더):
# python mission2_experiment.py --data-root /data/mission2 --output-root /results/mission2 --initial-checkpoint /models/best_model.pt --minutes 300
# 재개: 같은 --output-root와 --run-name으로 실행. --minutes는 이번 실행에 추가로 허용할 시간.
# python mission2_experiment.py --output-root /results/mission2 --minutes 180
# 기존 초기 가중치가 없다면 --initial-checkpoint 생략: ImageNet 다운로드에 인터넷 필요.
# 4조건: 기존 Mel/미세 Mel × 평균 pooling/attention. 표본 비교 후 선정 모델 전체 학습.
# 결과: final_report.json, predictions.npz, listening_audit.csv, best_model/.
# NVIDIA CUDA GPU 필요. CPU 학습은 지원하지 않음. 동일 결과 폴더에 동시 실행 금지.
# Colab 저장본을 다른 PC로 옮기는 경로/패키지 마이그레이션은 자동 지원하지 않음.

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Mission 2 ResNet 4조건 비교 및 체크포인트 재개')
    parser.add_argument('--data-root', help='Training/Validation 하위 폴더를 포함한 원본 데이터 경로')
    parser.add_argument('--train-root', help='Training 폴더 또는 WAV+JSON ZIP 경로')
    parser.add_argument('--valid-root', help='Validation 폴더 또는 WAV+JSON ZIP 경로')
    parser.add_argument('--output-root', required=True, help='영구 저장 디스크의 결과 상위 폴더')
    parser.add_argument('--run-name', default='resnet_4way_v1')
    parser.add_argument('--initial-checkpoint', help='기존 resnet18_smallstem/first_plus_whole best_model.pt (선택)')
    parser.add_argument('--local-cache', help='빠른 SSD 캐시 경로 (선택, 기본: 결과 폴더의 .cache)')
    parser.add_argument('--minutes', type=int, default=300, help='이번 실행 추가 허용 시간, 기본 300분')
    args = parser.parse_args(argv)
    if args.minutes <= 0 or not args.run_name or '/' in args.run_name or '\\' in args.run_name or '..' in args.run_name:
        parser.error('minutes는 양수, run-name은 경로 기호 없는 이름이어야 합니다.')
    def absolute(value):
        return str(Path(value).expanduser().resolve()) if value else ''
    output = absolute(args.output_root)
    folder = Path(output) / args.run_name
    saved = folder / 'config.json'
    if saved.exists():
        cfg = read_json(saved)
        cfg.pop('runner_sha256', None)
        if cfg['output_root'] != output or cfg['run_name'] != args.run_name:
            parser.error('저장본의 결과 경로와 다릅니다. 기존 경로를 사용하세요.')
        for key in ('data_root', 'train_root', 'valid_root', 'initial_checkpoint'):
            supplied = getattr(args, key)
            if supplied is not None and absolute(supplied) != cfg.get(key, ''):
                parser.error(f'재개 시 {key} 변경 불가. 기존 경로를 사용하거나 새 run-name으로 실행하세요.')
        state = Store(folder/'time_state', 'time-v1').load()
        if state is None:
            parser.error('누적 시간 저장본이 없습니다. 결과 폴더의 체크포인트를 확인하세요.')
        used = float(state['seconds'])
        cfg.update(max_minutes=math.ceil(used/60)+args.minutes, extra_minutes=0)
        print(f'재개: 저장된 누적 {used/60:.1f}분 / 이번 추가 {args.minutes}분', flush=True)
    else:
        if not args.data_root and not (args.train_root and args.valid_root):
            parser.error('새 실험에는 --data-root 또는 --train-root와 --valid-root가 필요합니다.')
        initial = absolute(args.initial_checkpoint)
        if initial and not Path(initial).is_file():
            parser.error('initial-checkpoint 파일을 찾지 못했습니다.')
        cfg = dict(data_root=absolute(args.data_root), train_root=absolute(args.train_root),
            valid_root=absolute(args.valid_root), output_root=output, run_name=args.run_name,
            initial_checkpoint=initial, max_minutes=args.minutes, extra_minutes=0,
            checkpoint_seconds=180, seed=42, screen_train_samples=100_000,
            screen_valid_samples=20_000, screen_max_epochs=3, final_max_epochs=4,
            screen_minutes=80, final_minutes=100, workers=2, batch_size=128,
            lr=1e-5 if initial else 1e-4)
        for path in resolve_roots(cfg).values():
            if not Path(path).exists():
                parser.error(f'원본 데이터 경로 없음: {path}')
    cfg['local_cache'] = absolute(args.local_cache) if args.local_cache else str(Path(output)/'.cache')
    cfg['checkpoint_seconds'] = 180
    result = run(cfg)
    if result.get('status') in ('paused', 'interrupted'):
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
