from __future__ import annotations
import copy,json,time,math,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ilrom_basic.case1 import CaseData,PODCurveRepresentation,SensorLayout,U_COL_FRAC,U_HEIGHTS,fixed_layouts,regression_metrics,save_json,seed_everything,FieldCache,SPLITS
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from ilrom_basic import case1 as vc
from .deeponet import DeepONet1D

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/case1/vidon'
OUTPUT_OVERRIDE=None
def budget(case,seed):
    return read_json(ROOT/'configs/case1_forced_convection/vidon.json')
def dir_for(case,seed,smoke):
    if OUTPUT_OVERRIDE is not None:return Path(OUTPUT_OVERRIDE)
    return (OUT/'smoke' if smoke else OUT)/f'seed_{seed}'
def protocol_sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/baselines/deeponet.py',ROOT/'src/ilrom_basic/case1.py']}
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def save_pt(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Data:

    def __init__(self, case, seed, device):
        (self.case, self.seed, self.device) = (case, seed, torch.device(device))
        source = vc.CaseData('case2')
        self.source = source
        self.indices = {k: np.asarray(v) for (k, v) in vc.SPLITS.items()}
        self.layouts = vc.fixed_layouts('case2')
        self.means = np.float32([source.t_mean, source.u_mean])
        self.stds = np.float32([source.t_std, source.u_std])
        self.fields_out = np.asarray(source.flux, dtype=np.float32)
        self.frequencies = (1, 2, 3, 4, 6, 8, 12, 16)
        train = self.truth('train')
        self.shape = train.shape[1:]
        (self.y_mean, self.y_std) = (float(train.mean()), float(train.std() + 1e-08))
        self.n_mod = len(self.means)

    def truth(self, split):
        return np.asarray(self.fields_out[self.indices[split]], dtype=np.float32)

    def random_layout(self, rng):
        return self.source.random_layout(rng)

    def arrays(self, split, local, layout, level=0.0, repeat=20261111):
        ids = self.indices[split][np.asarray(local, dtype=int)]
        grid = self.source.grid
        xy = np.column_stack((layout.row_frac, layout.col_frac))
        cells = np.clip(np.rint(xy * (grid - 1)).astype(int), 0, grid - 1)
        raw = self.source.temperature[ids[:, None], cells[:, 0], cells[:, 1]]
        mod = np.zeros(len(cells), dtype=np.int64)
        if layout.with_velocity:
            uxy = np.column_stack((vc.U_HEIGHTS, np.full_like(vc.U_HEIGHTS, vc.U_COL_FRAC)))
            velocity = 6.0 * self.source.umean[ids, None] * vc.U_HEIGHTS[None, :] * (1 - vc.U_HEIGHTS[None, :])
            cells = np.vstack((cells, np.rint(uxy * (grid - 1)).astype(int)))
            xy = np.vstack((xy, uxy))
            mod = np.concatenate((mod, np.ones(len(uxy), dtype=np.int64)))
            raw = np.concatenate((raw, velocity), axis=1)
        raw = np.asarray(raw, dtype=np.float32)
        if level:
            scales = self.stds
            raise ValueError('This release entry only evaluates clean observations')
        raw = (raw - self.means[mod]) / self.stds[mod]
        tokens = np.empty((*raw.shape, 3), dtype=np.float32)
        tokens[:, :, :2] = xy
        tokens[:, :, 2] = raw
        return (tokens, np.broadcast_to(mod, raw.shape).astype(np.int64, copy=True))

    def tokens(self, split, local, layout, level=0.0, repeat=20261111):
        (x, m) = self.arrays(split, local, layout, level, repeat)
        return (torch.as_tensor(x, device=self.device), torch.as_tensor(m, device=self.device), torch.zeros(m.shape, dtype=torch.bool, device=self.device))

    def batch_individual_layouts(self, local, rng):
        rows = [self.arrays('train', [i], self.random_layout(rng)) for i in local]
        length = max((m.shape[1] for (_, m) in rows))
        x = np.zeros((len(rows), length, 3), dtype=np.float32)
        m = np.zeros((len(rows), length), dtype=np.int64)
        mask = np.ones_like(m, dtype=bool)
        for (i, (values, mids)) in enumerate(rows):
            n = mids.shape[1]
            (x[i, :n], m[i, :n], mask[i, :n]) = (values[0], mids[0], False)
        return tuple((torch.as_tensor(v, device=self.device) for v in (x, m, mask)))

def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))

class SetBranch(nn.Module):

    def __init__(self, modalities):
        super().__init__()
        self.modalities = modalities
        self.coord = mlp(2 + modalities, 128, 128)
        self.value = mlp(1, 128, 128)
        self.scores = nn.ModuleList((mlp(128, 32, 1) for _ in range(4)))
        self.values = nn.ModuleList((mlp(128, 32, 32) for _ in range(4)))
        self.combine = mlp(128, 128, 128)

    def forward(self, tokens, modality, mask):
        onehot = F.one_hot(modality, self.modalities).to(tokens.dtype)
        enc = self.coord(torch.cat((tokens[:, :, :2], onehot), dim=-1)) + self.value(tokens[:, :, 2:])
        heads = []
        for (score, value) in zip(self.scores, self.values):
            weights = (score(enc).squeeze(-1) / math.sqrt(128)).masked_fill(mask, -torch.inf).softmax(dim=1)
            heads.append((weights[:, :, None] * value(enc)).sum(dim=1))
        return self.combine(torch.cat(heads, dim=-1))

class VIDON(nn.Module):

    def __init__(self, data):
        super().__init__()
        original = DeepONet1D(1, data.shape[0])
        grid = original.coords
        self.trunk = original.trunk
        self.bias = original.bias
        self.scale = original.scale
        self.branch = SetBranch(data.n_mod)
        (self.y_mean, self.y_std) = (data.y_mean, data.y_std)
        features = [grid]
        mult = 2 * math.pi
        for freq in original.frequencies:
            for v in grid.split(1, dim=1):
                features.extend([torch.sin(mult * freq * v), torch.cos(mult * freq * v)])
        self.register_buffer('features', torch.cat(features, dim=1), persistent=False)

    def basis(self, query=None):
        return self.trunk(self.features if query is None else self.features[query])

    def decode(self, inputs, basis):
        return self.branch(*inputs) @ basis.T / self.scale + self.bias

    def forward(self, tokens, modality, mask, query=None):
        return self.decode((tokens, modality, mask), self.basis(query))

@torch.inference_mode()
def predict(data, model, split, layout, batch=32, level=0.0, repeat=20261111, limit=None, basis=None):
    model.eval()
    n = len(data.indices[split]) if limit is None else min(len(data.indices[split]), limit)
    basis = model.basis() if basis is None else basis
    out = []
    for start in range(0, n, batch):
        token = data.tokens(split, np.arange(start, min(start + batch, n)), layout, level, repeat)
        y = model.decode(token, basis) * model.y_std + model.y_mean
        out.append(y.reshape(-1, *data.shape).cpu().numpy())
    return np.concatenate(out)

def stats(data):
    return dict(mean=data.means.tolist(), std=data.stds.tolist(), y_mean=data.y_mean, y_std=data.y_std, frequencies=list(data.frequencies), output_shape=list(data.shape), modalities=['temperature', 'velocity'])

def load_run(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    data = Data(checkpoint['case'], checkpoint['seed'], device)
    if stats(data) != checkpoint['stats']:
        raise ValueError('Source normalization, mesh or frequency specification changed')
    model = VIDON(data).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return (data, model.eval())

def stamp():
    return datetime.now(timezone.utc).isoformat()

@torch.inference_mode()
def validate(data, model):
    model.eval()
    truth = data.truth('val')
    basis = model.basis()
    scores = {}
    for (name, layout) in data.layouts.items():
        pred = predict(data, model, 'val', layout, basis=basis)
        scores[name] = vc.regression_metrics(pred, truth)['rel_l2']
    return (float(np.mean(list(scores.values()))), scores)

def inputs_check(data, model):
    (token, mod, mask) = data.tokens('val', [0, 1], data.layouts['C1_standard'])
    if not torch.isfinite(token).all():
        raise ValueError('nonfinite sensor input')
    query = torch.arange(min(256, len(model.features)), device=data.device)
    order = np.random.default_rng(77468).permutation(token.shape[1])
    with torch.no_grad():
        y = model(token, mod, mask, query)
        shuffled = model(token[:, order], mod[:, order], mask[:, order], query)
        extra = torch.full((len(token), 7, 3), 17.0, device=data.device)
        padded = model(torch.cat((token, extra), dim=1), F.pad(mod, (0, 7)), F.pad(mask, (0, 7), value=True), query)
        torch.testing.assert_close(shuffled, y, atol=1e-05, rtol=1e-05)
        torch.testing.assert_close(padded, y, atol=1e-05, rtol=1e-05)
    model.zero_grad(set_to_none=True)
    model(token, mod, mask, query).square().mean().backward()
    for (name, p) in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError(f'Nonfinite or missing gradient at {name}')
    n_grad = sum((p.grad.abs().sum().item() > 0 for p in model.parameters()))
    model.zero_grad(set_to_none=True)
    return dict(permutation_max=float((y - shuffled).abs().max()), padding_max=float((y - padded).abs().max()), finite_gradient_tensors=len(list(model.parameters())), nonzero_gradient_tensors=n_grad)

def physical_loss(pred, truth):
    return F.l1_loss(pred, truth) + 0.2 * F.mse_loss(pred, truth)

def training(case, seed, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    vc.seed_everything(seed)
    cfg = budget(case, seed)
    path = dir_for(case, seed, smoke)
    path.mkdir(parents=True, exist_ok=True)
    config = dict(case=case, seed=seed, smoke=smoke, **cfg, enc_dim=128, heads=4, head_hidden=32, head_out=32, p=128, activation='GELU', branch_hidden_layers=2, coord_value_merge='add', masked_softmax_temperature=math.sqrt(128), bias='scalar', physical_target='source train scalar mean/std', loss='L1 + 0.2 MSE (standardized physical outputs)', query_policy='all coordinates', select='mean C1-C5, full native validation', no_angle=True, reference_batch=32, source_sha256=protocol_sources())
    begin = stamp()
    if (path / 'complete.json').exists():
        done = read_json(path / 'complete.json')
        if read_json(path / 'config.json') != config:
            raise ValueError(f'configuration mismatch at {path}')
        for (name, expected) in done['sha256'].items():
            assert sha(path / name) == expected, name
        print('Verified existing complete run', path, flush=True)
        return
    if (path / 'config.json').exists() and read_json(path / 'config.json') != config:
        raise ValueError(f'configuration mismatch at {path}; archive incomplete run before retry')
    write_json(path / 'config.json', config)
    data = Data(case, seed, 'cuda')
    model = VIDON(data).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    y_train = data.truth('train')
    target = torch.from_numpy(((y_train - data.y_mean) / data.y_std).reshape(len(y_train), -1)).to(data.device)
    sampling = None
    checks = inputs_check(data, model) if smoke else {}
    (best, scores) = validate(data, model)
    (best_epoch, stale, first, seconds) = (0, 0, 0, 0.0)
    best_state = copy.deepcopy(model.state_dict())
    history = [[0, best, scores]]
    steps = 0
    state_path = path / 'resume.pt'
    if state_path.exists():
        s = torch.load(state_path, map_location=data.device, weights_only=False)
        model.load_state_dict(s['model'])
        opt.load_state_dict(s['opt'])
        scheduler.load_state_dict(s['scheduler'])
        (best, best_epoch, best_state, history) = (s['best'], s['best_epoch'], s['best_state'], s['history'])
        (stale, first, seconds, steps) = (s['stale'], s['epoch'], s['seconds'], s['steps'])
        print(f'Resume at epoch {first}, step {steps}', flush=True)
    started = time.perf_counter()
    (limit, interval) = (2, 1) if smoke else (cfg['epochs'], cfg['eval_every'])

    def finished():
        return stale >= cfg['patience_evals'] and first >= cfg['min_epochs']
    for epoch in range(first, limit):
        if finished():
            break
        model.train()
        order = np.random.default_rng(seed + epoch + 0).permutation(len(y_train))
        for (step, start) in enumerate(range(0, len(order), cfg['batch_size'])):
            local = order[start:start + cfg['batch_size']]
            rng = np.random.default_rng(seed * 100000 + epoch * 1000 + step)
            truth = target[local]
            lay = data.random_layout(rng)
            sub = vc.sub_layout(lay, rng)
            assert cfg['sub_layout_supervision']
            basis = model.basis()
            loss = physical_loss(model.decode(data.tokens('train', local, lay), basis), truth)
            loss = loss + physical_loss(model.decode(data.tokens('train', local, sub), basis), truth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            opt.step()
            steps += 1
        scheduler.step()
        if (epoch + 1) % interval == 0 or epoch + 1 == limit:
            (score, scores) = validate(data, model)
            if not np.isfinite(score):
                raise ValueError('nonfinite validation metric')
            history.append([epoch + 1, score, scores])
            if score < best - 1e-06:
                (best, stale, best_epoch, best_state) = (score, 0, epoch + 1, copy.deepcopy(model.state_dict()))
            else:
                stale += 1
            elapsed = seconds + time.perf_counter() - started
            first = epoch + 1
            save_pt(state_path, dict(model=model.state_dict(), opt=opt.state_dict(), scheduler=scheduler.state_dict(), best=best, best_epoch=best_epoch, best_state=best_state, history=history, stale=stale, epoch=first, seconds=elapsed, steps=steps))
            write_json(path / 'progress.json', dict(epoch=first, max_epochs=limit, steps=steps, validation=score, best=best, best_epoch=best_epoch, stale=stale, seconds=elapsed, updated_utc=stamp(), smoke=smoke))
            print(f'{case} seed={seed} epoch={first}/{limit} step={steps} val={score:.6g} best={best:.6g} elapsed={elapsed:.1f}s', flush=True)
    train_seconds = seconds + time.perf_counter() - started
    model.load_state_dict(best_state)
    save_pt(path / 'deployment.pt', dict(case=case, seed=seed, stats=stats(data), best_epoch=best_epoch, model={k: v.detach().cpu() for (k, v) in model.state_dict().items()}))
    ref_batch = config['reference_batch']
    arrays = {'truth': data.truth('test'), 'test_ids': data.indices['test']}
    for (name, layout) in data.layouts.items():
        arrays[f'pred_{name}'] = predict(data, model, 'test', layout, batch=ref_batch)
    with (path / 'reference.tmp').open('wb') as f:
        np.savez_compressed(f, **arrays)
    (path / 'reference.tmp').replace(path / 'reference.npz')
    (replay_data, loaded) = load_run(path / 'deployment.pt', 'cuda')
    replay = {}
    for (name, layout) in replay_data.layouts.items():
        y = predict(replay_data, loaded, 'test', layout, batch=ref_batch)
        np.testing.assert_array_equal(replay_data.truth('test'), arrays['truth'])
        np.testing.assert_allclose(y, arrays[f'pred_{name}'], atol=2e-05, rtol=1e-05)
        replay[name] = float(np.max(np.abs(y - arrays[f'pred_{name}'])))
    if smoke and (first, steps) != (2, 2 * math.ceil(len(y_train) / cfg['batch_size'])):
        raise ValueError('incomplete smoke budget')
    write_json(path / 'metrics.json', dict(case=case, seed=seed, smoke=smoke, epochs_run=first, optimizer_steps=steps, train_seconds=train_seconds, reference_batch=ref_batch, history=history, best_epoch=best_epoch, best_validation=best, stats=stats(data), smoke_checks=checks, reload_max_abs_difference=replay, split_sizes={k: len(v) for (k, v) in data.indices.items()}, trainable_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)), started_utc=begin, finished_utc=stamp(), gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    files = ('deployment.pt', 'reference.npz', 'metrics.json', 'config.json')
    write_json(path / 'complete.json', dict(finished_utc=stamp(), sha256={k: sha(path / k) for k in files}))
    print('COMPLETE', path, flush=True)
