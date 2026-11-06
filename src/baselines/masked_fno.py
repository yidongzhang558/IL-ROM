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
from .voronoi_input import complete_voronoi,restore_pod
class SpectralConv2d(nn.Module):

    def __init__(self, cin, cout, modes_x, modes_y):
        super().__init__()
        (self.cin, self.cout, self.mx, self.my) = (cin, cout, modes_x, modes_y)
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, modes_x, modes_y, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, modes_x, modes_y, dtype=torch.cfloat))

    def forward(self, x):
        (b, c, nx, ny) = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, self.cout, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.mx, :self.my] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :self.mx, :self.my], self.w1)
        out[:, :, -self.mx:, :self.my] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -self.mx:, :self.my], self.w2)
        return torch.fft.irfft2(out, s=(nx, ny))

class MaskFNOLatent(nn.Module):
    """Rasterised sparse readings -> target latent.  Same body as the RB masked-grid FNO, latent head."""

    def __init__(self, n_modalities: int, nx: int, ny: int, latent_dim: int, width: int=32, modes=(16, 16), layers: int=4):
        super().__init__()
        (self.nx, self.ny) = (nx, ny)
        self.lift = nn.Conv2d(2 * n_modalities + 2, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes[0], modes[1]) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * width, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, latent_dim))
        xs = (torch.arange(nx) + 0.5) / nx
        ys = (torch.arange(ny) + 0.5) / ny
        self.register_buffer('coords', torch.stack(torch.meshgrid(xs, ys, indexing='ij')), persistent=False)

    def forward(self, image):
        b = image.shape[0]
        x = torch.cat([image, self.coords.unsqueeze(0).expand(b, -1, -1, -1)], dim=1)
        x = self.lift(x)
        for (i, (s, p)) in enumerate(zip(self.spec, self.pw)):
            y = s(x) + p(x)
            x = F.gelu(y) if i < len(self.spec) - 1 else y
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)

def batches(idx, size):
    for start in range(0, len(idx), size):
        yield idx[start:start + size]

class ConvCase:
    """Paper case 1: forced convection.  Observable = temperature field + velocity profile on a 320 x 320 grid."""
    (name, raster) = ('case2', 160)

    def __init__(self, latent_dim: int, seed: int, input_mode='sparse', deployment=None):
        self.input_mode = input_mode
        self.data = CaseData('case2')
        if deployment is not None:
            for (key, value) in deployment['normalization'].items():
                setattr(self.data, key, value)
        self.cache = FieldCache(self.data)
        self.rep = restore_pod(deployment['representation'], PODCurveRepresentation) if deployment is not None else PODCurveRepresentation(latent_dim).fit(self.cache.flux['train'])
        self.z = {s: self.rep.encode(self.cache.flux[s]) for s in SPLITS}
        self.layouts = fixed_layouts('case2')
        self.n_modalities = 2
        self.latent_dim = latent_dim
        self.metric_key = 'rel_l2'

    def n_split(self, split):
        return len(self.cache.flux[split])

    def rasterise(self, split, local, layout: SensorLayout):
        (n, g) = (len(local), self.raster)
        image = np.zeros((n, 2 * self.n_modalities, g, g), dtype=np.float32)
        raw = self.cache.raw_sensors(split, layout, local)
        ct = layout.temperature_count
        rows = np.clip(np.rint(np.asarray(layout.row_frac) * (g - 1)).astype(int), 0, g - 1)
        cols = np.clip(np.rint(np.asarray(layout.col_frac) * (g - 1)).astype(int), 0, g - 1)
        vals = (raw[:, :ct] - self.data.t_mean) / self.data.t_std
        for j in range(ct):
            image[:, 0, cols[j], rows[j]] = vals[:, j]
            image[:, 2, cols[j], rows[j]] = 1.0
        if self.cache.velocity_count(layout):
            uvals = (raw[:, ct:] - self.data.u_mean) / self.data.u_std
            urows = np.clip(np.rint(U_HEIGHTS * (g - 1)).astype(int), 0, g - 1)
            ucol = int(np.clip(round(U_COL_FRAC * (g - 1)), 0, g - 1))
            for j in range(len(U_HEIGHTS)):
                image[:, 1, ucol, urows[j]] = uvals[:, j]
                image[:, 3, ucol, urows[j]] = 1.0
        if self.input_mode == 'voronoi':
            image = complete_voronoi(image, self.n_modalities, lengths=(0.1, 0.1))
        return torch.from_numpy(image).to(DEVICE)

    def random_layout(self, rng):
        return self.data.random_layout(rng)

    def target_latent(self, split, local):
        return torch.from_numpy(self.z[split][local]).to(DEVICE)

    def decode(self, latent_np):
        return self.rep.decode(latent_np.astype(np.float32))

    def truth(self, split):
        return self.cache.flux[split]

    def metrics(self, pred, split):
        return regression_metrics(pred, self.truth(split))

def predict_latent(model, case, split, layout, batch=32):
    model.eval()
    out = []
    with torch.no_grad():
        for local in batches(np.arange(case.n_split(split)), batch):
            out.append(model(case.rasterise(split, local, layout)).cpu().numpy())
    return np.concatenate(out)

def validation_score(model, case):
    report = evaluate(model, case, 'val')
    return (float(np.mean([r['metrics'][case.metric_key] for r in report.values()])), {k: r['metrics'][case.metric_key] for (k, r) in report.items()})

def train(case, seed, epochs, batch_size, lr, eval_every, patience_evals, min_epochs, width=32, modes=16, layers=4):
    seed_everything(seed + 21)
    model = MaskFNOLatent(case.n_modalities, case.raster, case.raster, case.latent_dim, width=width, modes=(modes, modes), layers=layers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
    n_train = case.n_split('train')
    (best, best_layouts) = validation_score(model, case)
    (best_state, stale, started) = (copy.deepcopy(model.state_dict()), 0, time.time())
    history = [[0, best, best_layouts]]
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(n_train)
        for (step, local) in enumerate(batches(order, batch_size)):
            rng = np.random.default_rng(seed * 100000 + epoch * 1000 + step)
            layout = case.random_layout(rng)
            pred = model(case.rasterise('train', local, layout))
            loss = F.mse_loss(pred, case.target_latent('train', local))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            (score, layouts) = validation_score(model, case)
            history.append([epoch + 1, score, layouts])
            if score < best - 1e-09:
                (best, best_layouts, best_state, stale) = (score, layouts, copy.deepcopy(model.state_dict()), 0)
            else:
                stale += 1
            print(f'[mask_fno {case.name} seed={seed}] epoch={epoch + 1}/{epochs} val={score:.5f} best={best:.5f} elapsed={time.time() - started:.0f}s', flush=True)
            if stale >= patience_evals and epoch + 1 >= min_epochs:
                break
    model.load_state_dict(best_state)
    return (model, {'best_validation': best, 'best_validation_layouts': best_layouts, 'history': history, 'train_seconds': time.time() - started, 'epochs_run': epoch + 1})
def evaluate(model,case,split):
    return {name:{'metrics':case.metrics(case.decode(predict_latent(model,case,split,layout)),split)} for name,layout in case.layouts.items()}
