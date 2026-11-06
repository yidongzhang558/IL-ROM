"""Fixed-input one-dimensional FNO baseline."""
import torch
from torch import nn
from torch.nn import functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.cin, self.cout, self.modes = cin, cout, modes
        scale = 1 / (cin * cout)
        self.w = nn.Parameter(scale * torch.rand(cin, cout, modes,
                                                 dtype=torch.cfloat))

    def forward(self, x):
        b = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out = torch.zeros(b, self.cout, x.size(-1) // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum(
            'bix,iox->box', x_ft[:, :, :self.modes], self.w)
        return torch.fft.irfft(out, n=x.size(-1))


class FNORecon1D(nn.Module):
    """Direct regression: sensors [B,N_s] -> q_w [B,n] (MLP embedding -> FNO1d)."""

    def __init__(self, sensor_num, fc_size, out_size, modes=16, width=48):
        super().__init__()
        self.fc_size, self.out_size, self.width = fc_size, out_size, width
        self.fc = nn.Sequential(
            nn.Linear(sensor_num, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, fc_size),
        )
        self.embedding = nn.Conv1d(1, width, 1)
        self.convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.fc1, self.fc2 = nn.Linear(width, 128), nn.Linear(128, 1)

    def forward(self, s):
        x = self.fc(s).unsqueeze(1)                      # [B,1,fc]
        x = self.embedding(x)
        x = F.interpolate(x, self.out_size, mode='linear',
                          align_corners=False)
        for i in range(4):
            x1, x2 = self.convs[i](x), self.ws[i](x)
            x = F.gelu(x1 + x2) if i < 3 else x1 + x2
        x = x.permute(0, 2, 1)
        return self.fc2(F.gelu(self.fc1(x))).squeeze(-1)  # [B,n]
