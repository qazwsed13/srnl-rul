# -*- coding: utf-8 -*-
"""Models for E1 smoke: B1 CNN, B2 LSTM, B5 TCN, B8 fixed-rule SRNL, ours bi-SRNL.

SRNL follows the probe implementation (scripts/probe/srnl_probe.py,
srnl_probe_chunk.py): five coupled memories stacked in S, transition
T = alpha I - eta (k + delta) k^T, chunk-wise M_t = M_b G_t.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRNLScan(nn.Module):
    """One direction, one layer. rule='self' uses M^eta/M^alpha; 'fixed' uses scalars."""

    def __init__(self, dim, head_dim=16, gamma=0.025, alpha_init=0.95, eps=1e-6,
                 chunk=16, control=True, rule="self"):
        super(SRNLScan, self).__init__()
        assert dim % head_dim == 0
        self.h = dim // head_dim
        self.d = head_dim
        self.gamma, self.eps, self.control, self.chunk, self.rule = (
            gamma, eps, control, chunk, rule)
        self.logit_a = math.log(alpha_init / (1.0 - alpha_init))
        rows = 3 * self.d + (2 if rule == "self" else 0)
        M0 = torch.zeros(self.h, rows, self.d)
        nn.init.trunc_normal_(M0[:, :3 * self.d], std=0.01)
        self.M0 = nn.Parameter(M0)
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_out = nn.Linear(dim, dim, bias=False)
        if rule == "fixed":
            self.eta_raw = nn.Parameter(torch.zeros(self.h))
            self.alpha_logit = nn.Parameter(torch.full((self.h,), self.logit_a))

    def _step_params(self, Sx, log=None):
        d = self.d
        k_raw, v = Sx[..., d:2 * d], Sx[..., 2 * d:3 * d]
        k = k_raw / torch.sqrt((k_raw ** 2).sum(-1, keepdim=True) + 1e-6)
        if self.rule == "self":
            eta = self.gamma * F.softplus(Sx[..., 3 * d])
            alpha = torch.sigmoid(Sx[..., 3 * d + 1] + self.logit_a)
        else:
            eta = (self.gamma * F.softplus(self.eta_raw))[None, None, :]
            alpha = torch.sigmoid(self.alpha_logit)[None, None, :]
        # keep 1-alpha away from exact 0; otherwise eta_inj=0 and exp(-eta/eta_inj)
        # produces an inf whose backward is 0*inf = NaN (seen at FD002 seed 1)
        alpha = alpha.clamp(1e-6, 1.0 - 1e-6)
        delta = k - v
        if self.control:
            r = torch.sqrt((delta ** 2).sum(-1) + self.eps ** 2) + self.eps
            eta_inj = (1.0 - alpha) / r * (1.0 - 1e-3)
            eta_bar = eta_inj * (1.0 - torch.exp(-eta / eta_inj))
            eta_spec = 2.0 * alpha / (k ** 2).sum(-1).clamp_min(1e-20) * (1.0 - 1e-6)
            eta_x = torch.minimum(eta_bar, eta_spec)
        else:
            eta_x = eta
        if log is not None:
            with torch.no_grad():
                log["alpha_max"] = max(float(log.get("alpha_max", 0.0)), float(alpha.max()))
                log["alpha_min"] = min(float(log.get("alpha_min", 1.0)), float(alpha.min()))
                log["eta_max"] = max(float(log.get("eta_max", 0.0)), float(eta.max()))
        return k, delta, alpha, eta_x

    def forward(self, x, log=None):
        B, L, D = x.shape
        h, d, C = self.h, self.d, self.chunk
        xs = self.w_in(x).view(B, L, h, d)
        qs = F.normalize(self.w_q(x).view(B, L, h, d), dim=-1)
        Sb = self.M0.unsqueeze(0).expand(B, -1, -1, -1)
        I = torch.eye(d, device=x.device, dtype=x.dtype)
        ys = []
        for c in range(0, L, C):
            xc, qc = xs[:, c:c + C], qs[:, c:c + C]
            Sx = torch.einsum("bhrd,bchd->bchr", Sb, xc)
            k, delta, alpha, eta_x = self._step_params(Sx, log)
            kkT = torch.einsum("bchi,bchj->bchij", k, k)
            A = alpha[..., None, None] * I - eta_x[..., None, None] * kkT
            Bm = -eta_x[..., None, None] * torch.einsum("bchi,bchj->bchij", delta, k)
            G = I.expand(B, h, d, d)
            ys_c = []
            for j in range(xc.shape[1]):
                # readout at token j uses M_{j-1} = M_b G_{j-1} (per-token, Eq. 16)
                qj = torch.einsum("bhde,bhe->bhd", G, qc[:, j])
                ys_c.append(torch.einsum("bhrd,bhd->bhr", Sb[:, :, :d], qj))
                G = torch.matmul(G, A[:, j]) + Bm[:, j]
            ys.append(torch.stack(ys_c, 1))
            Sb_new = torch.einsum("bhrd,bhde->bhre", Sb, G)
            if log is not None:
                n0 = Sb.flatten(2).norm(dim=-1)
                n1 = Sb_new.flatten(2).norm(dim=-1)
                log["max_ratio"] = max(log["max_ratio"],
                                       float((n1 / n0.clamp_min(1e-30)).max()))
                log["final_norm"] = max(log["final_norm"], float(n1.max()))
                log["finite"] = log["finite"] and bool(torch.isfinite(Sb_new).all())
            Sb = Sb_new
        y = torch.cat(ys, 1).reshape(B, L, D)
        return self.w_out(y)


def new_log():
    return {"max_ratio": 0.0, "final_norm": 0.0, "finite": True}


class BiSRNLLayer(nn.Module):
    """Forward + backward scans with independent memories; convex fusion."""

    def __init__(self, dim, **kw):
        super(BiSRNLLayer, self).__init__()
        self.fwd = SRNLScan(dim, **kw)
        self.bwd = SRNLScan(dim, **kw)
        self.fuse_logits = nn.Parameter(torch.zeros(2))

    def forward(self, x, log=None):
        yf = self.fwd(x, log)
        yb = self.bwd(torch.flip(x, [1]), log)
        yb = torch.flip(yb, [1])
        w = torch.softmax(self.fuse_logits, 0)
        return w[0] * yf + w[1] * yb


class SRNLEncoder(nn.Module):
    def __init__(self, in_dim, dim=64, layers=2, chunk=16, rule="self",
                 control=True, gamma=0.025, alpha_init=0.95, norm=False):
        super(SRNLEncoder, self).__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            BiSRNLLayer(dim, chunk=chunk, rule=rule, control=control,
                        gamma=gamma, alpha_init=alpha_init)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)]) if norm else None

    def forward(self, x, log=None):
        h = self.inp(x)
        for i, layer in enumerate(self.layers):
            h = h + layer(h, log)
            if self.norms is not None:
                h = self.norms[i](h)
        return h


class RULHead(nn.Module):
    def __init__(self, dim, cap=125.0):
        super(RULHead, self).__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.cap = cap

    def forward(self, h):
        z = torch.cat([h[:, -1], h.mean(1)], dim=-1)
        z = F.gelu(self.fc1(z))
        return self.cap * torch.sigmoid(self.fc2(z)).squeeze(-1)


class ConvStem(nn.Module):
    """Local degradation features before the memory encoder (1D)."""

    def __init__(self, in_dim, dim=64):
        super(ConvStem, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class SRNLModel(nn.Module):
    def __init__(self, in_dim, rule="self", dim=64, layers=2, chunk=16,
                 control=True, cap=125.0, gamma=0.025, alpha_init=0.95,
                 norm=False, stem=False):
        super(SRNLModel, self).__init__()
        self.stem = ConvStem(in_dim, dim) if stem else None
        enc_in = dim if stem else in_dim
        self.encoder = SRNLEncoder(enc_in, dim, layers, chunk, rule, control,
                                   gamma, alpha_init, norm)
        self.head = RULHead(dim, cap)

    def forward(self, x, log=None):
        if self.stem is not None:
            s = self.stem(x)
            h = self.encoder(s, log) + s
        else:
            h = self.encoder(x, log)
        return self.head(h)


class CNN1D(nn.Module):
    def __init__(self, in_dim, ch=80, cap=125.0):
        super(CNN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, ch, 5, padding=2), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.GELU(),
        )
        self.fc1 = nn.Linear(2 * ch, ch)
        self.fc2 = nn.Linear(ch, 1)
        self.cap = cap

    def forward(self, x, log=None):
        z = self.net(x.transpose(1, 2))
        z = torch.cat([z.mean(-1), z.amax(-1)], dim=-1)
        return self.cap * torch.sigmoid(self.fc2(F.gelu(self.fc1(z)))).squeeze(-1)


class LSTM1D(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, cap=125.0):
        super(LSTM1D, self).__init__()
        self.rnn = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.cap = cap

    def forward(self, x, log=None):
        out, _ = self.rnn(x)
        z = F.gelu(self.fc1(out[:, -1]))
        return self.cap * torch.sigmoid(self.fc2(z)).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super(Chomp1d, self).__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1):
        super(TCNBlock, self).__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        out = F.gelu(self.chomp1(self.conv1(x)))
        out = F.gelu(self.chomp2(self.conv2(out)))
        res = x if self.down is None else self.down(x)
        return F.gelu(out + res)


class TCN1D(nn.Module):
    def __init__(self, in_dim, ch=48, cap=125.0):
        super(TCN1D, self).__init__()
        blocks = [TCNBlock(in_dim, ch, dilation=1)]
        blocks += [TCNBlock(ch, ch, dilation=d) for d in (2, 4, 8)]
        self.net = nn.Sequential(*blocks)
        self.fc1 = nn.Linear(2 * ch, ch)
        self.fc2 = nn.Linear(ch, 1)
        self.cap = cap

    def forward(self, x, log=None):
        z = self.net(x.transpose(1, 2))
        z = torch.cat([z[:, :, -1], z.mean(-1)], dim=-1)
        return self.cap * torch.sigmoid(self.fc2(F.gelu(self.fc1(z)))).squeeze(-1)


class BiGRU1D(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, cap=125.0):
        super(BiGRU1D, self).__init__()
        self.rnn = nn.GRU(in_dim, hidden, layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(4 * hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.cap = cap

    def forward(self, x, log=None):
        out, _ = self.rnn(x)
        z = torch.cat([out[:, -1], out.mean(1)], dim=-1)
        return self.cap * torch.sigmoid(self.fc2(F.gelu(self.fc1(z)))).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer1D(nn.Module):
    def __init__(self, in_dim, dim=64, nhead=4, layers=2, ff=128, cap=125.0):
        super(Transformer1D, self).__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.pos = PositionalEncoding(dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=ff,
                                           dropout=0.1, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.cap = cap

    def forward(self, x, log=None):
        h = self.enc(self.pos(self.inp(x)))
        z = torch.cat([h[:, -1], h.mean(1)], dim=-1)
        return self.cap * torch.sigmoid(self.fc2(F.gelu(self.fc1(z)))).squeeze(-1)


def build_model(name, in_dim, cap=125.0, chunk=16, norm=False, stem=False):
    if name == "B1_cnn":
        return CNN1D(in_dim, cap=cap)
    if name == "B2_lstm":
        return LSTM1D(in_dim, cap=cap)
    if name == "B3_bigru":
        return BiGRU1D(in_dim, cap=cap)
    if name == "B4_transformer":
        return Transformer1D(in_dim, cap=cap)
    if name == "B5_tcn":
        return TCN1D(in_dim, cap=cap)
    if name == "B6_mamba" or name == "B7_bntta":
        raise NotImplementedError(name)
    if name == "B8_srnl_fixed":
        return SRNLModel(in_dim, rule="fixed", cap=cap, chunk=chunk,
                         norm=norm, stem=stem)
    if name == "ours_srnl":
        return SRNLModel(in_dim, rule="self", cap=cap, chunk=chunk,
                         norm=norm, stem=stem)
    raise ValueError(name)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
