# -*- coding: utf-8 -*-
"""Ablation variants for E3, isolated from models.py used by E2b.

Variants (all keep the same data/protocol):
  - soft_cap=False: drop the soft injection cap eta_bar
  - spectral=False: drop the spectral clamp eta_spec
  - use_v=False: remove the value memory from delta (delta = k)
  - shuffle_ae=True: permute alpha/eta across token positions within a chunk
  - bidirectional=False: forward scan only
  - layers / dim / chunk / gamma / alpha_init as hyperparameters
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SRNLScan, ConvStem, RULHead


class AblScan(SRNLScan):
    def __init__(self, dim, head_dim=16, gamma=0.025, alpha_init=0.95, eps=1e-6,
                 chunk=16, rule="self", soft_cap=True, spectral=True,
                 use_v=True, shuffle_ae=False):
        super(AblScan, self).__init__(dim, head_dim=head_dim, gamma=gamma,
                                      alpha_init=alpha_init, eps=eps, chunk=chunk,
                                      control=(soft_cap and spectral), rule=rule)
        self.soft_cap = soft_cap
        self.spectral = spectral
        self.use_v = use_v
        self.shuffle_ae = shuffle_ae

    def _step_params(self, Sx, log=None):
        d = self.d
        k_raw, v = Sx[..., d:2 * d], Sx[..., 2 * d:3 * d]
        k = k_raw / torch.sqrt((k_raw ** 2).sum(-1, keepdim=True) + 1e-6)
        if not self.use_v:
            v = torch.zeros_like(v)
        if self.rule == "self":
            eta = self.gamma * F.softplus(Sx[..., 3 * d])
            alpha = torch.sigmoid(Sx[..., 3 * d + 1] + self.logit_a)
        else:
            eta = (self.gamma * F.softplus(self.eta_raw))[None, None, :]
            alpha = torch.sigmoid(self.alpha_logit)[None, None, :]
        alpha = alpha.clamp(1e-6, 1.0 - 1e-6)
        if self.shuffle_ae:
            Lc = alpha.shape[1]
            perm = torch.randperm(Lc, device=alpha.device)
            alpha = alpha[:, perm]
            eta = eta[:, perm]
        delta = k - v
        eta_bar = eta
        eta_spec = None
        if self.soft_cap:
            r = torch.sqrt((delta ** 2).sum(-1) + self.eps ** 2) + self.eps
            eta_inj = (1.0 - alpha) / r * (1.0 - 1e-3)
            eta_bar = eta_inj * (1.0 - torch.exp(-eta / eta_inj))
        if self.spectral:
            eta_spec = 2.0 * alpha / (k ** 2).sum(-1).clamp_min(1e-20) * (1.0 - 1e-6)
        if self.soft_cap and self.spectral:
            eta_x = torch.minimum(eta_bar, eta_spec)
        elif self.soft_cap:
            eta_x = eta_bar
        elif self.spectral:
            eta_x = eta_spec
        else:
            eta_x = eta
        return k, delta, alpha, eta_x


class AblBiLayer(nn.Module):
    def __init__(self, dim, bidirectional=True, **kw):
        super(AblBiLayer, self).__init__()
        self.fwd = AblScan(dim, **kw)
        self.bwd = AblScan(dim, **kw) if bidirectional else None
        self.fuse_logits = nn.Parameter(torch.zeros(2)) if bidirectional else None

    def forward(self, x, log=None):
        yf = self.fwd(x, log)
        if self.bwd is None:
            return yf
        yb = torch.flip(self.bwd(torch.flip(x, [1]), log), [1])
        w = torch.softmax(self.fuse_logits, 0)
        return w[0] * yf + w[1] * yb


class AblEncoder(nn.Module):
    def __init__(self, in_dim, dim=64, layers=2, chunk=16, bidirectional=True, **kw):
        super(AblEncoder, self).__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            AblBiLayer(dim, bidirectional=bidirectional, chunk=chunk, **kw)
            for _ in range(layers)
        ])

    def forward(self, x, log=None):
        h = self.inp(x)
        for layer in self.layers:
            h = h + layer(h, log)
        return h


class AblModel(nn.Module):
    def __init__(self, in_dim, dim=64, layers=2, chunk=16, stem=False,
                 bidirectional=True, cap=125.0, **kw):
        super(AblModel, self).__init__()
        self.stem = ConvStem(in_dim, dim) if stem else None
        enc_in = dim if stem else in_dim
        self.encoder = AblEncoder(enc_in, dim, layers, chunk, bidirectional, **kw)
        self.head = RULHead(dim, cap)

    def forward(self, x, log=None):
        if self.stem is not None:
            s = self.stem(x)
            h = self.encoder(s, log) + s
        else:
            h = self.encoder(x, log)
        return self.head(h)


def build_ablation(spec, in_dim, cap=125.0):
    """spec: dict with rule/soft_cap/spectral/use_v/shuffle_ae/bidirectional/
    layers/dim/chunk/gamma/alpha_init/stem."""
    allowed = ("rule", "soft_cap", "spectral", "use_v", "shuffle_ae",
               "gamma", "alpha_init")
    kw = {k: spec[k] for k in allowed if k in spec}
    return AblModel(in_dim, dim=spec.get("dim", 64), layers=spec.get("layers", 2),
                    chunk=spec.get("chunk", 16), stem=spec.get("stem", False),
                    bidirectional=spec.get("bidirectional", True), cap=cap, **kw)
