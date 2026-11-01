# -*- coding: utf-8 -*-
"""Regression metrics for C-MAPSS RUL.

PHM08 Score coefficients a1=13, a2=10 are used as the working convention.
They are NOT yet verified against Saxena et al. (2008) in this project:
evidence/pre_draft_check_20261026.md marks them 待核 (U1). Any number produced
here is provisional until that check is done.
"""
import numpy as np


def rmse(pred, true):
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def phm_score(pred, true, a1=13.0, a2=10.0, reduce="sum"):
    """d = pred - true; d<0 uses exp(-d/a1), d>=0 uses exp(d/a2)."""
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    d = pred - true
    s = np.where(d < 0, np.exp(-d / a1) - 1.0, np.exp(d / a2) - 1.0)
    if reduce == "sum":
        return float(s.sum())
    if reduce == "mean":
        return float(s.mean())
    raise ValueError(reduce)


def regression_report(pred, true):
    d = np.asarray(pred, np.float64) - np.asarray(true, np.float64)
    return {
        "rmse": rmse(pred, true),
        "score_sum": phm_score(pred, true, reduce="sum"),
        "mae": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "n": int(len(d)),
        "score_coeffs": "a1=13,a2=10 (pending Saxena 2008 verification)",
    }
