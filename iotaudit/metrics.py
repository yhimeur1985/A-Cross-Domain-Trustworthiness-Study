"""Metrics: AUROC, DGG, fingerprint indices, thresholds, cluster bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def auroc(y, s, w=None) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


def group_balanced_weights(y, g) -> np.ndarray:
    """Weights so every (group, class) cell has equal total mass."""
    y, g = np.asarray(y), np.asarray(g)
    w = np.zeros(len(y), dtype=float)
    cells = {}
    for i, key in enumerate(zip(g, y)):
        cells.setdefault(key, []).append(i)
    for idx in cells.values():
        w[idx] = 1.0 / len(idx)
    return w * len(y) / w.sum()


def dgg(auc_random_per_domain: dict, auc_lodo: dict) -> dict:
    """Domain Generalisation Gap per domain and averaged over domains where
    both terms exist.  Both terms are *per-domain* AUROCs, so the gap is not
    contaminated by between-domain composition (see identity null)."""
    per = {d: auc_random_per_domain[d] - auc_lodo[d]
           for d in auc_lodo
           if d in auc_random_per_domain
           and np.isfinite(auc_random_per_domain[d]) and np.isfinite(auc_lodo[d])}
    per["mean"] = float(np.mean(list(per.values()))) if per else float("nan")
    return per


def dfi(y_true, y_pred) -> float:
    """Domain Fingerprint Index: chance-corrected balanced accuracy of a
    dataset-of-origin classifier.  0 = no fingerprint, 1 = perfect."""
    k = len(np.unique(y_true))
    ba = balanced_accuracy_score(y_true, y_pred)
    return float((ba - 1.0 / k) / (1.0 - 1.0 / k))


def identity_null_auroc(y, domains) -> float:
    """AUROC of the feature-free score s(x) = prevalence of x's domain.

    It uses only *which dataset* a flow came from.  It equals 0.5 inside any
    single domain, so every point above 0.5 in a pooled evaluation is
    produced by composition, not by detection."""
    y, domains = np.asarray(y), np.asarray(domains)
    prev = {d: y[domains == d].mean() for d in np.unique(domains)}
    s = np.array([prev[d] for d in domains])
    return auroc(y, s)


def threshold_at_fpr(y, s, fpr=0.01) -> float:
    """Score threshold whose FPR on (y, s) is <= fpr (benign quantile)."""
    ben = np.sort(np.asarray(s)[np.asarray(y) == 0])
    if len(ben) == 0:
        return float("nan")
    k = int(np.ceil((1 - fpr) * len(ben))) - 1
    return float(ben[min(max(k, 0), len(ben) - 1)])


def rates_at(y, s, thr):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = y == 1, y == 0
    tpr = float((s[pos] > thr).mean()) if pos.any() else float("nan")
    fpr = float((s[neg] > thr).mean()) if neg.any() else float("nan")
    return tpr, fpr


def cluster_bootstrap_auroc(y, s, clusters, n_boot=200, seed=0, alpha=0.05):
    """Percentile CI for AUROC, resampling *conversations* (host pairs), not
    flows -- flows that share a host pair are not independent."""
    y, s, clusters = np.asarray(y), np.asarray(s), np.asarray(clusters)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [[] for _ in range(len(uniq))]
    for i, c in enumerate(inv):
        members[c].append(i)
    members = [np.array(m) for m in members]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[p] for p in pick])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
