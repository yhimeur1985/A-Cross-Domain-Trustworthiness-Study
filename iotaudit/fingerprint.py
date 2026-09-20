"""
Feature-level fingerprint analysis and Fingerprint-Aware Feature Pruning (FAFP).

For each feature j we estimate, from *training domains only*:

  D_j  domain information given the class:
       mean over classes c of  I(x_j ; domain | y=c) / log K_c,
       with every domain given equal mass (removes prevalence effects);
  L_j  label information given the domain:
       mean over two-class domains d of  I(x_j ; y | domain=d) / log 2,
       with both classes given equal mass.

The fingerprint excess  phi_j = D_j - L_j  is large for coordinates that say
much about *where* a flow was captured and little about *whether* it is an
attack.  FAFP removes the top-q fraction of features by phi, with q chosen by
nested leave-one-domain-out inside the training domains (target-blind).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .metrics import auroc


def quantile_bins(X, n_bins=32):
    """Per-column quantile binning; returns int codes."""
    X = np.asarray(X, dtype=float)
    codes = np.empty(X.shape, dtype=np.int16)
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
        codes[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return codes


def _weighted_mi(b, z, w):
    """I(b; z) for integer codes b, z with sample weights w (sum-normalised)."""
    w = w / w.sum()
    nb, nz = b.max() + 1, z.max() + 1
    joint = np.zeros((nb, nz))
    np.add.at(joint, (b, z), w)
    pb = joint.sum(1, keepdims=True)
    pz = joint.sum(0, keepdims=True)
    nz_ = joint > 0
    return float((joint[nz_] * np.log(joint[nz_] / (pb @ pz)[nz_])).sum())


def feature_scores(X, y, dom, n_bins=32, min_cell=50):
    """Return dict with arrays D, L, phi (one entry per column of X)."""
    y, dom = np.asarray(y), np.asarray(dom)
    B = quantile_bins(X, n_bins)
    levels, did = np.unique(dom, return_inverse=True)
    p = X.shape[1]
    D = np.zeros(p)
    L = np.zeros(p)
    # ---- D_j : domain information within each class
    nD = 0
    for c in (0, 1):
        m = y == c
        doms_c = [k for k in range(len(levels)) if ((did == k) & m).sum() >= min_cell]
        if len(doms_c) < 2:
            continue
        m = m & np.isin(did, doms_c)
        zc = np.searchsorted(doms_c, did[m])
        w = np.zeros(m.sum())
        for k in range(len(doms_c)):
            w[zc == k] = 1.0 / (zc == k).sum()
        for j in range(p):
            D[j] += _weighted_mi(B[m, j], zc, w) / np.log(len(doms_c))
        nD += 1
    D /= max(nD, 1)
    # ---- L_j : label information within each two-class domain
    nL = 0
    for k in range(len(levels)):
        m = did == k
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) < min_cell:
            continue
        yy = y[m].astype(int)
        w = np.where(yy == 1, 1.0 / (yy == 1).sum(), 1.0 / (yy == 0).sum())
        for j in range(p):
            L[j] += _weighted_mi(B[m, j], yy, w) / np.log(2)
        nL += 1
    L /= max(nL, 1)
    return {"D": D, "L": L, "phi": D - L}


def prune_order(scores, rule="phi", seed=0):
    """Feature indices in the order they are removed."""
    p = len(scores["phi"])
    if rule == "phi":
        return np.argsort(-scores["phi"])
    if rule == "D":
        return np.argsort(-scores["D"])
    if rule == "L_low":
        return np.argsort(scores["L"])
    if rule == "random":
        return np.random.default_rng(seed).permutation(p)
    raise ValueError(rule)


def keep_mask(order, frac):
    p = len(order)
    k = int(round(frac * p))
    m = np.ones(p, dtype=bool)
    m[order[:k]] = False
    return m


def hgb(seed=0, max_iter=150):
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=seed)


def nested_select_fraction(X, y, dom, fracs, rule="phi", max_rows=40000, seed=0):
    """Choose the pruning fraction by inner leave-one-domain-out over the
    two-class *training* domains only.  Returns (best_frac, curve)."""
    y, dom = np.asarray(y), np.asarray(dom)
    rng = np.random.default_rng(seed)
    two_class = [d for d in np.unique(dom)
                 if min((y[dom == d] == 0).sum(), (y[dom == d] == 1).sum()) >= 50]
    curve = {f: [] for f in fracs}
    for inner in two_class:
        tr = np.where(dom != inner)[0]
        te = np.where(dom == inner)[0]
        if len(tr) > max_rows:
            tr = rng.choice(tr, max_rows, replace=False)
        sc = feature_scores(X[tr], y[tr], dom[tr])
        order = prune_order(sc, rule, seed)
        for f in fracs:
            m = keep_mask(order, f)
            clf = hgb(seed, max_iter=100).fit(X[tr][:, m], y[tr])
            curve[f].append(auroc(y[te], clf.predict_proba(X[te][:, m])[:, 1]))
    mean_curve = {f: float(np.nanmean(v)) if v else float("nan") for f, v in curve.items()}
    best = max(fracs, key=lambda f: (mean_curve[f], -f))
    return best, mean_curve
