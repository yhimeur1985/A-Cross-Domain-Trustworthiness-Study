"""
Pure-NumPy multilayer perceptron with domain-generalisation objectives.

No deep-learning framework is required, so the whole audit runs on a laptop
CPU.  Gradients are derived by hand and verified in tests/test_gradients.py.

Objectives (``method``)
-----------------------
erm       plain empirical risk minimisation, uniform minibatches
erm_gb    ERM with group-balanced minibatches (every domain x class cell equal)
groupdro  Group DRO (Sagawa et al., 2020) over domain x class cells
vrex      V-REx risk-variance penalty (Krueger et al., 2021)
coral     marginal Deep-CORAL alignment of domain representations
dann      domain-adversarial training with gradient reversal (Ganin et al.)
c2da      **class-conditional domain alignment (ours)**: aligns first and
          second moments of each class *separately* across the domains that
          contain that class.  Unlike marginal alignment it is well defined
          when domains have different attack prevalence or contain a single
          class (three of nine domains here ship no benign traffic).
"""
from __future__ import annotations

import numpy as np

from .metrics import auroc, group_balanced_weights


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _softplus(z):
    return np.logaddexp(0.0, z)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            if self.wd and p.ndim > 1:
                g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------
# Moment-alignment penalties and their gradients w.r.t. the representation
# ----------------------------------------------------------------------------
def moment_alignment(h, groups, cov_weight=1.0):
    """Penalty  sum_g ||mu_g - mu_bar||^2 + cov_weight/(4k^2) ||C_g - C_bar||_F^2
    over a list of index arrays ``groups``.  Returns (value, dL/dh)."""
    k = h.shape[1]
    groups = [g for g in groups if len(g) > 2]
    grad = np.zeros_like(h)
    if len(groups) < 2:
        return 0.0, grad
    mus = [h[g].mean(0) for g in groups]
    covs = [np.cov(h[g], rowvar=False) for g in groups]
    mu_bar = np.mean(mus, 0)
    c_bar = np.mean(covs, 0)
    cw = cov_weight / (4.0 * k * k)
    val = 0.0
    for g, mu, C in zip(groups, mus, covs):
        dm = mu - mu_bar
        dC = C - c_bar
        val += float(dm @ dm) + cw * float((dC * dC).sum())
        n = len(g)
        grad[g] += 2.0 * dm / n
        grad[g] += cw * (2.0 / (n - 1)) * 2.0 * (h[g] - mu) @ dC
    G = len(groups)
    return val / G, grad / G


class DGMLP:
    def __init__(self, method="erm", hidden=(128, 64), lr=1e-3, steps=1500,
                 batch_per_group=64, batch_erm=1024, lam=1.0, cov_weight=1.0,
                 dro_eta=0.01, vrex_warmup=0.2, weight_decay=1e-5,
                 eval_every=250, seed=0, verbose=False):
        self.method = method
        self.hidden = hidden
        self.lr, self.steps = lr, steps
        self.bpg, self.berm = batch_per_group, batch_erm
        self.lam, self.cov_weight = lam, cov_weight
        self.dro_eta, self.vrex_warmup = dro_eta, vrex_warmup
        self.wd, self.eval_every = weight_decay, eval_every
        self.seed, self.verbose = seed, verbose

    # ---------------------------------------------------------------- init
    def _init(self, d, n_dom):
        rng = np.random.default_rng(self.seed)
        h1, h2 = self.hidden

        def he(a, b):
            return (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float64)

        self.W1, self.b1 = he(d, h1), np.zeros(h1)
        self.W2, self.b2 = he(h1, h2), np.zeros(h2)
        self.w3, self.b3 = he(h2, 1)[:, 0] * 0.5, np.zeros(1)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.w3, self.b3]
        if self.method == "dann":
            self.W4, self.b4 = he(h2, 64), np.zeros(64)
            self.W5, self.b5 = he(64, n_dom) * 0.5, np.zeros(n_dom)
            self.params += [self.W4, self.b4, self.W5, self.b5]
        self.opt = Adam(self.params, lr=self.lr, wd=self.wd)

    # ------------------------------------------------------------- forward
    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        h = _relu(z2)
        logit = h @ self.w3 + self.b3[0]
        return z1, a1, z2, h, logit

    def embed(self, X):
        return self._forward(np.asarray(X, dtype=np.float64))[3]

    def decision_function(self, X, chunk=65536):
        X = np.asarray(X, dtype=np.float64)
        return np.concatenate([self._forward(X[i:i + chunk])[4]
                               for i in range(0, len(X), chunk)])

    def predict_proba(self, X):
        return _sigmoid(self.decision_function(X))

    # ------------------------------------------------------------- batches
    def _sample(self, rng):
        if self.method == "erm":
            idx = rng.integers(0, self.n, self.berm)
            return idx, None
        parts, cells = [], []
        start = 0
        for c, members in enumerate(self.cells):
            take = members[rng.integers(0, len(members), self.bpg)]
            parts.append(take)
            cells.append(np.arange(start, start + len(take)))
            start += len(take)
        return np.concatenate(parts), cells

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, dom, X_val=None, y_val=None, dom_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(float)
        dom = np.asarray(dom)
        self.dom_levels = np.unique(dom)
        dom_id = np.searchsorted(self.dom_levels, dom)
        self.n = len(X)
        # domain x class cells
        keys = sorted(set(zip(dom_id.tolist(), y.astype(int).tolist())))
        self.cell_keys = keys
        self.cells = [np.where((dom_id == d) & (y == c))[0] for d, c in keys]
        G = len(keys)
        self._init(X.shape[1], len(self.dom_levels))
        q = np.ones(G) / G
        rng = np.random.default_rng(self.seed + 1)
        best, best_state = -np.inf, None
        if X_val is not None:
            w_val = group_balanced_weights(y_val, dom_val)

        for step in range(1, self.steps + 1):
            idx, cells = self._sample(rng)
            xb, yb = X[idx], y[idx]
            z1, a1, z2, h, logit = self._forward(xb)
            p = _sigmoid(logit)
            li = _softplus(logit) - yb * logit          # per-sample BCE
            dli = p - yb                                # d li / d logit

            # ---- per-sample weights on the classification loss
            if self.method == "erm":
                wi = np.full(len(idx), 1.0 / len(idx))
            else:
                R = np.array([li[c].mean() for c in cells])
                if self.method == "groupdro":
                    q = q * np.exp(self.dro_eta * R)
                    q = q / q.sum()
                    gw = q
                elif self.method == "vrex":
                    lam = self.lam if step > self.vrex_warmup * self.steps else 0.0
                    gw = 1.0 / G + lam * 2.0 * (R - R.mean()) / G
                else:
                    gw = np.full(G, 1.0 / G)
                wi = np.empty(len(idx))
                for c, cidx in enumerate(cells):
                    wi[cidx] = gw[c] / len(cidx)
            dlogit = dli * wi

            # ---- representation-level gradient
            dh = np.outer(dlogit, self.w3)
            g_w3 = h.T @ dlogit
            g_b3 = np.array([dlogit.sum()])
            extra = []
            if self.method in ("coral", "c2da"):
                cell_dom = np.array([k[0] for k in keys])
                cell_cls = np.array([k[1] for k in keys])
                if self.method == "coral":
                    groups = [np.concatenate([cells[c] for c in np.where(cell_dom == d)[0]])
                              for d in np.unique(cell_dom)]
                    _, gh = moment_alignment(h, groups, self.cov_weight)
                else:
                    gh = np.zeros_like(h)
                    for cls in (0, 1):
                        grp = [cells[c] for c in np.where(cell_cls == cls)[0]]
                        _, g_ = moment_alignment(h, grp, self.cov_weight)
                        gh += g_
                    gh /= 2.0
                dh += self.lam * gh
            elif self.method == "dann":
                prog = step / self.steps
                coef = self.lam * (2.0 / (1.0 + np.exp(-10 * prog)) - 1.0)
                dom_b = np.empty(len(idx), dtype=int)
                for c, cidx in enumerate(cells):
                    dom_b[cidx] = keys[c][0]
                u = h @ self.W4 + self.b4
                a4 = _relu(u)
                zd = a4 @ self.W5 + self.b5
                zd -= zd.max(1, keepdims=True)
                P = np.exp(zd)
                P /= P.sum(1, keepdims=True)
                dz = P.copy()
                dz[np.arange(len(idx)), dom_b] -= 1.0
                dz /= len(idx)
                g_W5 = a4.T @ dz
                g_b5 = dz.sum(0)
                da4 = dz @ self.W5.T
                du = da4 * (u > 0)
                g_W4 = h.T @ du
                g_b4 = du.sum(0)
                dh_dom = du @ self.W4.T
                dh += -coef * dh_dom                      # gradient reversal
                extra = [g_W4, g_b4, g_W5, g_b5]

            dz2 = dh * (z2 > 0)
            g_W2 = a1.T @ dz2
            g_b2 = dz2.sum(0)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)
            g_W1 = xb.T @ dz1
            g_b1 = dz1.sum(0)
            self.opt.step([g_W1, g_b1, g_W2, g_b2, g_w3, g_b3] + extra)

            if X_val is not None and (step % self.eval_every == 0 or step == self.steps):
                s = self.decision_function(X_val)
                score = auroc(y_val, s, w_val)
                if self.verbose:
                    print(f"  step {step:5d}  val-AUROC(group-balanced)={score:.4f}")
                if score > best:
                    best = score
                    best_state = [p.copy() for p in self.params]
        if best_state is not None:
            for p, b in zip(self.params, best_state):
                p[...] = b
        self.val_score_ = best
        return self
