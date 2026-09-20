"""
All experiments of the paper.  Each function writes JSON/CSV into
``ctx.res_dir`` and returns a dict; figures are drawn by figures.py.

E0  corpus census and curation audit (benign availability, serialisation
    fingerprints, label conflicts, cross-domain duplicates)
E1  evaluation protocols: random split (pooled and per-domain) vs LODO,
    identity-only null, DGG, single-source transfer matrix, threshold transfer
E2  dataset-of-origin fingerprinting (all / benign-only / family-controlled /
    precision-harmonised / minimal fingerprint) + 2-D embedding
E3  per-feature information map (D_j, L_j, phi_j)
E4  Fingerprint-Aware Feature Pruning (FAFP) vs controls, nested selection
E5  domain-generalisation objectives (NumPy MLP) with and without FAFP
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from . import data as D
from .fingerprint import (feature_scores, hgb, keep_mask, nested_select_fraction,
                          prune_order)
from .metrics import (auroc, cluster_bootstrap_auroc, dfi, dgg, identity_null_auroc,
                      rates_at, threshold_at_fpr)
from .models import DGMLP


# ----------------------------------------------------------------------------
@dataclass
class Ctx:
    corpus: str
    out: str
    cap_attack_file: int = 2500
    cap_benign_domain: int = 10000
    seeds: tuple = (0, 1, 2)
    mlp_steps: int = 3000
    n_boot: int = 200
    quick: bool = False
    df: pd.DataFrame = field(default=None, repr=False)

    def setup(self):
        self.res_dir = os.path.join(self.out, "results")
        self.fig_dir = os.path.join(self.out, "figures")
        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        t = time.time()
        self.df = D.load_corpus(self.corpus, self.cap_attack_file, self.cap_benign_domain, seed=0)
        Xf, Xi = D.build_matrices(self.df)
        self.flow_names = list(Xf.columns)
        self.id_names = list(Xi.columns)
        self.X_flow = Xf.values.astype(float)
        self.X_all = np.hstack([self.X_flow, Xi.values.astype(float)])
        self.y = self.df.y.values.astype(int)
        self.dom = self.df.domain.values.astype(str)
        self.fam = self.df.family.values.astype(str)
        a = self.df["Src IP"].astype(str).values
        b = self.df["Dst IP"].astype(str).values
        self.cluster = np.array([f"{min(u, v)}|{max(u, v)}" for u, v in zip(a, b)])
        self.domains = sorted(np.unique(self.dom))
        self.two_class = [d for d in self.domains
                          if min(((self.dom == d) & (self.y == 0)).sum(),
                                 ((self.dom == d) & (self.y == 1)).sum()) >= 50]
        log(f"loaded {len(self.y):,} flows, {len(self.flow_names)} flow features, "
            f"{len(self.domains)} domains ({len(self.two_class)} two-class) "
            f"in {time.time()-t:.0f}s")
        return self

    def save(self, name, obj):
        with open(os.path.join(self.res_dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def _split_val(idx, y, dom, frac=0.15, seed=0):
    strat = np.char.add(dom[idx].astype(str), y[idx].astype(str))
    tr, va = train_test_split(idx, test_size=frac, stratify=strat, random_state=seed)
    return tr, va


def _fold_metrics(ctx, y_te, s_te, cl_te, y_va, s_va, seed=0):
    thr = threshold_at_fpr(y_va, s_va, 0.01)
    tpr_f, fpr_f = rates_at(y_te, s_te, thr)
    thr_r = threshold_at_fpr(y_te, s_te, 0.01)
    tpr_r, _ = rates_at(y_te, s_te, thr_r) if np.isfinite(thr_r) else (float("nan"), None)
    a = auroc(y_te, s_te)
    ci = cluster_bootstrap_auroc(y_te, s_te, cl_te, ctx.n_boot, seed) if np.isfinite(a) \
        else (float("nan"), float("nan"))
    return {"auroc": a, "ci": ci, "tpr_frozen": tpr_f, "fpr_frozen": fpr_f,
            "tpr_retro": tpr_r, "n_test": int(len(y_te)),
            "n_test_benign": int((y_te == 0).sum())}


# ============================================================================
# E0  Census and curation audit
# ============================================================================
_TS_PATTERNS = [
    (r"^\d{2}/\d{2}/\d{2} \d{1,2}:\d{2}$", "dd/mm/yy H:MM (no seconds)"),
    (r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [AP]M$", "dd/mm/yyyy hh:mm:ss AM/PM"),
    (r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", "ISO-8601"),
]


def _ts_format(s):
    for pat, name in _TS_PATTERNS:
        if re.match(pat, s):
            return name
    return "other"


def serialisation_fingerprints(corpus, max_lines=3000):
    """Inspect the *raw text* of each file: timestamp format, float precision,
    how integral floats are written, number of columns."""
    rows = []
    for dom, path in D.list_corpus_files(corpus):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().rstrip("\r\n").split(",")
            its = header.index("Timestamp") if "Timestamp" in header else None
            dec, dotzero, n, tsf = [], 0, 0, {}
            fl_idx = [i for i, c in enumerate(header) if c in
                      ("Flow Packets/s", "Flow IAT Mean", "Packet Length Mean",
                       "Fwd Packet Length Mean", "Average Packet Size")]
            ints_idx = [i for i, c in enumerate(header) if c in
                        ("Total Length of Fwd Packet", "Fwd Packet Length Max",
                         "Packet Length Min", "Idle Mean")]
            crlf = False
            for k, line in enumerate(fh):
                if k >= max_lines:
                    break
                crlf |= line.endswith("\r\n")
                parts = line.rstrip("\r\n").split(",")
                if len(parts) != len(header):
                    continue
                n += 1
                if its is not None:
                    f = _ts_format(parts[its])
                    tsf[f] = tsf.get(f, 0) + 1
                for i in fl_idx:
                    v = parts[i]
                    if "." in v and "e" not in v.lower():
                        dec.append(len(v.split(".")[1]))
                for i in ints_idx:
                    dotzero += parts[i].endswith(".0")
        rows.append({"domain": D.DOMAIN_SHORT[dom], "file": os.path.basename(path)[:-7],
                     "n_columns": len(header), "has_device_col": "Device" in header,
                     "timestamp_format": max(tsf, key=tsf.get) if tsf else "n/a",
                     "max_decimals": int(max(dec)) if dec else 0,
                     "median_decimals": float(np.median(dec)) if dec else 0.0,
                     "int_fields_written_as_x.0": dotzero / max(1, n * max(1, len(ints_idx))),
                     "crlf": crlf})
    return pd.DataFrame(rows)


def e0_census(ctx: Ctx):
    log("E0 census and curation audit")
    man = D.load_manifest(ctx.corpus)
    cen = pd.DataFrame(man.get("files", []))
    out = {}
    if len(cen):
        cen["domain"] = cen["domain"].map(D.DOMAIN_SHORT)
        recs = {}
        for dname, t in cen.groupby("domain"):
            recs[dname] = {"files": len(t), "attack_files": int((~t.benign).sum()),
                           "rows_total": int(t.rows_total.sum()),
                           "benign_rows_total": int(t.loc[t.benign, "rows_total"].sum()),
                           "rows_sampled": int(t.rows_sampled.sum()),
                           "gb": t.bytes.sum() / 1e9}
        g = pd.DataFrame.from_dict(recs, orient="index")
        g.index.name = "domain"
        g["attack_prevalence"] = 1 - g.benign_rows_total / g.rows_total
        g.to_csv(os.path.join(ctx.res_dir, "e0_census.csv"))
        out["census"] = g.reset_index().to_dict(orient="records")
    # modelling subset composition
    comp = ctx.df.groupby(["domain", "y"]).size().unstack(fill_value=0)
    comp.columns = ["benign", "attack"] if comp.shape[1] == 2 else comp.columns
    comp.to_csv(os.path.join(ctx.res_dir, "e0_model_subset.csv"))
    out["model_subset"] = comp.reset_index().to_dict(orient="records")
    fams = ctx.df[ctx.df.y == 1].groupby("family")["domain"].agg(lambda s: sorted(set(s)))
    out["families"] = {k: v for k, v in fams.items()}
    # serialisation fingerprints
    ser = serialisation_fingerprints(ctx.corpus)
    ser.to_csv(os.path.join(ctx.res_dir, "e0_serialisation.csv"), index=False)
    agg = ser.groupby("domain").agg(n_columns=("n_columns", "first"),
                                    device_col=("has_device_col", "first"),
                                    timestamp_format=("timestamp_format",
                                                      lambda s: s.value_counts().index[0]),
                                    max_decimals=("max_decimals", "max"),
                                    median_decimals=("median_decimals", "median"),
                                    dotzero=("int_fields_written_as_x.0", "mean"))
    agg.to_csv(os.path.join(ctx.res_dir, "e0_serialisation_by_domain.csv"))
    out["serialisation"] = agg.reset_index().to_dict(orient="records")
    # label conflicts and cross-domain duplicates (on the full sampled corpus)
    full = D.load_corpus(ctx.corpus)
    Xf, _ = D.build_matrices(full)
    key = pd.util.hash_pandas_object(Xf.round(6), index=False).values
    full = full[["domain", "y"]].assign(key=key)
    g = full.groupby("key")
    lab = g["y"].agg(["min", "max"])
    conflicting = lab.index[lab["min"] != lab["max"]]
    ndom = g["domain"].nunique()
    cross = ndom.index[ndom > 1]
    full["conflict"] = full.key.isin(conflicting)
    full["cross_domain_dup"] = full.key.isin(cross)
    dup = full.groupby("domain").agg(
        rows=("y", "size"),
        conflict_rate=("conflict", "mean"),
        cross_domain_dup_rate=("cross_domain_dup", "mean"),
        within_dup_rate=("key", lambda s: 1 - s.nunique() / len(s)))
    dup.to_csv(os.path.join(ctx.res_dir, "e0_duplicates.csv"))
    out["duplicates"] = dup.reset_index().to_dict(orient="records")
    out["duplicates_total"] = {"rows": int(len(full)), "unique_vectors": int(len(lab)),
                               "conflicting_vectors": int(len(conflicting)),
                               "rows_in_conflicting": int(full.conflict.sum()),
                               "rows_cross_domain": int(full.cross_domain_dup.sum())}
    ctx.save("e0_census.json", out)
    return out


# ============================================================================
# E1  Protocols: random split vs LODO, identity null, DGG, transfer matrix
# ============================================================================
def _fit_predict_hgb(Xtr, ytr, Xte, seed=0):
    c = hgb(seed).fit(Xtr, ytr)
    return c.predict_proba(Xte)[:, 1], c


def _fit_predict_mlp(Xtr, ytr, dtr, Xva, yva, dva, Xte, method="erm", seed=0, steps=3000,
                     ttn_te=None, ttn_tr=None, ttn_va=None, **kw):
    if ttn_tr is not None:
        A, B, C = ttn_tr, ttn_va, ttn_te
    else:
        sc = D.SignedLogScaler().fit(Xtr)
        A, B, C = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    m = DGMLP(method, steps=steps, seed=seed, **kw).fit(A, ytr, dtr, B, yva, dva)
    return m.decision_function(C), m.decision_function(B), m


def e1_protocols(ctx: Ctx):
    log("E1 protocols: random split vs leave-one-dataset-out")
    y, dom, cl = ctx.y, ctx.dom, ctx.cluster
    feats = {"flow": ctx.X_flow, "flow+id": ctx.X_all}
    res = {"random": {}, "lodo": {}, "identity_null": {}, "dgg": {}}
    idx = np.arange(len(y))
    strat = np.char.add(dom, y.astype(str))
    tr, te = train_test_split(idx, test_size=0.3, stratify=strat, random_state=0)
    tr, va = _split_val(tr, y, dom, 0.15, 0)
    res["identity_null"]["random_test_pooled"] = identity_null_auroc(y[te], dom[te])
    models = ["HGB", "MLP-ERM"]
    for fname, X in feats.items():
        for mname in models:
            key = f"{mname}|{fname}"
            log(f"  random split  {key}")
            if mname == "HGB":
                s, _ = _fit_predict_hgb(X[tr], y[tr], X[te])
            else:
                s, _, _ = _fit_predict_mlp(X[tr], y[tr], dom[tr], X[va], y[va], dom[va], X[te],
                                           "erm", 0, ctx.mlp_steps)
            per = {d: auroc(y[te][dom[te] == d], s[dom[te] == d]) for d in ctx.domains}
            res["random"][key] = {"pooled": auroc(y[te], s), "per_domain": per,
                                  "per_domain_mean": float(np.nanmean(
                                      [per[d] for d in ctx.two_class]))}
            # LODO
            lodo = {}
            for T in ctx.domains:
                trn = np.where(dom != T)[0]
                ttr, tva = _split_val(trn, y, dom, 0.15, 0)
                tst = np.where(dom == T)[0]
                if mname == "HGB":
                    c = hgb(0).fit(X[ttr], y[ttr])
                    s_te = c.predict_proba(X[tst])[:, 1]
                    s_va = c.predict_proba(X[tva])[:, 1]
                else:
                    s_te, s_va, _ = _fit_predict_mlp(X[ttr], y[ttr], dom[ttr], X[tva], y[tva],
                                                     dom[tva], X[tst], "erm", 0, ctx.mlp_steps)
                lodo[T] = _fold_metrics(ctx, y[tst], s_te, cl[tst], y[tva], s_va)
            res["lodo"][key] = lodo
            lodo_auc = {d: lodo[d]["auroc"] for d in ctx.two_class}
            res["dgg"][key] = dgg(per, lodo_auc)
            res["lodo"][key]["_mean_auroc"] = float(np.nanmean(list(lodo_auc.values())))
            log(f"    pooled={res['random'][key]['pooled']:.3f} "
                f"per-domain={res['random'][key]['per_domain_mean']:.3f} "
                f"LODO={res['lodo'][key]['_mean_auroc']:.3f} DGG={res['dgg'][key]['mean']:.3f}")
    # natural-prevalence identity null from the manifest (full file sizes)
    man = pd.DataFrame(D.load_manifest(ctx.corpus).get("files", []))
    if len(man):
        pos = man[~man.benign].groupby("domain").rows_total.sum()
        neg = man[man.benign].groupby("domain").rows_total.sum()
        doms = sorted(set(pos.index) | set(neg.index))
        P = np.array([pos.get(d, 0) for d in doms], float)
        N = np.array([neg.get(d, 0) for d in doms], float)
        pi = P / (P + N)
        num = 0.0
        for i in range(len(doms)):
            for j in range(len(doms)):
                w = P[i] * N[j]
                num += w * (1.0 if pi[i] > pi[j] else 0.5 if pi[i] == pi[j] else 0.0)
        res["identity_null"]["natural_prevalence"] = num / (P.sum() * N.sum())
    # single-source -> single-target transfer matrix (two-class domains)
    log("  transfer matrix")
    Xs = ctx.X_flow
    M = pd.DataFrame(index=ctx.two_class, columns=ctx.two_class, dtype=float)
    for S in ctx.two_class:
        ms = dom == S
        c = hgb(0).fit(Xs[ms], y[ms])
        for T in ctx.two_class:
            mt = dom == T
            if S == T:
                a, b = train_test_split(np.where(ms)[0], test_size=0.3, random_state=0,
                                        stratify=y[ms])
                cc = hgb(0).fit(Xs[a], y[a])
                M.loc[S, T] = auroc(y[b], cc.predict_proba(Xs[b])[:, 1])
            else:
                M.loc[S, T] = auroc(y[mt], c.predict_proba(Xs[mt])[:, 1])
    M.to_csv(os.path.join(ctx.res_dir, "e1_transfer_matrix.csv"))
    res["transfer_matrix"] = M.to_dict()
    ctx.save("e1_protocols.json", res)
    return res


# ============================================================================
# E2  Dataset-of-origin fingerprinting
# ============================================================================
def _domain_clf(X, d, seed=0, max_iter=150):
    tr, te = train_test_split(np.arange(len(d)), test_size=0.3, stratify=d, random_state=seed)
    c = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                       class_weight="balanced", random_state=seed)
    c.fit(X[tr], d[tr])
    p = c.predict(X[te])
    return d[te], p, c


def _round_sig(X, sig=3):
    X = np.asarray(X, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(X == 0, 0, np.floor(np.log10(np.abs(X))))
    f = 10.0 ** (sig - 1 - mag)
    return np.round(X * f) / f


def e2_fingerprint(ctx: Ctx):
    log("E2 dataset-of-origin fingerprinting")
    y, dom, fam = ctx.y, ctx.dom, ctx.fam
    res = {"conditions": {}}

    def run(name, mask, X):
        dt, dp, _ = _domain_clf(X[mask], dom[mask])
        k = len(np.unique(dom[mask]))
        r = {"k": k, "n": int(mask.sum()), "dfi": dfi(dt, dp),
             "balanced_acc": float(np.mean([np.mean(dp[dt == c] == c) for c in np.unique(dt)])),
             "domains": sorted(np.unique(dom[mask]).tolist())}
        res["conditions"][name] = r
        log(f"  {name:38s} K={k} n={r['n']:6d} BA={r['balanced_acc']:.3f} DFI={r['dfi']:.3f}")
        return dt, dp

    allm = np.ones(len(y), bool)
    run("all flows | flow+id", allm, ctx.X_all)
    dt, dp = run("all flows | flow", allm, ctx.X_flow)
    labs = ctx.domains
    cm = confusion_matrix(dt, dp, labels=labs, normalize="true")
    res["confusion"] = {"labels": labs, "matrix": cm}
    run("benign only | flow", y == 0, ctx.X_flow)
    run("attack only | flow", y == 1, ctx.X_flow)
    run("all flows | flow rounded 3 s.f.", allm, _round_sig(ctx.X_flow, 3))
    run("all flows | flow rounded 2 s.f.", allm, _round_sig(ctx.X_flow, 2))
    # label task for reference (same learner, same features)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                              stratify=np.char.add(dom, y.astype(str)), random_state=0)
    c = hgb(0).fit(ctx.X_flow[tr], y[tr])
    p = c.predict(ctx.X_flow[te])
    res["label_task_balanced_acc"] = float(np.mean([np.mean(p[y[te] == c_] == c_) for c_ in (0, 1)]))
    # family-controlled: same attack family, different datasets
    fam_res = {}
    for f in sorted(set(fam[y == 1])):
        m = (fam == f) & (y == 1)
        doms = [d for d in np.unique(dom[m]) if (m & (dom == d)).sum() >= 100]
        if len(doms) < 2:
            continue
        m = m & np.isin(dom, doms)
        dt_, dp_ = run(f"family={f} | flow", m, ctx.X_flow)
        fam_res[f] = res["conditions"][f"family={f} | flow"]
    res["family_controlled"] = fam_res
    # minimal fingerprint: how few features identify the dataset?
    _, _, c_full = _domain_clf(ctx.X_flow, dom, max_iter=100)
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(0)
    sub = rng.choice(len(y), min(20000, len(y)), replace=False)
    pi = permutation_importance(c_full, ctx.X_flow[sub], dom[sub], n_repeats=3,
                                random_state=0, scoring="balanced_accuracy")
    order = np.argsort(-pi.importances_mean)
    curve = []
    for k in (1, 2, 3, 5, 8, 12, 20):
        dt_, dp_, _ = _domain_clf(ctx.X_flow[:, order[:k]], dom, max_iter=100)
        curve.append({"k": k, "dfi": dfi(dt_, dp_),
                      "features": [ctx.flow_names[j] for j in order[:k]]})
        log(f"  minimal fingerprint k={k:2d}: DFI={curve[-1]['dfi']:.3f}")
    res["minimal_fingerprint"] = curve
    # 2-D embedding for the demonstration figure
    from sklearn.manifold import TSNE
    n_per = 700
    pick = []
    for d in ctx.domains:
        for yy in (0, 1):
            ii = np.where((dom == d) & (y == yy))[0]
            if len(ii):
                pick.extend(rng.choice(ii, min(n_per, len(ii)), replace=False))
    pick = np.array(pick)
    Z = D.SignedLogScaler().fit_transform(ctx.X_flow[pick])
    E = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(Z)
    pd.DataFrame({"x": E[:, 0], "y": E[:, 1], "domain": dom[pick], "label": y[pick],
                  "family": fam[pick]}).to_csv(os.path.join(ctx.res_dir, "e2_tsne.csv"),
                                                index=False)
    ctx.save("e2_fingerprint.json", res)
    return res


# ============================================================================
# E3  Per-feature information map
# ============================================================================
def e3_feature_map(ctx: Ctx):
    log("E3 per-feature information map")
    sc = feature_scores(ctx.X_flow, ctx.y, ctx.dom)
    tab = pd.DataFrame({"feature": ctx.flow_names, "D": sc["D"], "L": sc["L"],
                        "phi": sc["phi"]}).sort_values("phi", ascending=False)
    tab.to_csv(os.path.join(ctx.res_dir, "e3_feature_map.csv"), index=False)
    ctx.save("e3_feature_map.json", {"table": tab.to_dict(orient="records")})
    return tab


# ============================================================================
# E4  Fingerprint-aware feature pruning
# ============================================================================
FRACS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def e4_pruning(ctx: Ctx):
    log("E4 fingerprint-aware feature pruning (FAFP)")
    y, dom, cl, X = ctx.y, ctx.dom, ctx.cluster, ctx.X_flow
    res = {"curve": {}, "nested": {}, "kept_features": {}}
    # (a) oracle curves: every fraction on every fold (for the figure only)
    for rule in ("phi", "D", "random"):
        res["curve"][rule] = {}
        for f in FRACS:
            vals = {}
            for T in ctx.two_class:
                tr = dom != T
                sc = feature_scores(X[tr], y[tr], dom[tr])
                reps = []
                for rs in (range(3) if rule == "random" and f > 0 else [0]):
                    m = keep_mask(prune_order(sc, rule, seed=rs), f)
                    c = hgb(0).fit(X[tr][:, m], y[tr])
                    reps.append(auroc(y[~tr], c.predict_proba(X[~tr][:, m])[:, 1]))
                vals[T] = float(np.mean(reps))
            res["curve"][rule][str(f)] = vals
            log(f"  curve {rule:6s} frac={f:.1f}  mean LODO AUROC="
                f"{np.nanmean(list(vals.values())):.4f}")
    # (b) nested, target-blind FAFP on all nine folds, vs baselines
    for T in ctx.domains:
        trn = np.where(dom != T)[0]
        ttr, tva = _split_val(trn, y, dom, 0.15, 0)
        tst = np.where(dom == T)[0]
        sc = feature_scores(X[ttr], y[ttr], dom[ttr])
        fold = {}
        q, inner = nested_select_fraction(X[ttr], y[ttr], dom[ttr], FRACS, "phi")
        q_d, _ = nested_select_fraction(X[ttr], y[ttr], dom[ttr], FRACS, "D")
        variants = {
            "no pruning": np.ones(X.shape[1], bool),
            "FAFP (nested)": keep_mask(prune_order(sc, "phi"), q),
            "D-only (same q)": keep_mask(prune_order(sc, "D"), q),
            "D-only (own nested q)": keep_mask(prune_order(sc, "D"), q_d),
            "random (same q)": keep_mask(prune_order(sc, "random", seed=1), q),
        }
        for name, m in variants.items():
            c = hgb(0).fit(X[ttr][:, m], y[ttr])
            s_te = c.predict_proba(X[tst][:, m])[:, 1]
            s_va = c.predict_proba(X[tva][:, m])[:, 1]
            fold[name] = _fold_metrics(ctx, y[tst], s_te, cl[tst], y[tva], s_va)
            fold[name]["n_features"] = int(m.sum())
        fold["_q"] = q
        fold["_q_D"] = q_d
        fold["_inner_curve"] = inner
        res["nested"][T] = fold
        res["kept_features"][T] = [ctx.flow_names[j] for j in
                                   np.where(variants["FAFP (nested)"])[0]]
        res.setdefault("kept_features_D", {})[T] = [ctx.flow_names[j] for j in
                                                    np.where(variants["D-only (own nested q)"])[0]]
        log(f"  fold {T:13s} q*={q:.1f} " + " ".join(
            f"{k}={v['auroc']:.3f}/{v['tpr_frozen']:.2f}" for k, v in fold.items()
            if not k.startswith("_")))
    # (c) in-distribution cost of pruning (random split)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=np.char.add(dom, y.astype(str)),
                              random_state=0)
    sc = feature_scores(X[tr], y[tr], dom[tr])
    cost = {}
    for f in FRACS:
        m = keep_mask(prune_order(sc, "phi"), f)
        c = hgb(0).fit(X[tr][:, m], y[tr])
        s = c.predict_proba(X[te][:, m])[:, 1]
        cost[str(f)] = {"pooled": auroc(y[te], s),
                        "per_domain_mean": float(np.nanmean(
                            [auroc(y[te][dom[te] == d], s[dom[te] == d]) for d in ctx.two_class]))}
    res["in_distribution_cost"] = cost
    ctx.save("e4_pruning.json", res)
    return res


# ============================================================================
# E5  Domain-generalisation objectives
# ============================================================================
DG_METHODS = ["erm", "erm_gb", "groupdro", "vrex", "coral", "dann", "c2da"]


def e5_dg(ctx: Ctx, methods=None, with_fafp=True, with_ttn=True, configs=None, merge=False):
    log("E5 domain-generalisation objectives (NumPy MLP)")
    methods = methods or DG_METHODS
    y, dom, cl, X = ctx.y, ctx.dom, ctx.cluster, ctx.X_flow
    out_path = os.path.join(ctx.res_dir, "e5_dg.json")
    res = json.load(open(out_path)) if merge and os.path.exists(out_path) else {}
    prev = os.path.join(ctx.res_dir, "e4_pruning.json")
    e4 = json.load(open(prev)) if os.path.exists(prev) else {}
    kept = {"fafp": e4.get("kept_features"), "fafpD": e4.get("kept_features_D")}
    if configs is None:
        configs = [(m, "flow") for m in methods]
        if with_fafp:
            for fs in ("fafp", "fafpD"):
                if kept[fs] is not None:
                    configs += [("erm_gb", fs), ("c2da", fs)]
        if with_ttn:
            configs += [("erm_gb", "ttn")]
    for method, fs in configs:
        key = f"{method}|{fs}"
        res[key] = {}
        for seed in ctx.seeds:
            per = {}
            for T in ctx.domains:
                trn = np.where(dom != T)[0]
                ttr, tva = _split_val(trn, y, dom, 0.15, seed)
                tst = np.where(dom == T)[0]
                cols = np.arange(X.shape[1])
                if fs in ("fafp", "fafpD"):
                    cols = np.array([ctx.flow_names.index(c) for c in kept[fs][T]])
                Xc = X[:, cols]
                kw = {}
                if fs == "ttn":
                    Z = D.per_domain_standardise(Xc, dom)
                    kw = dict(ttn_tr=Z[ttr], ttn_va=Z[tva], ttn_te=Z[tst])
                s_te, s_va, _ = _fit_predict_mlp(Xc[ttr], y[ttr], dom[ttr], Xc[tva], y[tva],
                                                 dom[tva], Xc[tst], method, seed,
                                                 ctx.mlp_steps, **kw)
                per[T] = _fold_metrics(ctx, y[tst], s_te, cl[tst], y[tva], s_va, seed)
            res[key][str(seed)] = per
            mean_auc = np.nanmean([per[d]["auroc"] for d in ctx.two_class])
            mean_tpr = np.nanmean([per[d]["tpr_frozen"] for d in ctx.domains])
            log(f"  {key:16s} seed={seed} mean LODO AUROC={mean_auc:.4f} "
                f"mean TPR@frozen={mean_tpr:.3f}")
        ctx.save("e5_dg.json", res)
    return res


def e5_lambda_sensitivity(ctx: Ctx, lams=(0.1, 1.0, 10.0)):
    """Oracle sensitivity of alignment strength (reported, never used to pick)."""
    log("E5b lambda sensitivity (oracle, for the appendix)")
    y, dom, cl, X = ctx.y, ctx.dom, ctx.cluster, ctx.X_flow
    res = {}
    for method in ("coral", "c2da", "dann"):
        for lam in lams:
            vals = {}
            for T in ctx.two_class:
                trn = np.where(dom != T)[0]
                ttr, tva = _split_val(trn, y, dom, 0.15, 0)
                tst = np.where(dom == T)[0]
                s_te, _, _ = _fit_predict_mlp(X[ttr], y[ttr], dom[ttr], X[tva], y[tva], dom[tva],
                                              X[tst], method, 0, ctx.mlp_steps, lam=lam)
                vals[T] = auroc(y[tst], s_te)
            res[f"{method}|{lam}"] = vals
            log(f"  {method} lam={lam}: {np.nanmean(list(vals.values())):.4f}")
    ctx.save("e5_lambda.json", res)
    return res


# ============================================================================
# E1b  Exact pair decomposition of the pooled AUROC
# ============================================================================
def pair_decomposition(y, s, dom):
    """Pooled AUROC = sum_{i,j} w_ij A_ij over (attack domain i, benign domain j)
    with w_ij = P_i N_j / (P N).  Splits it into a within-dataset part (i=j)
    and a cross-dataset part (i!=j) that dataset identity alone can solve."""
    from scipy.stats import rankdata
    y, s, dom = np.asarray(y), np.asarray(s), np.asarray(dom)
    doms = sorted(np.unique(dom))
    P = {d: ((dom == d) & (y == 1)).sum() for d in doms}
    N = {d: ((dom == d) & (y == 0)).sum() for d in doms}
    Pt, Nt = sum(P.values()), sum(N.values())
    rows = []
    for i in doms:
        for j in doms:
            if P[i] == 0 or N[j] == 0:
                continue
            a = s[(dom == i) & (y == 1)]
            b = s[(dom == j) & (y == 0)]
            r = rankdata(np.concatenate([a, b]))
            A = (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
            rows.append({"attack_domain": i, "benign_domain": j, "w": P[i] * N[j] / (Pt * Nt),
                         "A": float(A)})
    t = pd.DataFrame(rows)
    within = t[t.attack_domain == t.benign_domain]
    cross = t[t.attack_domain != t.benign_domain]
    omega = within.w.sum()
    return {"pooled_reconstructed": float((t.w * t.A).sum()),
            "within_mass": float(omega),
            "within_auroc": float((within.w * within.A).sum() / max(omega, 1e-12)),
            "cross_auroc": float((cross.w * cross.A).sum() / max(1 - omega, 1e-12)),
            "table": t.to_dict(orient="records")}


def e1b_decomposition(ctx: Ctx):
    log("E1b pair decomposition of pooled AUROC")
    y, dom = ctx.y, ctx.dom
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=np.char.add(dom, y.astype(str)),
                              random_state=0)
    tr, va = _split_val(tr, y, dom, 0.15, 0)
    out = {}
    for fname, X in (("flow", ctx.X_flow), ("flow+id", ctx.X_all)):
        s, _ = _fit_predict_hgb(X[tr], y[tr], X[te])
        out[f"HGB|{fname}"] = pair_decomposition(y[te], s, dom[te])
        out[f"HGB|{fname}"]["pooled_direct"] = auroc(y[te], s)
    prev = {d: y[tr][dom[tr] == d].mean() for d in ctx.domains}
    s_id = np.array([prev[d] for d in dom[te]])
    out["identity_null"] = pair_decomposition(y[te], s_id, dom[te])
    for k, v in out.items():
        log(f"  {k:14s} pooled={v['pooled_reconstructed']:.3f} within-mass={v['within_mass']:.3f} "
            f"within={v['within_auroc']:.3f} cross={v['cross_auroc']:.3f}")
    ctx.save("e1b_decomposition.json", out)
    return out


# ============================================================================
# E1c  Which identifiers help, which hurt?  (ports vs. addresses vs. time)
# ============================================================================
ID_GROUPS = {
    "flow": [],
    "flow+ports": ["src_port", "dst_port"],
    "flow+IP octets": [f"{p}_ip_o{i}" for p in ("src", "dst") for i in range(1, 5)],
    "flow+timestamp": ["ts_hour", "ts_year"],
    "flow+all ids": None,
}


def e1c_identifier_ablation(ctx: Ctx):
    log("E1c identifier ablation (ports / IP octets / timestamp)")
    y, dom = ctx.y, ctx.dom
    nf = len(ctx.flow_names)
    res = {}
    for gname, cols in ID_GROUPS.items():
        if cols is None:
            idx = np.arange(ctx.X_all.shape[1])
        else:
            idx = np.r_[np.arange(nf), [nf + ctx.id_names.index(c) for c in cols]].astype(int)
        X = ctx.X_all[:, idx]
        for mname in ("HGB", "MLP-ERM"):
            per = {}
            for T in ctx.two_class:
                trn = np.where(dom != T)[0]
                ttr, tva = _split_val(trn, y, dom, 0.15, 0)
                tst = np.where(dom == T)[0]
                if mname == "HGB":
                    s = hgb(0).fit(X[ttr], y[ttr]).predict_proba(X[tst])[:, 1]
                else:
                    s, _, _ = _fit_predict_mlp(X[ttr], y[ttr], dom[ttr], X[tva], y[tva], dom[tva],
                                               X[tst], "erm", 0, ctx.mlp_steps)
                per[T] = auroc(y[tst], s)
            res[f"{mname}|{gname}"] = per
            log(f"  {mname:8s} {gname:16s} mean LODO AUROC={np.nanmean(list(per.values())):.3f} "
                f"worst={np.nanmin(list(per.values())):.3f}")
    ctx.save("e1c_identifiers.json", res)
    return res
