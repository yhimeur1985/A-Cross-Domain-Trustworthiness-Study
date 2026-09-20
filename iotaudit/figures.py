"""Draw every figure of the paper from the saved JSON/CSV results."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GRAY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e0"
CAT9 = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, GRAY]
MARK9 = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
COL1 = 3.45   # IEEE single-column width (in)
COL2 = 7.1


def _load(res, name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        return None
    if p.endswith(".json"):
        return json.load(open(p, encoding="utf-8"))
    return pd.read_csv(p)


def _save(fig, figs, name):
    fig.savefig(os.path.join(figs, name + ".pdf"))
    fig.savefig(os.path.join(figs, name + ".png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
def fig_census(res, figs):
    c = _load(res, "e0_census.csv")
    if c is None:
        return
    c = c.sort_values("rows_total")
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    yy = np.arange(len(c))
    att = c.rows_total - c.benign_rows_total
    ax.barh(yy + 0.2, att, height=0.38, color=ORANGE, label="attack flows")
    ben = c.benign_rows_total.replace(0, np.nan)
    ax.barh(yy - 0.2, ben, height=0.38, color=BLUE, label="benign flows")
    for i, (d, b) in enumerate(zip(c.domain, c.benign_rows_total)):
        if b == 0:
            msg = "no benign traffic in release"
            ax.text(1.05e2, i - 0.24, msg, va="center", fontsize=5.2, color=RED)
    ax.set_xscale("log")
    ax.set_xlim(1e2, 3e6)
    ax.set_yticks(yy, c.domain)
    ax.set_xlabel("flows in the release (log scale)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    _save(fig, figs, "fig_census")


def fig_protocols(res, figs):
    r = _load(res, "e1_protocols.json")
    if r is None:
        return
    key = "HGB|flow"
    rnd = r["random"][key]["per_domain"]
    lodo = r["lodo"][key]
    doms = [d for d in rnd if rnd[d] is not None and lodo[d]["auroc"] is not None
            and np.isfinite(rnd[d]) and np.isfinite(lodo[d]["auroc"])]
    doms = sorted(doms, key=lambda d: lodo[d]["auroc"])
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for i, d in enumerate(doms):
        a, b = rnd[d], lodo[d]["auroc"]
        lo, hi = lodo[d]["ci"]
        ax.plot([b, a], [i, i], color=GRAY, lw=1.2, zorder=1)
        ax.errorbar(b, i, xerr=[[b - lo], [hi - b]], fmt="o", color=ORANGE, ms=5,
                    capsize=2, lw=1, zorder=3, label="leave-one-dataset-out" if i == 0 else None)
        ax.plot(a, i, "o", color=BLUE, ms=5, zorder=3,
                label="random split (same dataset)" if i == 0 else None)
        ax.text(1.005, i, f"$-${a-b:.2f}", va="center", fontsize=6.5, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.axvline(r["identity_null"]["random_test_pooled"], color=RED, lw=0.9, ls="--")
    ax.text(r["identity_null"]["random_test_pooled"] - 0.005, -0.45, "identity-only null ",
            fontsize=6, color=RED, va="bottom", ha="right")
    ax.set_yticks(range(len(doms)), doms)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(-0.6, len(doms) - 0.4)
    ax.set_xlabel("AUROC on the dataset (HGB, flow features)")
    ax.legend(loc="lower center", frameon=False, bbox_to_anchor=(0.45, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    _save(fig, figs, "fig_protocols")


def fig_transfer(res, figs):
    m = _load(res, "e1_transfer_matrix.csv")
    if m is None:
        return
    m = m.set_index(m.columns[0])
    fig, ax = plt.subplots(figsize=(COL1 * 0.95, 2.5))
    im = ax.imshow(m.values.astype(float), cmap="Blues", vmin=0.3, vmax=1.0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = float(m.values[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.75 else INK)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=35, ha="right")
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.set_xlabel("test dataset")
    ax.set_ylabel("training dataset")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="AUROC")
    _save(fig, figs, "fig_transfer")


def fig_confusion(res, figs):
    r = _load(res, "e2_fingerprint.json")
    if r is None:
        return
    labs = r["confusion"]["labels"]
    M = np.array(r["confusion"]["matrix"])
    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if M[i, j] >= 0.01:
                ax.text(j, i, f"{M[i,j]:.2f}".lstrip("0") if M[i, j] < 1 else "1",
                        ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right")
    ax.set_yticks(range(len(labs)), labs)
    ax.set_xlabel("predicted dataset of origin")
    ax.set_ylabel("true dataset of origin")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="row-normalised rate")
    _save(fig, figs, "fig_confusion")


def fig_dfi(res, figs):
    r = _load(res, "e2_fingerprint.json")
    if r is None:
        return
    cond = r["conditions"]
    fam = {k.split("|")[0].replace("family=", "").strip(): v for k, v in cond.items()
           if k.startswith("family=")}
    fam = dict(sorted(fam.items(), key=lambda kv: kv[1]["dfi"]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    names = list(fam)
    vals = [fam[n]["dfi"] for n in names]
    ax.barh(range(len(names)), vals, color=BLUE, height=0.6)
    for i, n in enumerate(names):
        ax.text(vals[i] + 0.005, i, f"{vals[i]:.2f}  (K={fam[n]['k']})", va="center",
                fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Domain Fingerprint Index (0 = no fingerprint, 1 = perfect)")
    ax.set_title("(a) Same attack family, different datasets", loc="left")
    ax.grid(axis="y", visible=False)
    mf = r["minimal_fingerprint"]
    ks = [m["k"] for m in mf]
    ax2.plot(ks, [m["dfi"] for m in mf], "-o", color=ORANGE, ms=4, lw=1.6)
    full = cond["all flows | flow"]["dfi"]
    ax2.axhline(full, color=BLUE, ls="--", lw=1)
    ax2.text(ks[-1], full - 0.03, "all 71 flow features", ha="right", va="top", fontsize=6.5,
             color=BLUE)
    ax2.set_xscale("log")
    ax2.set_xticks(ks, [str(k) for k in ks])
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("number of flow features used")
    ax2.set_ylabel("DFI (9 datasets)")
    ax2.set_title("(b) Minimal fingerprint", loc="left")
    _save(fig, figs, "fig_dfi")


def fig_tsne(res, figs):
    t = _load(res, "e2_tsne.csv")
    if t is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9))
    ax = axes[0]
    for lab, col, name in ((0, BLUE, "benign"), (1, ORANGE, "attack")):
        s = t[t.label == lab]
        ax.scatter(s.x, s.y, s=2.5, c=col, alpha=0.55, lw=0, label=name, rasterized=True)
    ax.set_title("(a) coloured by label", loc="left")
    ax.legend(markerscale=4, frameon=False, loc="upper right")
    ax = axes[1]
    for i, d in enumerate(sorted(t.domain.unique())):
        s = t[t.domain == d]
        ax.scatter(s.x, s.y, s=3, c=CAT9[i], marker=MARK9[i], alpha=0.6, lw=0, label=d,
                   rasterized=True)
    ax.set_title("(b) coloured by dataset of origin", loc="left")
    ax.legend(markerscale=3, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    _save(fig, figs, "fig_tsne")


def fig_feature_map(res, figs):
    t = _load(res, "e3_feature_map.csv")
    if t is None:
        return
    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    ax.scatter(t.L, t.D, s=10, c=np.where(t.phi > 0.1, ORANGE, BLUE), lw=0, alpha=0.85)
    lim = max(t.D.max(), t.L.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls="--")
    ax.text(lim * 0.97, lim * 0.9, "$D=L$", color=GRAY, fontsize=6.5, ha="right")
    lab = pd.concat([t.sort_values("phi", ascending=False).head(4),
                     t.sort_values("phi").head(2)])
    lab = lab.sort_values("D", ascending=False)
    ys = []
    for _, r in lab.iterrows():          # stagger labels vertically to avoid collisions
        y = r.D
        while any(abs(y - y0) < 0.028 for y0 in ys):
            y -= 0.028
        ys.append(y)
        ax.annotate(r.feature, (r.L, r.D), xytext=(r.L + 0.05, y), textcoords="data",
                    fontsize=5.5, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5))
    ax.set_xlabel("label information within datasets  $L_j$")
    ax.set_ylabel("dataset information within class  $D_j$")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.text(0.02, 0.97, "fingerprint-dominated ($\\varphi_j>0.1$)", color=ORANGE,
            transform=ax.transAxes, fontsize=6.5, va="top")
    _save(fig, figs, "fig_feature_map")


def fig_pruning(res, figs):
    r = _load(res, "e4_pruning.json")
    if r is None:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.3))
    styles = {"phi": (ORANGE, "o", "FAFP-$\\varphi$ ranking (fingerprint excess)"),
              "D": (BLUE, "s", "FAFP-$D$ ranking (dataset information)"),
              "random": (GRAY, "^", "prune at random")}
    for rule, (col, mk, lab) in styles.items():
        cur = r["curve"][rule]
        fr = sorted(cur, key=float)
        m = [np.nanmean([v for v in cur[f].values() if v is not None]) for f in fr]
        ax.plot([float(f) for f in fr], m, marker=mk, color=col, lw=1.6, ms=4, label=lab)
    ax.set_xlabel("fraction of flow features removed")
    ax.set_ylabel("mean LODO AUROC (two-class datasets)")
    ax.set_title("(a) Transfer to an unseen dataset", loc="left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    cost = r["in_distribution_cost"]
    fr = sorted(cost, key=float)
    ax2.plot([float(f) for f in fr], [cost[f]["per_domain_mean"] for f in fr], "-o",
             color=ORANGE, ms=4, lw=1.6, label="per-dataset (random split)")
    ax2.plot([float(f) for f in fr], [cost[f]["pooled"] for f in fr], "--s",
             color=BLUE, ms=3.5, lw=1.2, label="pooled (random split)")
    ax2.set_xlabel("fraction of flow features removed (by $\\varphi$)")
    ax2.set_ylabel("in-distribution AUROC")
    ax2.set_title("(b) Cost on the conventional benchmark", loc="left")
    ax2.legend(frameon=False, loc="lower left")
    _save(fig, figs, "fig_pruning")


NICE = {"erm": "ERM", "erm_gb": "ERM (group-bal.)", "groupdro": "GroupDRO",
        "vrex": "V-REx", "coral": "CORAL", "dann": "DANN", "c2da": "C2DA (ours)"}


def dg_table(r, two_class, all_domains):
    rows = []
    for key, seeds in r.items():
        m, fs = key.split("|")
        a = [np.nanmean([seeds[s][d]["auroc"] for d in two_class]) for s in seeds]
        worst = [np.nanmin([seeds[s][d]["auroc"] for d in two_class]) for s in seeds]
        tp = [np.nanmean([seeds[s][d]["tpr_frozen"] for d in all_domains]) for s in seeds]
        fp = [np.nanmean([seeds[s][d]["fpr_frozen"] for d in two_class]) for s in seeds]
        name = NICE.get(m, m) + {"flow": "", "fafp": " + FAFP-$\\varphi$", "fafpD": " + FAFP-$D$", "ttn": " + TTN"}[fs]
        rows.append({"method": name, "key": key, "auroc": np.mean(a), "auroc_sd": np.std(a),
                     "worst": np.mean(worst), "tpr_frozen": np.mean(tp),
                     "fpr_frozen": np.mean(fp), "n_seeds": len(a)})
    return pd.DataFrame(rows)


def fig_dg(res, figs):
    r = _load(res, "e5_dg.json")
    meta = _load(res, "run_meta.json")
    if r is None or meta is None:
        return
    t = dg_table(r, meta["two_class_domains"], meta["domains"])
    t.to_csv(os.path.join(res, "e5_dg_table.csv"), index=False)
    t = t.sort_values("auroc")
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    cols = [ORANGE if ("C2DA" in m or "FAFP" in m) else (GRAY if "TTN" in m else BLUE)
            for m in t.method]
    ax.barh(range(len(t)), t.auroc, xerr=t.auroc_sd, color=cols, height=0.62,
            error_kw={"lw": 0.8, "capsize": 2, "ecolor": INK2})
    for i, (a, w) in enumerate(zip(t.auroc, t.worst)):
        ax.plot(w, i, "|", color=INK, ms=7, mew=1.2)
    ax.set_yticks(range(len(t)), t.method)
    ax.set_xlim(0.35, max(0.8, t.auroc.max() + 0.05))
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("mean LODO AUROC   (| = worst held-out dataset)")
    ax.grid(axis="y", visible=False)
    _save(fig, figs, "fig_dg")


def fig_threshold(res, figs):
    r = _load(res, "e1_protocols.json")
    if r is None:
        return
    lodo = r["lodo"]["HGB|flow"]
    doms = [d for d in lodo if not d.startswith("_")]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    xs = np.arange(len(doms))
    fpr = [lodo[d]["fpr_frozen"] if lodo[d]["fpr_frozen"] is not None else np.nan for d in doms]
    tpr = [lodo[d]["tpr_frozen"] for d in doms]
    ax.bar(xs - 0.2, tpr, 0.38, color=ORANGE, label="detection rate (TPR)")
    ax.bar(xs + 0.2, fpr, 0.38, color=BLUE, label="false-alarm rate (FPR)")
    for i, f in enumerate(fpr):
        if not np.isfinite(f):
            ax.text(xs[i] + 0.2, 0.02, "n/a", rotation=90, fontsize=5.5, color=INK2,
                    ha="center")
    ax.axhline(0.01, color=RED, lw=0.9, ls="--")
    ax.text(-0.5, 0.035, " intended FPR = 1%", color=RED, fontsize=6, ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.set_xticks(xs, doms, rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate at frozen source threshold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.grid(axis="x", visible=False)
    _save(fig, figs, "fig_threshold")


def draw_all(out):
    res = os.path.join(out, "results")
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)
    for f in (fig_census, fig_protocols, fig_transfer, fig_confusion, fig_dfi, fig_tsne,
              fig_feature_map, fig_pruning, fig_dg, fig_threshold):
        try:
            f(res, figs)
        except Exception as e:  # keep drawing the others
            print(f"[figures] {f.__name__} failed: {e!r}")
