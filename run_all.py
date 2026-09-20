#!/usr/bin/env python3
"""
Do IoT Intrusion Detectors Learn Attacks or Datasets?
Main entry point: runs every experiment of the paper and draws every figure.

Usage
-----
    # 1) build the compact corpus once (reads the raw CIC folder, stdlib only)
    python data_prep/sample_corpus.py  "C:/path/to/CIC-BCCC-NRC-TabularIoTAttacks-2024"

    # 2) run the audit (~1.5 h on a 2-core laptop; --quick ~20 min)
    python run_all.py --corpus "C:/path/to/CIC-BCCC-NRC-TabularIoTAttacks-2024/_corpus" --out results_run

    # only some experiments, or only redraw figures from saved results
    python run_all.py --corpus ... --out ... --only e1 e2
    python run_all.py --out results_run --figures-only
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iotaudit import experiments as E  # noqa: E402
from iotaudit import figures as F      # noqa: E402

ALL = ["e0", "e1", "e1b", "e1c", "e2", "e3", "e4", "e5", "e5b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="folder written by data_prep/sample_corpus.py")
    ap.add_argument("--out", default="results_run")
    ap.add_argument("--only", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the MLP experiments")
    ap.add_argument("--cap-attack", type=int, default=2500, help="max flows per attack file")
    ap.add_argument("--cap-benign", type=int, default=10000, help="max benign flows per domain")
    ap.add_argument("--mlp-steps", type=int, default=3000)
    ap.add_argument("--quick", action="store_true", help="small caps, 1 seed: smoke test")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()

    if not a.figures_only:
        if not a.corpus:
            ap.error("--corpus is required unless --figures-only")
        if a.quick:
            a.cap_attack, a.cap_benign, a.seeds, a.mlp_steps = 600, 3000, 1, 800
        ctx = E.Ctx(corpus=a.corpus, out=a.out, cap_attack_file=a.cap_attack,
                    cap_benign_domain=a.cap_benign, seeds=tuple(range(a.seeds)),
                    mlp_steps=a.mlp_steps, quick=a.quick,
                    n_boot=50 if a.quick else 200).setup()
        meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a),
                "python": platform.python_version(), "numpy": np.__version__,
                "n_flows": int(len(ctx.y)), "domains": ctx.domains,
                "two_class_domains": ctx.two_class, "flow_features": ctx.flow_names}
        json.dump(meta, open(os.path.join(ctx.res_dir, "run_meta.json"), "w"), indent=1)
        t0 = time.time()
        steps = {"e0": E.e0_census, "e1": E.e1_protocols, "e1b": E.e1b_decomposition, "e1c": E.e1c_identifier_ablation, "e2": E.e2_fingerprint,
                 "e3": E.e3_feature_map, "e4": E.e4_pruning, "e5": E.e5_dg,
                 "e5b": E.e5_lambda_sensitivity}
        for k in ALL:
            if k in a.only:
                steps[k](ctx)
        E.log(f"all experiments finished in {(time.time()-t0)/60:.1f} min")
        if set(a.only) == set(ALL):
            meta["runtime_min"] = int(round((time.time() - t0) / 60))
            json.dump(meta, open(os.path.join(ctx.res_dir, "run_meta.json"), "w"), indent=1)
    F.draw_all(a.out)
    print(f"\nResults: {os.path.join(a.out, 'results')}\nFigures: {os.path.join(a.out, 'figures')}")


if __name__ == "__main__":
    main()
