#!/usr/bin/env python3
"""
sample_corpus.py  --  Step 0 of the LODO trustworthiness audit.

Streams every CSV of CIC-BCCC-NRC-TabularIoTAttacks-2024 once and writes a
compact, *byte-exact* uniform random sample of each file (reservoir sampling,
Algorithm R on raw lines, so no value is re-formatted or re-parsed).

    python sample_corpus.py            # run inside the dataset root folder
    python sample_corpus.py  D:\\path\\to\\CIC-BCCC-NRC-TabularIoTAttacks-2024

Standard library only. Never modifies or deletes source files.
Output:   <root>/_corpus/<domain>/<file>.csv.gz   (header + sampled lines)
          <root>/_corpus/manifest.json            (exact row counts per file)
          <root>/_corpus/progress.log             (live progress)
          <root>/_corpus/DONE                     (written last)
"""
import gzip
import json
import os
import random
import sys
import time

SEED = 2027
CAP_ATTACK = 20_000     # max sampled flows per attack file
CAP_BENIGN = 60_000     # max sampled flows per benign file


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, "_corpus")
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "progress.log"), "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    domains = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and d.startswith("CIC-BCCC-NRC-"))
    manifest = {"seed": SEED, "cap_attack": CAP_ATTACK, "cap_benign": CAP_BENIGN, "files": []}
    t0 = time.time()
    for d in domains:
        ddir = os.path.join(root, d)
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for f in sorted(os.listdir(ddir)):
            if not f.lower().endswith(".csv") or ".part" in f.lower():
                continue
            path = os.path.join(ddir, f)
            benign = f.lower().startswith("benign")
            cap = CAP_BENIGN if benign else CAP_ATTACK
            rng = random.Random(f"{SEED}|{d}|{f}")
            keep, n = [], 0
            with open(path, "rb") as fh:
                header = fh.readline()
                for line in fh:
                    if len(line) < 3:
                        continue
                    n += 1
                    if n <= cap:
                        keep.append(line)
                    else:
                        j = int(rng.random() * n)
                        if j < cap:
                            keep[j] = line
            with gzip.open(os.path.join(out, d, f + ".gz"), "wb", compresslevel=6) as gz:
                gz.write(header)
                gz.writelines(keep)
            manifest["files"].append({"domain": d, "file": f, "bytes": os.path.getsize(path),
                                      "rows_total": n, "rows_sampled": len(keep),
                                      "benign": benign})
            say(f"[{time.time()-t0:7.0f}s] {d}/{f}: {n:,} rows -> {len(keep):,}")
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    open(os.path.join(out, "DONE"), "w").write("ok\n")
    say(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
