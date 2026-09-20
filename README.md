# Do IoT Intrusion Detectors Learn Attacks or Datasets?

### A Cross-Domain Trustworthiness Study — code for the IEEE SaTML 2027 submission

This repository reproduces every number, table and figure of the paper from the
public **CIC-BCCC-NRC TabularIoTAttack-2024** release (nine IoT datasets
re-extracted with CICFlowMeter). It needs only a CPU: NumPy, pandas,
scikit-learn, SciPy and matplotlib. No deep-learning framework is required.
The DG objectives (GroupDRO, V-REx, CORAL, DANN and our C2DA) are written in
NumPy with hand-derived, gradient-checked backward passes.

```
lodo\_audit/
├── data\_prep/sample\_corpus.py   step 0: stream the raw CSVs -> compact byte-exact sample (\_corpus/)
├── run\_all.py                   MAIN ENTRY POINT: runs every experiment + draws every figure
├── iotaudit/
│   ├── data.py                  loading, family mapping, feature sets, preprocessing
│   ├── metrics.py               AUROC, DGG, DFI, identity-only null, frozen thresholds, host-pair bootstrap
│   ├── fingerprint.py           feature information map (D\_j, L\_j, phi\_j) and FAFP (nested selection)
│   ├── models.py                NumPy MLP with ERM / GroupDRO / V-REx / CORAL / DANN / C2DA
│   ├── experiments.py           E0-E5 (see table below)
│   └── figures.py               all paper figures (PDF + PNG)
├── tests/test\_gradients.py      finite-difference check of the alignment gradients
├── paper/
│   ├── main.tex, refs.bib       the paper (IEEEtran conference, SaTML format)
│   ├── make\_paper\_assets.py     writes paper/generated/\*.tex (all numbers + tables) from results
│   ├── generated/               auto-generated numbers.tex and tables (do not edit by hand)
│   └── figures/                 figures copied from the run
└── results\_final/               the results used in the submitted paper (JSON/CSV + figures)
```

\---

## 1\. Install (once)

Python 3.9 or later. On Windows, open **Command Prompt** or **PowerShell** in this folder:

```bash
python -m pip install -r requirements.txt
```

## 2\. Build the compact corpus (once, about 5–20 min, standard library only)

The raw release is about 11.5 GB in 69 CSVs. This step streams every file once and keeps a
uniform random sample of each (≤20 000 flows per attack file, ≤60 000 per benign file).
The lines are copied **byte-exactly**, so formatting artefacts are preserved.
The raw files are never modified.

```bash
python data\_prep/sample\_corpus.py "C:\\Users\\<you>\\Desktop\\CIC-BCCC-NRC-TabularIoTAttacks-2024"
```

This writes `CIC-BCCC-NRC-TabularIoTAttacks-2024\\\_corpus\\` (about 70–100 MB) and a
`manifest.json` with the exact row count of every source file.
The seed is fixed, so two runs give identical output.

## 3\. Run the audit (main code)

```bash
# full paper run (\~2-2.5 h on a 2-core laptop; 3 seeds for the MLP experiments)
python run\_all.py --corpus "C:\\Users\\<you>\\Desktop\\CIC-BCCC-NRC-TabularIoTAttacks-2024\\\_corpus" --out results\_run

# \~20-minute smoke test (smaller caps, 1 seed)
python run\_all.py --corpus "...\\\_corpus" --out results\_quick --quick

# selected experiments only, or only re-draw the figures
python run\_all.py --corpus "...\\\_corpus" --out results\_run --only e1 e2
python run\_all.py --out results\_run --figures-only
```

Useful options: `--seeds 5`, `--cap-attack 2500` (flows per attack file),
`--cap-benign 10000` (benign flows per dataset) and `--mlp-steps 3000`.

|Key|Experiment|Paper section|
|-|-|-|
|`e0`|census; serialisation fingerprints; label conflicts; duplicates|IV (corpus)|
|`e1`|random split (pooled / per-dataset) vs. LODO; DGG; transfer matrix; frozen thresholds|VI-A, VI-E|
|`e1b`|exact pair decomposition of pooled AUROC; identity-only null|V-B, VI-A|
|`e1c`|identifier ablation: ports / IP octets / timestamp / all (HGB and MLP)|VI-A|
|`e2`|dataset-of-origin probes (DFI), family-controlled, precision, minimal fingerprint, t-SNE|VI-B|
|`e3`|per-feature information map D\_j, L\_j, phi\_j|VI-C|
|`e4`|Fingerprint-Aware Feature Pruning, D and phi rankings (nested, target-blind) vs. controls|VI-D|
|`e5`|DG objectives (ERM, GroupDRO, V-REx, CORAL, DANN, C2DA, TTN, +FAFP-phi, +FAFP-D)|VI-D|
|`e5b`|oracle sensitivity to the alignment weight lambda|Appendix|

Outputs: `results\_run/results/\*.json|csv` and `results\_run/figures/\*.pdf|png`.

## 4\. Regenerate the paper

```bash
python paper/make\_paper\_assets.py --run results\_run    # writes paper/generated/\*.tex and copies figures
```

Then compile `paper/` with the official **IEEEtran** class. On Overleaf, upload the `paper/` folder and
compile `main.tex`. Locally, run `pdflatex main \&\& bibtex main \&\& pdflatex main \&\& pdflatex main`.
No number in the paper is typed by hand; every number is a macro in `generated/numbers.tex`.

## 5\. Tests

```bash
python -m tests.test\_gradients
```

\---

### Data coverage

The submitted results use **all 69 CSV files (22.9 million flows)** of the release, including the eight files
larger than 400 MB. `sample\_corpus.py` streams each file once, so memory use stays small even for the
1.9 GB files. `results\_final/corpus\_manifest.json` records the exact row count of every file.

### Key definitions (see paper Sec. V)

* **DGG**: per-dataset AUROC under a random split minus AUROC on the same dataset when held out (LODO).
* **Identity-only null**: pooled AUROC of the feature-free score *s(x) = attack prevalence of x's dataset*.
By construction it is exactly 0.5 within any single dataset.
* **DFI**: chance-corrected balanced accuracy of a dataset-of-origin classifier, `(BA − 1/K)/(1 − 1/K)`.
* **phi\_j = D\_j − L\_j**: dataset information given the class, minus label information given the dataset.
* **FAFP**: remove the top-q features ranked by D\_j (FAFP-D) or phi\_j (FAFP-phi). q is chosen by nested LODO inside the training datasets only.
* **C2DA**: class-conditional alignment of the first and second moments of the representation across datasets.
It is well defined when some datasets contain a single class.

