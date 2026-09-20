"""
Corpus loading, family normalisation, feature sets and preprocessing.

The corpus is the directory written by data_prep/sample_corpus.py:
    <corpus>/<domain folder>/<source file>.csv.gz
    <corpus>/manifest.json
Every gz file keeps the original CICFlowMeter header and byte-exact lines.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Domains
# ----------------------------------------------------------------------------
DOMAIN_SHORT = {
    "CIC-BCCC-NRC-ACI-IOT-2023": "ACI-IoT-23",
    "CIC-BCCC-NRC-Edge-IIoTSet-2022": "Edge-IIoT-22",
    "CIC-BCCC-NRC-IoMT-2024": "CICIoMT-24",
    "CIC-BCCC-NRC-IoT-2022": "CICIoT-22",
    "CIC-BCCC-NRC-IoT-2023-Original Training and Testing": "CICIoT-23",
    "CIC-BCCC-NRC-IoT-HCRL-2019": "IoT-HCRL-19",
    "CIC-BCCC-NRC-MQTTIoT-IDS-2020": "MQTT-IoT-20",
    "CIC-BCCC-NRC-TONIoT-2021": "TON-IoT-21",
    "CIC-BCCC-NRC-UQ-IOT-2022": "UQ-IoT-22",
}

# ----------------------------------------------------------------------------
# Canonical attack families (used for family-controlled experiments).
# Mapping is by the "Attack Name" column, which the CIC re-release fills in.
# ----------------------------------------------------------------------------
FAMILY_MAP = {
    "Benign Traffic": "Benign",
    # volumetric TCP
    "DoS SYN Flood": "SYN/TCP flood", "DDoS TCP SYN Flood": "SYN/TCP flood",
    "DoS TCP Flood": "SYN/TCP flood", "Mirai ACK Flood": "SYN/TCP flood",
    "DDoS ACK Fragmentation": "SYN/TCP flood", "DDoS PSHACK Flood": "SYN/TCP flood",
    "DDoS RSTFIN Flood": "SYN/TCP flood", "ACK Flood": "SYN/TCP flood",
    "SYN Flood": "SYN/TCP flood",
    # volumetric UDP / ICMP / app-layer
    "DoS UDP Flood": "UDP flood", "DDoS UDP Flood": "UDP flood",
    "Mirai UDP Flood": "UDP flood", "Mirai UDP Plain": "UDP flood",
    "DoS ICMP Flood": "ICMP flood", "DDoS ICMP Flood": "ICMP flood",
    "DDoS ICMP Fragmentation": "ICMP flood",
    "DoS DNS Flood": "DNS flood",
    "DDoS HTTP Flood": "HTTP flood", "Mirai HTTP Flood": "HTTP flood",
    # reconnaissance
    "Recon Port Scan": "Port scan", "Port Scanning": "Port scan",
    "Scan Host Port": "Port scan", "Scan Aggressive": "Port scan",
    "Scan UDP Attack": "Port scan",
    "Recon Host Discovery": "Host discovery", "Recon Ping Sweep": "Host discovery",
    "Recon OS Scan": "OS scan", "OS Fingerprinting": "OS scan", "Scan Port OS": "OS scan",
    "Recon Vulnerability Scan": "Vulnerability scan", "Vulnerability Scanner": "Vulnerability scan",
    # spoofing
    "MITM ARP Spoofing": "ARP spoofing/MITM", "MITM": "ARP spoofing/MITM",
    # credential attacks
    "Dictionary Brute Force": "Brute force", "Mirai Host Brute Force": "Brute force",
    "Telnet Brute Force": "Brute force", "Password Attack": "Brute force",
    "MQTT Brute Force": "Brute force", "Sparta SSH Brute Force": "Brute force",
    # application / malware
    "XSS": "Web attack", "SQL Injection": "Web attack", "Uploading Attack": "Web attack",
    "Backdoor": "Malware", "Ransomware": "Malware",
    "MQTT DDoS Publish Flood": "MQTT abuse", "MQTT DoS Connect Flood": "MQTT abuse",
    "MQTT DoS Publish Flood": "MQTT abuse", "MQTT Malformed": "MQTT abuse",
}


def canonical_family(name: str) -> str:
    """Map a CIC 'Attack Name' to a canonical cross-dataset family.
    Unknown names (e.g. files only present in a later corpus build) fall back
    to keyword rules so the pipeline never silently drops them."""
    name = str(name).strip()
    if name in FAMILY_MAP:
        return FAMILY_MAP[name]
    s = name.lower()
    for kw, fam in (("benign", "Benign"), ("mqtt", "MQTT abuse"), ("brute", "Brute force"),
                    ("syn", "SYN/TCP flood"), ("tcp", "SYN/TCP flood"), ("ack flood", "SYN/TCP flood"),
                    ("udp", "UDP flood"), ("icmp", "ICMP flood"), ("scan", "Port scan")):
        if kw in s:
            return fam
    return "Other"


ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Device"]
META_COLS = ["Attack Name", "Label"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def list_corpus_files(corpus: str):
    out = []
    for p in sorted(glob.glob(os.path.join(corpus, "*", "*.csv.gz"))):
        dom = os.path.basename(os.path.dirname(p))
        if dom in DOMAIN_SHORT:
            out.append((dom, p))
    return out


def load_corpus(corpus: str, cap_attack_file: int | None = None,
                cap_benign_domain: int | None = None, seed: int = 0,
                keep_raw_text: bool = False) -> pd.DataFrame:
    """Load every sampled file, optionally down-sampling further.

    Down-sampling caps are applied per *attack file* (so no single campaign
    dominates) and per *domain* for benign flows.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for dom, path in list_corpus_files(corpus):
        df = pd.read_csv(path, low_memory=False)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        benign = os.path.basename(path).lower().startswith("benign")
        cap = cap_benign_domain if benign else cap_attack_file
        if cap is not None and len(df) > cap:
            df = df.iloc[np.sort(rng.choice(len(df), cap, replace=False))]
        df = df.copy()
        df["domain"] = DOMAIN_SHORT[dom]
        df["source_file"] = os.path.basename(path)[:-7]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *.csv.gz files under {corpus}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["y"] = pd.to_numeric(df["Label"], errors="coerce").fillna(1).astype(int).clip(0, 1)
    df["family"] = df["Attack Name"].map(canonical_family)
    df.loc[df.y == 0, "family"] = "Benign"
    return df


def load_manifest(corpus: str) -> dict:
    p = os.path.join(corpus, "manifest.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {"files": []}


# ----------------------------------------------------------------------------
# Feature sets
# ----------------------------------------------------------------------------
def flow_feature_columns(df: pd.DataFrame) -> list[str]:
    """CICFlowMeter flow statistics: everything except identifiers/meta."""
    skip = set(ID_COLS) | set(META_COLS) | {"domain", "source_file", "y", "family"}
    cols = [c for c in df.columns if c not in skip]
    num = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.99:
                df[c] = conv
                num.append(c)
    # drop constant columns
    return [c for c in num if df[c].nunique(dropna=True) > 1]


def _ip_octets(s: pd.Series, prefix: str) -> pd.DataFrame:
    parts = s.astype(str).str.split(".", expand=True).iloc[:, :4]
    parts = parts.apply(pd.to_numeric, errors="coerce").fillna(-1)
    parts.columns = [f"{prefix}_o{i+1}" for i in range(parts.shape[1])]
    return parts


def identifier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric encodings of identifiers that naive pipelines often keep."""
    out = pd.DataFrame(index=df.index)
    out["src_port"] = pd.to_numeric(df["Src Port"], errors="coerce").fillna(-1)
    out["dst_port"] = pd.to_numeric(df["Dst Port"], errors="coerce").fillna(-1)
    out = pd.concat([out, _ip_octets(df["Src IP"], "src_ip"),
                     _ip_octets(df["Dst IP"], "dst_ip")], axis=1)
    ts = df["Timestamp"].astype(str)
    hour = ts.str.extract(r"\s(\d{1,2}):")[0].astype(float)
    pm = ts.str.contains("PM", na=False) & (hour < 12)
    out["ts_hour"] = (hour + 12 * pm).fillna(-1)
    out["ts_year"] = ts.str.extract(r"/(\d{2,4})\s")[0].astype(float).fillna(-1)
    out.loc[out.ts_year < 100, "ts_year"] += 2000
    return out


def build_matrices(df: pd.DataFrame):
    """Return (X_flow DataFrame, X_id DataFrame)."""
    fcols = flow_feature_columns(df)
    Xf = df[fcols].astype(float)
    Xf = Xf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xi = identifier_features(df)
    return Xf, Xi


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
class SignedLogScaler:
    """sign(x)*log1p(|x|) followed by standardisation fitted on train only."""

    def fit(self, X: np.ndarray):
        Z = np.sign(X) * np.log1p(np.abs(X))
        self.mu = Z.mean(0)
        self.sd = Z.std(0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.sign(X) * np.log1p(np.abs(X))
        return np.clip((Z - self.mu) / self.sd, -10, 10).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def per_domain_standardise(X: np.ndarray, domains: np.ndarray) -> np.ndarray:
    """Test-time normalisation: standardise each domain by its own statistics
    (uses *unlabelled* target data; an adaptation baseline, not DG)."""
    Z = np.sign(X) * np.log1p(np.abs(X))
    out = np.empty_like(Z, dtype=np.float32)
    for d in np.unique(domains):
        m = domains == d
        mu, sd = Z[m].mean(0), Z[m].std(0) + 1e-6
        out[m] = np.clip((Z[m] - mu) / sd, -10, 10)
    return out
