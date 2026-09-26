# ─────────────────────────────────────────────────────────────────
# data_loader.py  —  Load and parse NVD CVE 2.0 JSON
# NVD v2.0 structure is different from v1.1:
#   v1.1: data["CVE_Items"][i]["cve"]["description"]["description_data"][0]["value"]
#   v2.0: data["vulnerabilities"][i]["cve"]["descriptions"][j]["value"] (lang == "en")
# ─────────────────────────────────────────────────────────────────

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import NVD_JSON_PATH, MIN_DESC_LENGTH


def load_nvd_v2(json_path: Path = NVD_JSON_PATH) -> pd.DataFrame:
    """
    Parse NVD CVE 2.0 JSON feed into a clean DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        id          — CVE identifier  (e.g. "CVE-2023-12345")
        description — English description text
        severity    — CVSS v3.1 base severity string or "UNKNOWN"
        cvss_score  — CVSS v3.1 base score float or None
        cwe         — first CWE ID string or "UNKNOWN"
        published   — publication date string
    """
    print(f"[data_loader] Loading {json_path} ...")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("vulnerabilities", [])
    print(f"[data_loader] Found {len(items):,} raw vulnerability entries")

    rows = []
    for item in tqdm(items, desc="Parsing CVEs"):
        cve = item.get("cve", {})

        # ── CVE ID ────────────────────────────────────────────────
        cve_id = cve.get("id", "UNKNOWN")

        # ── English description ───────────────────────────────────
        desc = ""
        for d in cve.get("descriptions", []):
            if d.get("lang", "") == "en":
                desc = d.get("value", "").strip()
                break
        if not desc or len(desc) < MIN_DESC_LENGTH:
            continue                        # skip empty / very short

        # ── CVSS v3.1 score & severity ────────────────────────────
        cvss_score = None
        severity   = "UNKNOWN"
        metrics    = cve.get("metrics", {})

        # try cvssMetricV31 first, then V30, then V2
        for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
            metric_list = metrics.get(key, [])
            if metric_list:
                cvss_data  = metric_list[0].get("cvssData", {})
                cvss_score = cvss_data.get("baseScore")
                severity   = cvss_data.get("baseSeverity",
                             cvss_data.get("accessVector", "UNKNOWN"))
                break

        # ── CWE ──────────────────────────────────────────────────
        cwe = "UNKNOWN"
        weaknesses = cve.get("weaknesses", [])
        if weaknesses:
            for wd in weaknesses[0].get("description", []):
                if wd.get("lang", "") == "en":
                    cwe = wd.get("value", "UNKNOWN")
                    break

        # ── Published date ────────────────────────────────────────
        published = cve.get("published", "")[:10]  # keep YYYY-MM-DD only

        rows.append({
            "id":          cve_id,
            "description": desc,
            "severity":    str(severity).upper(),
            "cvss_score":  cvss_score,
            "cwe":         cwe,
            "published":   published,
        })

    df = pd.DataFrame(rows)
    print(f"[data_loader] Clean DataFrame: {len(df):,} CVEs")
    print(f"[data_loader] Severity breakdown:\n{df['severity'].value_counts().to_string()}")
    return df


def get_corpus_texts(df: pd.DataFrame) -> list[str]:
    """
    Returns a list of enriched text strings, one per CVE.
    Enrichment: prepend CVE ID + severity so BM25 can match on them.
    """
    texts = []
    for _, row in df.iterrows():
        enriched = (
            f"{row['id']} {row['severity']} {row['cwe']} "
            f"{row['description']}"
        )
        texts.append(enriched)
    return texts


if __name__ == "__main__":
    df = load_nvd_v2()
    print(df.head(3).to_string())
