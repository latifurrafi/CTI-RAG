# chunkers.py v6 — Simple, reliable chunks
# ONE chunk per CVE. No splitting. No fragments.
# Retrieval: full enriched text. Generation: same full text.
# This is the correct approach for 40-word atomic documents.
import pickle, numpy as np, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import SEMANTIC_MODEL, MIN_CHUNK_WORDS, FIXED_INDEX_FILE, SEMANTIC_INDEX_FILE

def _meta(row):
    return f"{row['id']} {row['severity']} {row['cwe']}"

def build_fixed_chunks(df):
    """Fixed = one full enriched CVE text per chunk (baseline)."""
    print("[chunker] Building fixed chunks (1 per CVE) ...")
    chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = f"{_meta(row)} {row['description']}"
        chunks.append({"cve_id": row["id"], "chunk_id": f"{row['id']}_F0",
                        "text": text, "parent_text": text,
                        "severity": row["severity"], "cvss_score": row["cvss_score"]})
    print(f"[chunker] Fixed chunks: {len(chunks):,}")
    with open(FIXED_INDEX_FILE,"wb") as f: pickle.dump(chunks,f)
    return chunks

def build_semantic_chunks(df):
    """
    Semantic = sentence-level chunks for retrieval precision.
    Every chunk keeps CVE ID prefix. parent_text = full CVE.
    """
    from config import COSINE_THRESHOLD
    print(f"[chunker] Building semantic chunks ...")
    enc = SentenceTransformer(SEMANTIC_MODEL)
    chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Semantic"):
        prefix = _meta(row)
        desc   = row["description"]
        parent = f"{prefix} {desc}"
        # split into sentences
        sents  = [s.strip() for s in desc.split(".") if len(s.strip()) > 8]
        if len(sents) < 3:
            # short CVE — keep as one chunk
            chunks.append({"cve_id": row["id"], "chunk_id": f"{row['id']}_S0",
                            "text": parent, "parent_text": parent,
                            "severity": row["severity"], "cvss_score": row["cvss_score"]})
            continue
        # semantic split
        embs = enc.encode(sents, normalize_embeddings=True, show_progress_bar=False)
        segs, cur = [], [sents[0]]
        for i in range(1, len(sents)):
            if float(np.dot(embs[i-1], embs[i])) < COSINE_THRESHOLD:
                segs.append(". ".join(cur))
                cur = []
            cur.append(sents[i])
        if cur: segs.append(". ".join(cur))
        for j, seg in enumerate(segs):
            text = f"{prefix} {seg}"
            if len(text.split()) >= MIN_CHUNK_WORDS:
                chunks.append({"cve_id": row["id"], "chunk_id": f"{row['id']}_S{j}",
                                "text": text, "parent_text": parent,
                                "severity": row["severity"], "cvss_score": row["cvss_score"]})
    # verify
    missing = sum(1 for c in chunks if c["cve_id"] not in c["text"])
    print(f"[chunker] Semantic chunks: {len(chunks):,}  Missing CVE ID: {missing}")
    with open(SEMANTIC_INDEX_FILE,"wb") as f: pickle.dump(chunks,f)
    return chunks

def load_chunks(t="semantic"):
    p = SEMANTIC_INDEX_FILE if t=="semantic" else FIXED_INDEX_FILE
    with open(p,"rb") as f: c = pickle.load(f)
    print(f"[chunker] Loaded {len(c):,} {t} chunks"); return c

def chunk_stats(chunks, label=""):
    L = np.array([len(c["text"].split()) for c in chunks])
    u = len(set(c["cve_id"] for c in chunks))
    m = sum(1 for c in chunks if c["cve_id"] not in c["text"])
    print(f"\n[chunker stats] {label}")
    print(f"  Total:{len(chunks):,}  Unique CVEs:{u:,}  Avg:{L.mean():.0f}w  Missing CVE ID:{m}")
