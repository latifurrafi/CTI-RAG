# retriever.py v10 — Clean three-system comparison
# System A: Naive RAG    — BM25 only, no dense, no confidence
# System B: Hybrid RAG   — BM25 + Dense + RRF, score-margin confidence
# System C: Conf-Aware   — Same as B + abstention on low confidence
# This is the honest, publishable comparison.

import re, numpy as np, faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import TOP_K, RRF_K, SEMANTIC_MODEL
from config import FAISS_FIXED_INDEX, FAISS_FIXED_EMBEDS

CVE_RE  = re.compile(r'CVE-\d{4}-\d+', re.IGNORECASE)
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(SEMANTIC_MODEL)
    return _encoder

def _tok(t):
    return re.sub(r'[^\w\-]', ' ', t.lower()).split()

def _rrf(b, d, k=TOP_K):
    sc, bm, dm, ds, cm = {},{},{},{},{}
    for r in b:
        cid=r["chunk"]["chunk_id"]; sc[cid]=sc.get(cid,0)+1/(RRF_K+r["rank"])
        bm[cid]=r["rank"]; cm[cid]=r["chunk"]
    for r in d:
        cid=r["chunk"]["chunk_id"]; sc[cid]=sc.get(cid,0)+1/(RRF_K+r["rank"])
        dm[cid]=r["rank"]; ds[cid]=r.get("dense_score",0); cm[cid]=r["chunk"]
    sids=sorted(sc,key=lambda x:sc[x],reverse=True)[:k]
    return [{"chunk":cm[c],"rrf_score":sc[c],"dense_score":ds.get(c,0.0),
             "lookup_type":"hybrid","rank":i+1} for i,c in enumerate(sids)]


# ── SYSTEM A: Naive RAG ───────────────────────────────────────────
class NaiveRetriever:
    """BM25 only on full CVE texts. No dense. No confidence."""
    def __init__(self, chunks):
        self.chunks  = chunks
        self.cve_map = {}
        for c in chunks:
            self.cve_map.setdefault(c["cve_id"],[]).append(c)
        print(f"[Naive] BM25 over {len(chunks):,} CVEs ...")
        self.bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])
        print("[Naive] Ready")

    def retrieve(self, query, k=TOP_K):
        m = CVE_RE.search(query)
        if m:
            cid = m.group(0).upper()
            if cid in self.cve_map:
                return [{"chunk":c,"rrf_score":1.0,
                         "dense_score":1.0-i*0.05,
                         "lookup_type":"direct","rank":i+1}
                        for i,c in enumerate(self.cve_map[cid][:k])]
        sc   = self.bm25.get_scores(_tok(query))
        top  = np.argsort(sc)[::-1][:k]
        mx   = float(sc[top[0]]) if len(top) else 1.0
        return [{"chunk":self.chunks[i],"rrf_score":float(sc[i]),
                 "dense_score":float(sc[i])/max(mx,1e-9),
                 "lookup_type":"hybrid","rank":r+1}
                for r,i in enumerate(top)]


# ── SYSTEM B & C: Hybrid RAG (BM25 + Dense + RRF) ────────────────
class HybridRetriever:
    """BM25 + Dense + RRF on full CVE texts. Used for both B and C."""
    def __init__(self, chunks, index_file, embeds_file):
        self.chunks  = chunks
        self.cve_map = {}
        for c in chunks:
            self.cve_map.setdefault(c["cve_id"],[]).append(c)
        print(f"\n[Hybrid] CVE map: {len(self.cve_map):,}")
        print(f"[Hybrid] BM25 over {len(chunks):,} chunks ...")
        self.bm25    = BM25Okapi([_tok(c["text"]) for c in chunks])
        self.encoder = get_encoder()
        if index_file.exists() and embeds_file.exists():
            self.index = faiss.read_index(str(index_file))
            if self.index.ntotal != len(chunks):
                self._build(index_file, embeds_file)
            else:
                print(f"[Hybrid] Dense loaded {self.index.ntotal:,} ✓")
        else:
            self._build(index_file, embeds_file)
        print("[Hybrid] Ready ✓")

    def _build(self, index_file, embeds_file):
        print(f"[Hybrid] Encoding {len(self.chunks):,} ...")
        embs = self.encoder.encode(
            [c["text"] for c in self.chunks],
            batch_size=256, normalize_embeddings=True,
            show_progress_bar=True).astype("float32")
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        faiss.write_index(self.index, str(index_file))
        np.save(str(embeds_file), embs)
        print(f"[Hybrid] Built {self.index.ntotal:,}")

    def retrieve(self, query, k=TOP_K):
        m = CVE_RE.search(query)
        if m:
            cid = m.group(0).upper()
            if cid in self.cve_map:
                return [{"chunk":c,"rrf_score":1.0,
                         "dense_score":1.0-i*0.05,
                         "lookup_type":"direct","rank":i+1}
                        for i,c in enumerate(self.cve_map[cid][:k])]
        # BM25
        bsc  = self.bm25.get_scores(_tok(query))
        btop = np.argsort(bsc)[::-1][:k*2]
        b    = [{"chunk":self.chunks[i],"score":float(bsc[i]),
                 "dense_score":0.0,"rank":r+1} for r,i in enumerate(btop)]
        # Dense
        qv   = self.encoder.encode([query],normalize_embeddings=True,
               show_progress_bar=False).astype("float32")
        ds,di = self.index.search(qv, k*2)
        d    = [{"chunk":self.chunks[i],"score":float(s),
                 "dense_score":float(s),"rank":r+1}
                for r,(i,s) in enumerate(zip(di[0],ds[0])) if i>=0]
        return _rrf(b, d, k=k)
