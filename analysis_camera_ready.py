# analysis_camera_ready.py — Reproduces every number added in the camera-ready
# (icoste2026_camera.tex) from the saved 200-query run in results/.
# Run: python analysis_camera_ready.py
#
# The gate does not change the answer to an accepted query, so System C at any
# threshold equals System B's saved answers with low-confidence queries removed
# (verified: C's answers are identical to B's wherever C answers). Answers are not
# regenerated, since greedy decoding in Ollama is not reproducible across versions. Only the score ablation needs s1/s2, which are recovered by
# re-running retrieval (no LLM calls).

import numpy as np, pandas as pd
from scipy import stats
from confidence import compute_ece

A = pd.read_csv("results/results_A. Naive RAG.csv")
B = pd.read_csv("results/results_B. Hybrid RAG (no abstain).csv")
C = pd.read_csv("results/results_C. Conf-Aware RAG (Proposed).csv")
assert (A["query"].values == B["query"].values).all() and np.allclose(B.confidence, C.confidence)
self_abs = (B.rouge_l == 0).values          # the LLM's own "Insufficient context" replies
gate = (C.abstained == 1).values
rng = np.random.default_rng(42)


def auroc(conf, cor):
    conf, cor = np.asarray(conf), np.asarray(cor)
    return stats.mannwhitneyu(conf[cor == 1], conf[cor == 0]).statistic / ((cor == 1).sum() * (cor == 0).sum())


def paired(x, y, name):
    d = np.asarray(x) - np.asarray(y); n = len(d)
    t, p = stats.ttest_rel(x, y); w = stats.wilcoxon(x, y, zero_method="zsplit").pvalue
    ci = np.percentile([rng.choice(d, n).mean() for _ in range(10000)], [2.5, 97.5])
    print(f"{name}: diff={d.mean():.3f} t({n-1})={t:.2f} p={p:.4f} Wilcoxon p={w:.4f} "
          f"d_z={d.mean()/d.std(ddof=1):.2f} 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")


print("== Table II: confidence quality ==")
for n, d in [("A", A), ("B", B), ("C", C)]:
    print(f"{n}: ECE(all)={compute_ece(d.confidence, d.correct):.3f}  "
          f"conf range=[{d.confidence.min():.2f}, {d.confidence.max():.2f}] mean={d.confidence.mean():.2f}  "
          f"acc={d.correct.mean():.3f}")
print(f"AUROC A={auroc(A.confidence, A.correct):.3f}  B/C score={auroc(B.confidence, B.correct):.3f}")
print(f"ECE on answered only: B={compute_ece(B.confidence[~self_abs], B.correct[~self_abs]):.3f} "
      f"C={compute_ece(C.confidence[~gate], C.correct[~gate]):.3f}")

print("\n== Significance ==")
paired(C.rouge_l, A.rouge_l, "ROUGE-L all, C vs A")
paired(C.rouge_l[~gate], A.rouge_l[~gate], "ROUGE-L on C-answered, C vs A")
diffs = []
for _ in range(10000):
    i = rng.integers(0, 200, 200)
    diffs.append(B.correct.values[i][~gate[i]].mean() - A.correct.values[i].mean())
print("Prec C - A = %.3f, paired bootstrap 95%% CI [%.3f, %.3f]"
      % (C.correct[~gate].mean() - A.correct.mean(), *np.percentile(diffs, [2.5, 97.5])))
a, n1 = int(C.correct[~gate].sum()), int((~gate).sum())
b, n2 = int(B.correct[gate].sum()), int(gate.sum())
odds, p = stats.fisher_exact([[a, n1 - a], [b, n2 - b]])
print(f"Accepted {a}/{n1}={a/n1:.3f} vs rejected (ungated B) {b}/{n2}={b/n2:.3f}: OR={odds:.1f} p={p:.1e}")
print(f"LLM self-refusals: {self_abs.sum()}, all with C=0: {(B.confidence[self_abs] == 0).all()}")

print("\n== Table IV: threshold sweep (200 queries) ==")
print(f"smallest non-zero score: {B.confidence[B.confidence > 0].min():.3f}")
for t in [None, 0.20, 0.30, 0.40, 0.50]:
    ans = ~self_abs if t is None else (B.confidence.values >= t) & ~self_abs
    print(f"theta={t}: coverage={ans.mean():.3f} answered={ans.sum()} "
          f"prec={B.correct[ans].mean():.3f} RL(ans)={B.rouge_l[ans].mean():.3f}")

print("\n== Discussion: failure analysis ==")
print(f"rejected={gate.sum()}, would be correct={int(B.correct[gate].sum())}, "
      f"target in top 5={int(C.recall_at_k[gate].sum())}, by type={C.query_type[gate].value_counts().to_dict()}")
wrong = ~gate & (C.correct == 0).values
hi = ~gate & (C.confidence >= 0.5).values
print(f"incorrect answers={wrong.sum()}, target missing from top 5={int((wrong & (C.recall_at_k == 0).values).sum())}")
print(f"answers with C>=0.5={hi.sum()}, incorrect={int((hi & wrong).sum())}, "
      f"of which target retrieved={int((hi & wrong & (C.recall_at_k == 1).values).sum())}")
print("precision by query type:", C[~gate].groupby("query_type").correct.mean().round(3).to_dict())
print("query types:", (C.query_type.value_counts(normalize=True) * 100).round(1).to_dict())

print("\n== Table III: score ablation (re-runs retrieval, ~1 min) ==")
from chunkers import load_chunks
from retriever import HybridRetriever
from config import FAISS_FIXED_INDEX, FAISS_FIXED_EMBEDS, TOP_K
R = HybridRetriever(load_chunks("fixed"), FAISS_FIXED_INDEX, FAISS_FIXED_EMBEDS)
s = np.array([[r["dense_score"] for r in R.retrieve(q, k=TOP_K)][:2] for q in B["query"]])
s1, s2 = s[:, 0], s[:, 1]
m = np.where(s1 > 0, np.clip((s1 - s2) / np.where(s1 > 0, s1, 1), 0, 1), 0)
cor = B.correct.values
for name, alpha in [("full a=0.6", 0.6), ("similarity only a=1", 1.0), ("margin only a=0", 0.0)]:
    c = alpha * np.minimum(s1, 1) + (1 - alpha) * m
    prec = [cor[np.argsort(-c, kind="stable")[:k]].mean() for k in (168, 100)]
    print(f"{name}: AUROC={auroc(c, cor):.3f} prec@84%={prec[0]:.3f} prec@50%={prec[1]:.3f}")
print("alpha 0.5..1.0 AUROC:", [round(auroc(a * s1 + (1 - a) * m, cor), 3) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
for q in ["Apache HTTP/2 rapid reset denial of service", "recent Linux bug",
          "Incorrect X-Forwarded-For pretix Incorrect parsing"]:
    from confidence import score_margin_confidence
    print(f"worked example '{q}': C={score_margin_confidence(R.retrieve(q, k=TOP_K)):.2f}")
