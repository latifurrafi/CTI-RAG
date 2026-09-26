# threshold_sweep.py — Find optimal abstention threshold
# Run: python threshold_sweep.py
# Takes ~25 min (uses existing indexes, no rebuild)

import json, time, numpy as np
from pathlib import Path
from data_loader  import load_nvd_v2
from chunkers     import load_chunks
from retriever    import HybridRetriever
from generator    import generate_answer, check_ollama
from confidence   import score_margin_confidence, interpret_confidence, compute_ece
from evaluate     import build_eval_set_from_nvd, recall_at_k, rouge_l, correct
from config       import (RESULTS_DIR, FIGURES_DIR, INDEX_DIR,
                           FAISS_FIXED_INDEX, FAISS_FIXED_EMBEDS, TOP_K)
from tqdm import tqdm

THRESHOLDS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ABSTAIN_MSG = "Insufficient context"

def run_sweep():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    check_ollama()

    print("[sweep] Loading data and indexes ...")
    df     = load_nvd_v2()
    chunks = load_chunks("fixed")

    print("[sweep] Building Hybrid retriever ...")
    retriever = HybridRetriever(chunks, FAISS_FIXED_INDEX, FAISS_FIXED_EMBEDS)

    eval_set = build_eval_set_from_nvd(df, n=100, seed=42)  # 100 queries for speed
    print(f"[sweep] {len(eval_set)} queries, {len(THRESHOLDS)} thresholds\n")

    # Pre-compute all retrievals and confidences (do once, reuse)
    print("[sweep] Pre-computing retrievals ...")
    precomputed = []
    for item in tqdm(eval_set, desc="Retrieve"):
        res  = retriever.retrieve(item["query"], k=TOP_K)
        conf = score_margin_confidence(res)
        precomputed.append((res, conf))

    results = []
    for threshold in THRESHOLDS:
        print(f"\n[sweep] Threshold = {threshold}")
        rec_all, rou_all, cor_all   = [], [], []
        rou_ans, cor_ans            = [], []
        conf_all                    = []
        abstained                   = 0

        for i, item in enumerate(tqdm(eval_set, desc=f"θ={threshold}")):
            res, conf = precomputed[i]
            conf_all.append(conf)

            rids = [r["chunk"]["cve_id"] for r in res]
            r5   = recall_at_k(rids, item["gold_cve_id"])
            rec_all.append(r5)

            ci = interpret_confidence(conf, threshold=threshold)
            if ci["flag"]:   # abstain
                pred = ABSTAIN_MSG
                abstained += 1
                rl = 0.0
                c  = False
            else:
                resp = generate_answer(item["query"], res)
                pred = resp.get("answer","")
                rl   = rouge_l(pred, item["gold_answer"])
                c    = correct(pred, item["gold_answer"])
                rou_ans.append(rl)
                cor_ans.append(c)

            rou_all.append(rl)
            cor_all.append(c)

        n          = len(eval_set)
        abst_rate  = abstained / n
        coverage   = 1.0 - abst_rate
        ece        = compute_ece(conf_all, cor_all)
        prec_ans   = float(np.mean(cor_ans))  if cor_ans  else 0.0
        rouge_ans  = float(np.mean(rou_ans))  if rou_ans  else 0.0
        rouge_all  = float(np.mean(rou_all))

        r = {
            "threshold":        threshold,
            "abstention_rate":  round(abst_rate, 3),
            "coverage":         round(coverage,  3),
            "recall_at_k":      round(float(np.mean(rec_all)), 4),
            "rouge_l_all":      round(rouge_all, 4),
            "rouge_l_answered": round(rouge_ans, 4),
            "ece":              round(ece,        4),
            "precision_answered": round(prec_ans, 4),
        }
        results.append(r)

        print(f"  Abstention={abst_rate*100:.1f}%  Coverage={coverage*100:.1f}%  "
              f"ROUGE(ans)={rouge_ans:.4f}  ECE={ece:.4f}  Prec@Ans={prec_ans:.4f}")

    # Print comparison table
    print("\n" + "═"*85)
    print("THRESHOLD SWEEP RESULTS")
    print("─"*85)
    h = (f"{'θ':>6} {'Abstain':>8} {'Coverage':>9} {'Recall@5':>9} "
         f"{'ROUGE(ans)':>11} {'ECE':>8} {'Prec@Ans':>9}")
    print(h); print("─"*85)
    for r in results:
        print(f"  {r['threshold']:>4}   {r['abstention_rate']*100:>7.1f}%"
              f"  {r['coverage']*100:>8.1f}%"
              f"  {r['recall_at_k']:>9.4f}"
              f"  {r['rouge_l_answered']:>11.4f}"
              f"  {r['ece']:>8.4f}"
              f"  {r['precision_answered']:>9.4f}")
    print("═"*85)

    # Find sweet spot: abstention 30-50%
    sweet = [r for r in results
             if 0.25 <= r["abstention_rate"] <= 0.55]
    if sweet:
        best = max(sweet, key=lambda x: x["rouge_l_answered"])
        print(f"\n✅ Recommended threshold: θ = {best['threshold']}")
        print(f"   Abstention = {best['abstention_rate']*100:.1f}%  "
              f"Coverage = {best['coverage']*100:.1f}%")
        print(f"   ROUGE(answered) = {best['rouge_l_answered']:.4f}  "
              f"ECE = {best['ece']:.4f}")
        print(f"   Precision@Answered = {best['precision_answered']:.4f}")
    else:
        best = max(results, key=lambda x: x["rouge_l_answered"])
        print(f"\n⚠ No threshold in 25–55% range. Best overall: θ = {best['threshold']}")

    # Save
    out = RESULTS_DIR / "threshold_sweep.json"
    with open(out,"w") as f: json.dump(results, f, indent=2)
    print(f"\n[sweep] Results saved → {out}")

    # Plot
    _plot_sweep(results)
    return results, best["threshold"]

def _plot_sweep(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(exist_ok=True)
    thresholds = [r["threshold"]        for r in results]
    abst       = [r["abstention_rate"]  for r in results]
    rouge_ans  = [r["rouge_l_answered"] for r in results]
    ece        = [r["ece"]              for r in results]
    prec       = [r["precision_answered"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(thresholds, [a*100 for a in abst], "o-", color="#E05C1A", lw=2)
    axes[0].axhspan(25, 55, alpha=0.12, color="#16A669", label="Sweet spot")
    axes[0].set_xlabel("Threshold θ"); axes[0].set_ylabel("Abstention Rate (%)")
    axes[0].set_title("Abstention Rate vs θ"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(thresholds, rouge_ans, "s-", color="#0EA5C9", lw=2, label="ROUGE-L (answered)")
    axes[1].plot(thresholds, prec,      "^-", color="#7048A8", lw=2, label="Precision@Answered")
    axes[1].set_xlabel("Threshold θ"); axes[1].set_ylabel("Score")
    axes[1].set_title("Quality vs θ"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(thresholds, ece, "D-", color="#C0392B", lw=2)
    axes[2].set_xlabel("Threshold θ"); axes[2].set_ylabel("ECE (↓ better)")
    axes[2].set_title("Calibration (ECE) vs θ"); axes[2].grid(alpha=0.3)

    fig.suptitle("Threshold Sensitivity Analysis — Confidence-Aware CTI-RAG",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_threshold_sweep.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[sweep] Figure saved → {out}")

if __name__ == "__main__":
    run_sweep()
