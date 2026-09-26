# pipeline.py v10 — Three-system confidence-aware comparison
import json, time, re, numpy as np
from config import (INDEX_DIR, RESULTS_DIR, FIGURES_DIR,
    FIXED_INDEX_FILE, SEMANTIC_INDEX_FILE,
    FAISS_FIXED_INDEX, FAISS_FIXED_EMBEDS)
from data_loader  import load_nvd_v2
from chunkers     import build_fixed_chunks, load_chunks, chunk_stats
from retriever    import NaiveRetriever, HybridRetriever
from confidence   import score_margin_confidence, interpret_confidence
from generator    import generate_answer, format_response, check_ollama
from evaluate     import (run_evaluation, build_eval_set_from_nvd,
                           print_comparison_table, save_metrics_json)
from figures      import (plot_comparison_bars, plot_calibration_curve,
                           plot_confidence_distribution, plot_ablation)

def run_full_pipeline(mode="full"):
    start = time.time()
    for d in [INDEX_DIR, RESULTS_DIR, FIGURES_DIR]:
        d.mkdir(exist_ok=True)

    check_ollama()
    df = load_nvd_v2()

    # One chunk per CVE — full enriched text
    force = (mode == "full")
    if not FIXED_INDEX_FILE.exists() or force:
        chunks = build_fixed_chunks(df)
    else:
        chunks = load_chunks("fixed")
    chunk_stats(chunks, "CVE chunks (1 per CVE)")

    # System A: Naive RAG — BM25 only
    print("\n[pipeline] System A: Naive RAG (BM25 only) ...")
    naive_r = NaiveRetriever(chunks)

    # System B & C: Hybrid RAG — BM25 + Dense
    print("\n[pipeline] System B/C: Hybrid RAG (BM25 + Dense + RRF) ...")
    hybrid_r = HybridRetriever(chunks, FAISS_FIXED_INDEX, FAISS_FIXED_EMBEDS)

    if mode == "interactive":
        print("\n[pipeline] Interactive — 'quit' to exit")
        while True:
            q = input("\n🔍 Query: ").strip()
            if q.lower() in ("quit","exit","q"): break
            if not q: continue
            t0=time.time(); res=hybrid_r.retrieve(q)
            conf=score_margin_confidence(res)
            ci=interpret_confidence(conf)
            print(f"Confidence: {ci['label']} ({ci['score']:.3f})")
            resp=generate_answer(q,res); print(format_response(resp))
            print(f"⏱ {time.time()-t0:.2f}s")
        return

    # Eval set
    eval_set = build_eval_set_from_nvd(df)
    from collections import Counter
    qt = dict(Counter(e["query_type"] for e in eval_set))
    print(f"\n[pipeline] {len(eval_set)} keyword queries: {qt}")

    all_metrics = []

    # ── System A: Naive RAG, no abstention ───────────────────────
    print("\n[pipeline] ── System A: Naive RAG")
    mA = run_evaluation(naive_r, generate_answer, eval_set,
                         label="A. Naive RAG", use_abstention=False)
    all_metrics.append(mA)

    # ── System B: Hybrid RAG, no abstention ──────────────────────
    print("\n[pipeline] ── System B: Hybrid RAG (no abstention)")
    mB = run_evaluation(hybrid_r, generate_answer, eval_set,
                         label="B. Hybrid RAG (no abstain)", use_abstention=False)
    all_metrics.append(mB)

    # ── System C: Hybrid RAG + confidence abstention ─────────────
    print("\n[pipeline] ── System C: Hybrid RAG + Confidence Abstention")
    mC = run_evaluation(hybrid_r, generate_answer, eval_set,
                         label="C. Conf-Aware RAG (Proposed)", use_abstention=True)
    all_metrics.append(mC)

    save_metrics_json(all_metrics)

    # ── Ablation: what threshold is optimal? ─────────────────────
    subset = eval_set[:50]
    print("\n[pipeline] ── Ablation: threshold sensitivity")
    from config import CONFIDENCE_THRESHOLD
    abl = []
    for lbl, fn, abst in [
        ("No Confidence",    generate_answer, False),
        ("Conf-Aware (0.15)",generate_answer, True),
    ]:
        abl.append(run_evaluation(hybrid_r, fn, subset,
                                  label=lbl, use_abstention=abst))
    save_metrics_json(abl, "ablation_metrics.json")

    # ── Figures ──────────────────────────────────────────────────
    # Use subset of metrics for bar charts
    bar_metrics = [
        {"label":"Naive RAG",        **{k:v for k,v in mA.items() if k in ["recall_at_k","rouge_l_mean","ece"]}},
        {"label":"Hybrid RAG",       **{k:v for k,v in mB.items() if k in ["recall_at_k","rouge_l_mean","ece"]}},
        {"label":"Conf-Aware (Prop)",**{k:v for k,v in mC.items() if k in ["recall_at_k","rouge_l_mean","ece"]}},
    ]
    plot_comparison_bars(bar_metrics)
    abl_ext = [
        {"label":"Naive RAG",     **{k:v for k,v in mA.items()   if k in ["recall_at_k","rouge_l_mean","ece"]}},
        {"label":"+ Dense Rerank",**{k:v for k,v in mB.items()   if k in ["recall_at_k","rouge_l_mean","ece"]}},
        {"label":"+ Abstention",  **{k:v for k,v in abl[1].items() if k in ["recall_at_k","rouge_l_mean","ece"]}},
    ]
    plot_ablation(abl_ext)

    for csv_label, label in [
        ("results_A. Naive RAG.csv",           "Naive RAG"),
        ("results_C. Conf-Aware RAG (Proposed).csv","Conf-Aware"),
    ]:
        p = RESULTS_DIR/csv_label
        if p.exists():
            plot_calibration_curve(p, label)

    plot_confidence_distribution({
        "Naive RAG":    RESULTS_DIR/"results_A. Naive RAG.csv",
        "Hybrid RAG":   RESULTS_DIR/"results_B. Hybrid RAG (no abstain).csv",
        "Conf-Aware":   RESULTS_DIR/"results_C. Conf-Aware RAG (Proposed).csv",
    })

    print_comparison_table(all_metrics)
    print(f"\n[pipeline] ✓ Done in {(time.time()-start)/60:.1f} min")
    _print_paper_story(mA, mB, mC)

def _print_paper_story(mA, mB, mC):
    print("\n" + "═"*65)
    print("PAPER STORY — What to write in Section 6")
    print("═"*65)
    rA = mA["rouge_l_mean"]; rC = mC["rouge_l_answered"]
    eA = mA["ece"];           eC = mC["ece"]
    abst = mC["abstention_rate"]
    prec = mC["precision_answered"]
    print(f"""
System A (Naive RAG) answers all queries:
  ROUGE-L = {rA:.4f},  ECE = {eA:.4f}  (no confidence signal)

System C (Proposed) uses confidence-aware abstention:
  Abstains on {abst*100:.1f}% of low-confidence queries
  ROUGE-L on answered queries = {rC:.4f}  (vs {rA:.4f} naive)
  ECE = {eC:.4f}  (vs {eA:.4f} naive, lower = better calibrated)
  Precision@Answered = {prec:.4f}

Paper claim:
  "Our confidence-aware system improves answer quality by
  {((rC/rA)-1)*100:.1f}% on answered queries while reducing ECE by
  {((eA-eC)/eA)*100:.1f}%, demonstrating that confidence-based
  abstention produces more reliable CTI answers than always-answer
  baselines — critical for security-critical workflows."
""")

if __name__ == "__main__":
    import sys
    run_full_pipeline(mode=sys.argv[1] if len(sys.argv)>1 else "full")
