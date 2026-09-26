# evaluate.py v10 — Confidence-aware abstention evaluation
#
# THREE systems compared:
#   A. Naive RAG        — always answers, no confidence score
#   B. Hybrid RAG       — always answers, with confidence score
#   C. Conf-Aware RAG   — abstains if confidence < threshold, answers otherwise
#
# Metrics that tell the real story:
#   Recall@5            — retrieval quality
#   ROUGE-L (all)       — generation quality on all queries
#   ROUGE-L (answered)  — generation quality only on non-abstained queries
#   Abstention rate     — fraction of queries the system refused to answer
#   ECE                 — confidence calibration quality
#   Precision@Answered  — fraction of answered queries that are correct

import json, time, re, numpy as np, pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from config import TOP_K, RESULTS_DIR, EVAL_SAMPLE_SIZE, ROUGE_TYPE, CONFIDENCE_THRESHOLD
from confidence import (score_margin_confidence, interpret_confidence,
                         compute_ece, calibration_curve_data)

ABSTAIN_PHRASE = "Insufficient context"

def recall_at_k(ids, gold, k=TOP_K):
    return float(gold.upper() in [i.upper() for i in ids[:k]])

def rouge_l(pred, ref):
    s = rouge_scorer.RougeScorer([ROUGE_TYPE], use_stemmer=True)
    return float(s.score(ref, pred)[ROUGE_TYPE].fmeasure)

def correct(pred, ref, t=0.20):
    if ABSTAIN_PHRASE in pred:
        return False     # abstained = not correct
    return rouge_l(pred, ref) >= t

STOP = {
    "a","an","the","is","in","of","to","and","or","that","this",
    "with","for","on","at","by","was","are","be","has","have",
    "it","its","as","from","via","could","allow","which","may",
    "can","issue","discovered","found","used","use","due","not",
    "user","users","remote","local","attacker","attackers",
    "affect","affected","allows","leads","before","after",
    "version","product","software","system","application","when",
}
VULN_TYPES = {
    "overflow","injection","traversal","bypass","disclosure",
    "corruption","escalation","dereference","execution","forgery",
    "scripting","hijacking","spoofing","leak","underflow","confusion",
    "race","condition","memory","null","pointer","dereference",
}

def build_eval_set_from_nvd(df, n=EVAL_SAMPLE_SIZE, seed=42):
    np.random.seed(seed)
    sample = df.sample(min(n, len(df))).reset_index(drop=True)
    out    = []
    for _, row in sample.iterrows():
        cve_id = row["id"]
        desc   = row["description"]
        sev    = row["severity"]
        gold   = desc.strip()

        words = [w.strip(".,();:\"'[]") for w in desc.split()
                 if (w.lower().strip(".,();:\"'[]") not in STOP
                     and len(w.strip(".,();:\"'[]")) > 3
                     and not w.lower().startswith("cve-"))]
        tech  = [w for w in words if (any(c.isdigit() for c in w)
                                       or any(c.isupper() for c in w[1:])
                                       or w.lower() in VULN_TYPES)]
        plain = [w for w in words if w not in tech]

        qt = np.random.choice(["technical","vuln_type","product"],
                               p=[0.50,0.30,0.20])
        if qt == "technical":
            pool  = (tech+plain)[:7]
            query = " ".join(pool[:5]) if len(pool)>=3 else " ".join(words[:5])
        elif qt == "vuln_type":
            vt    = [w for w in words if w.lower() in VULN_TYPES]
            pool  = vt[:2]+(tech+plain)[:4]
            query = f"{sev} "+" ".join(pool[:5]) if pool else " ".join(words[:5])
        else:
            prods = [w for w in words if len(w)>3 and w[0].isupper() and not w.isupper()][:3]
            pool  = prods+plain[:3]
            query = " ".join(pool[:5]) if pool else " ".join(words[:5])

        if not query.strip() or len(query.split()) < 2:
            query = " ".join(words[:5]) if words else f"{sev} vulnerability"

        out.append({"query":query.strip(),"gold_answer":gold,
                    "gold_cve_id":cve_id,"query_type":qt})
    return out


def run_evaluation(retriever, generator_fn, eval_set,
                   label="system", save_csv=True,
                   use_abstention=False):
    """
    use_abstention=True  → System C: skip generation if conf < threshold
    use_abstention=False → Systems A/B: always generate
    """
    print(f"\n[eval] '{label}' | {len(eval_set)} queries | "
          f"abstention={'ON' if use_abstention else 'OFF'}")

    rows = []
    rec,rou,con,cor,lat = [],[],[],[],[]
    abstained   = 0
    answered_rl = []   # ROUGE-L only on non-abstained queries
    answered_cor= []   # correctness only on non-abstained

    for item in tqdm(eval_set, desc=f"Eval [{label}]"):
        q, gold, gcve = item["query"], item["gold_answer"], item["gold_cve_id"]

        t0   = time.time()
        res  = retriever.retrieve(q, k=TOP_K)
        conf = score_margin_confidence(res)
        ci   = interpret_confidence(conf)

        # Abstention decision
        if use_abstention and ci["flag"]:
            pred = ABSTAIN_PHRASE
            abstained += 1
        else:
            resp = generator_fn(q, res)
            pred = resp.get("answer","")

        t1 = time.time()
        lat.append(t1-t0)
        rids = [r["chunk"]["cve_id"] for r in res]
        r5   = recall_at_k(rids, gcve)
        rl   = rouge_l(pred, gold) if ABSTAIN_PHRASE not in pred else 0.0
        c    = correct(pred, gold)

        rec.append(r5); rou.append(rl)
        con.append(conf); cor.append(c)

        if ABSTAIN_PHRASE not in pred:
            answered_rl.append(rl)
            answered_cor.append(c)

        rows.append({
            "query":q,"query_type":item.get("query_type",""),
            "gold_cve_id":gcve,"retrieved_ids":"|".join(rids),
            "recall_at_k":r5,"rouge_l":round(rl,4),
            "confidence":round(conf,4),"conf_label":ci["label"],
            "correct":int(c),"abstained":int(ABSTAIN_PHRASE in pred),
            "answer":pred,"gold_answer":gold,
            "latency_s":round(t1-t0,2),
        })

    ece = compute_ece(con, cor)
    n_answered = len(answered_rl)
    m = {
        "label":             label,
        "n_queries":         len(eval_set),
        "recall_at_k":       round(float(np.mean(rec)),   4),
        "rouge_l_mean":      round(float(np.mean(rou)),   4),   # all queries
        "rouge_l_answered":  round(float(np.mean(answered_rl)) if answered_rl else 0, 4),
        "ece":               round(ece,                    4),
        "avg_confidence":    round(float(np.mean(con)),   4),
        "abstention_rate":   round(abstained/len(eval_set),4),
        "precision_answered":round(float(np.mean(answered_cor)) if answered_cor else 0, 4),
        "n_answered":        n_answered,
        "low_conf_rate":     round(float(np.mean([c<CONFIDENCE_THRESHOLD for c in con])),4),
        "latency_mean_s":    round(float(np.mean(lat)),   2),
    }

    print(f"\n[eval] Results — '{label}':")
    for k,v in m.items():
        if k not in ("label","n_queries","n_answered"):
            print(f"  {k:<24}: {v}")

    if save_csv:
        RESULTS_DIR.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR/f"results_{label}.csv",index=False)
        print(f"[eval] Saved → results_{label}.csv")
    return m


def print_comparison_table(all_metrics):
    # Print the full table with all key metrics
    h = (f"{'System':<30} {'Recall@5':>9} {'ROUGE-L':>9} "
         f"{'ROUGE(ans)':>10} {'ECE':>8} {'Abstain':>8} {'Prec@Ans':>9}")
    sep = "─"*len(h)
    print(f"\n{'═'*len(h)}")
    print("RESULTS TABLE  (keyword-only queries)")
    print(sep); print(h); print(sep)
    for m in all_metrics:
        print(f"{m['label']:<30} "
              f"{m['recall_at_k']:>9.4f} "
              f"{m['rouge_l_mean']:>9.4f} "
              f"{m['rouge_l_answered']:>10.4f} "
              f"{m['ece']:>8.4f} "
              f"{m['abstention_rate']:>8.4f} "
              f"{m['precision_answered']:>9.4f}")
    print("═"*len(h))
    print("\nKey: ROUGE(ans)=ROUGE on answered queries only | "
          "Prec@Ans=fraction of answered queries correct")


def save_metrics_json(all_metrics, filename="all_metrics.json"):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR/filename,"w") as f:
        json.dump(all_metrics,f,indent=2)
    print(f"[eval] Saved → {filename}")
