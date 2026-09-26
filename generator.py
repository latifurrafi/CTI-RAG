# generator.py v6 — uses parent_text (full CVE) for generation
import requests
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MAX_TOKENS, TEMPERATURE, PROMPT_TEMPLATE
from confidence import score_margin_confidence, interpret_confidence

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models",[])]
        ok = any(OLLAMA_MODEL in m for m in models)
        print(f"[generator] Ollama {'ready' if ok else '⚠ model not found'}. Model: {OLLAMA_MODEL}")
        return ok
    except Exception as e:
        print(f"[generator] ⚠ Ollama not reachable: {e}"); return False

def ollama_generate(prompt, model=OLLAMA_MODEL):
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
            json={"model":model,"prompt":prompt,"stream":False,
                  "options":{"temperature":TEMPERATURE,"num_predict":MAX_TOKENS,
                             "stop":["\n\n\n"]}}, timeout=120)
        r.raise_for_status()
        return r.json().get("response","").strip()
    except Exception as e:
        return f"[ERROR] {e}"

def build_context(results, max_chunks=5):
    seen, lines = set(), []
    for r in results[:max_chunks]:
        c   = r["chunk"]
        cid = c.get("cve_id","?")
        if cid in seen: continue
        seen.add(cid)
        # use parent_text (full CVE) not child chunk
        text = c.get("parent_text") or c.get("text","")
        lines.append(f"[{len(lines)+1}] {cid} (severity:{c.get('severity','?')} "
                     f"CVSS:{c.get('cvss_score','?')})\n{text}")
    return "\n\n".join(lines)

def generate_answer(query, results, model=OLLAMA_MODEL):
    ctx   = build_context(results)
    conf  = score_margin_confidence(results)
    cinfo = interpret_confidence(conf)
    prompt = PROMPT_TEMPLATE.format(context=ctx, question=query)
    if cinfo["flag"]:
        prompt += "\nNote: Low confidence retrieval. Say 'Insufficient context' if unsure."
    ans = ollama_generate(prompt, model)
    src = list(dict.fromkeys(r["chunk"].get("cve_id","") for r in results))
    return {"query":query,"answer":ans,"confidence":cinfo,"sources":src,"context_used":ctx}

def format_response(r):
    c = r["confidence"]
    return "\n".join(["="*70, f"QUERY: {r['query']}",
        f"CONF:  {c['label']} ({c['score']:.4f}){' ⚠' if c['flag'] else ''}",
        f"SRCS:  {', '.join(r['sources'][:3])}", "-"*70,
        f"ANSWER:\n{r['answer']}", "="*70])
