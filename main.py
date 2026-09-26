#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────
# main.py  —  Entry Point
# Usage:
#   python main.py full          # complete pipeline
#   python main.py interactive   # query mode (after indexing)
#   python main.py eval_only     # eval only (indexes already built)
#   python main.py figures_only  # regenerate figures
#   python main.py check         # check environment
# ─────────────────────────────────────────────────────────────────

import sys


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Confidence-Aware RAG with Dynamic Chunking for CTI         ║
║   ML4CS 2026  ·  Springer LNCS                               ║
║   Md. Latifur Rahman Rafi  ·  DIU                            ║
╚══════════════════════════════════════════════════════════════╝
"""

USAGE = """
Commands:
  python main.py full          Run complete pipeline (Day 4-10)
  python main.py interactive   Interactive query mode
  python main.py eval_only     Evaluation only (indexes exist)
  python main.py figures_only  Regenerate figures from results
  python main.py check         Check all dependencies + data
"""


def check_environment():
    """Check all dependencies and data files exist."""
    print("Checking environment...\n")
    ok = True

    # Python packages
    packages = [
        ("rank_bm25",            "rank-bm25"),
        ("sentence_transformers","sentence-transformers"),
        ("faiss",                "faiss-cpu"),
        ("rouge_score",          "rouge-score"),
        ("numpy",                "numpy"),
        ("pandas",               "pandas"),
        ("tqdm",                 "tqdm"),
        ("matplotlib",           "matplotlib"),
        ("requests",             "requests"),
    ]
    for module, pip_name in packages:
        try:
            __import__(module)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ✗ {pip_name}  →  pip install {pip_name}")
            ok = False

    # Data file
    from config import NVD_JSON_PATH
    print()
    if NVD_JSON_PATH.exists():
        size_mb = NVD_JSON_PATH.stat().st_size / 1e6
        print(f"  ✓ NVD JSON found: {NVD_JSON_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ NVD JSON not found: {NVD_JSON_PATH}")
        print(f"    → Place nvdcve-2.0-2023.json in the data/ folder")
        ok = False

    # Ollama
    from generator import check_ollama
    print()
    ollama_ok = check_ollama()
    if not ollama_ok:
        ok = False

    print()
    if ok:
        print("✓ All checks passed. Ready to run.\n")
        print("  Next: python main.py full")
    else:
        print("✗ Some checks failed. Fix above issues first.\n")

    return ok


if __name__ == "__main__":
    print(BANNER)

    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "check":
        check_environment()

    elif mode in ("full", "eval_only", "interactive", "figures_only"):
        from pipeline import run_full_pipeline
        run_full_pipeline(mode=mode)

    else:
        print(USAGE)
