# CTI-RAG

**Confidence-Aware Retrieval-Augmented Generation with Abstention for Cyber Threat Intelligence**

A RAG pipeline over the NVD CVE corpus that scores its own retrieval confidence and
**abstains** instead of answering when evidence is weak. Abstention trades a small amount
of coverage for higher precision on the questions it does answer, which matters in a
security context where a confidently wrong answer about a vulnerability is worse than
"I don't know."

## Results

Evaluated on 200 keyword queries generated automatically from randomly sampled CVE records
(`results/all_metrics.json`, per-query results in `results/results_*.csv`):

| Configuration | Recall@5 | ROUGE-L (answered) | Precision (answered) | Coverage | ECE | AUROC |
|---|---|---|---|---|---|---|
| A. Naive RAG | 0.810 | 0.330 | 0.700 | 100% | 0.038 | 0.690 |
| B. Hybrid RAG (no gate) | 0.795 | 0.321 | 0.737 | 93% | 0.309 | 0.700 |
| C. Conf-Aware RAG (proposed) | 0.795 | 0.334 | 0.756 | 84% | 0.259 | 0.700 |

Abstaining on the lowest-confidence 16% of queries raises precision on answered
questions to 0.756; raising the threshold further reaches 0.850 at 20% coverage. The
ungated system answers only 31% of the rejected queries correctly. Hybrid RAG's coverage
is 93% because the language model itself declined 14 queries. Naive RAG's low ECE comes
from a nearly constant score, not from useful calibration; see Section V-A of the paper.

## Setup

**1. Dependencies**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Ollama** (generation runs locally)

```bash
ollama serve
ollama pull llama3.2:3b
```

**3. CVE data** — not in this repo, see below

Download the NVD JSON 2.0 feed for 2023 from the
[NVD data feeds page](https://nvd.nist.gov/vuln/data-feeds) and place it at:

```
data/nvdcve-2.0-2023.json
```

**4. Verify**

```bash
python main.py check
```

This checks every package, the CVE file, and the Ollama connection before you commit to
a long indexing run.

## Usage

```bash
python main.py full          # complete pipeline: index, retrieve, generate, evaluate
python main.py eval_only     # evaluation only (indexes already built)
python main.py interactive   # ask your own questions
python main.py figures_only  # regenerate figures from existing results
python main.py check         # environment check
```

The first `full` run builds the FAISS indexes and is slow; later runs reuse them.

## How it works

1. **Indexing** — each CVE record is indexed whole (30,932 records), with a short
   CVE/severity/CWE tag prepended. `chunkers.py` also implements semantic and fixed-size
   chunking, which the paper's results do not use
2. **Hybrid retrieval** — BM25 and dense FAISS results fused with Reciprocal Rank Fusion
   (k=60), top-5 retained
3. **Confidence** — `C = 0.6 x abs + 0.4 x margin`, where `abs` is the top-1 dense
   cosine similarity and `margin` is `(s1 - s2) / s1` over the top two scores. A top
   record found only by BM25 has no dense score, so `C = 0`; this retriever disagreement
   is what the gate mostly detects. In the ablation, the absolute term carries almost all
   of the signal. Direct CVE-ID lookups bypass this and floor at 0.70.
4. **Abstention gate** — below a confidence of 0.20, return an abstention instead of an
   answer
5. **Generation** — `llama3.2:3b` via Ollama at temperature 0, constrained to the
   retrieved CVE context

Tunables live in `config.py`.

## Layout

```
chunkers.py  confidence.py  retriever.py  generator.py   core components
config.py                                                all parameters
data_loader.py                                           NVD JSON 2.0 parser
pipeline.py  main.py                                     orchestration + CLI
evaluate.py                                              metrics (recall, ROUGE-L, ECE)
figures.py  plot_figure.py  threshold_sweep.py           extra figures and threshold sweep
icoste2026_camera.tex  icoste2026_camera.pdf             IEEE i-COSTE 2026 camera-ready paper
analysis_camera_ready.py                                 reproduces every number in the paper
plot_risk_coverage.py                                    Fig. 2
figures/fig1_pipeline.drawio  figures/make_fig1_drawio.py  Fig. 1 (draw.io source)
results/                                                 per-query results behind the paper
```

## What is not in this repo

Excluded via `.gitignore`, because these exceed GitHub's 100 MB per-file limit and are
all reproducible:

| Path | Size | How to get it back |
|---|---|---|
| `data/`, `Dataset/` | 184 MB | Download from NVD (above) |
| `index/` | 392 MB | Rebuilt by `python main.py full` |
| `venv/` | 1.0 GB | `pip install -r requirements.txt` |
| `Literature Review/` | 9.3 MB | Third-party published papers, not redistributed |

A fresh clone therefore needs the NVD download and one `main.py full` run before it can
answer queries.

## Paper

Accepted at IEEE i-COSTE 2026 (Paper ID ieee-icoste_3738).

```bash
pdflatex icoste2026_camera && pdflatex icoste2026_camera
python analysis_camera_ready.py   # every number in the paper, from results/*.csv
python plot_risk_coverage.py      # Fig. 2
```

Builds the 6-page IEEE camera-ready version with the full author list.
