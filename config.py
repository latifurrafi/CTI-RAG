from pathlib import Path
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
INDEX_DIR   = BASE_DIR / "index"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
NVD_JSON_PATH     = DATA_DIR / "nvdcve-2.0-2023.json"
MIN_DESC_LENGTH   = 50
FIXED_CHUNK_SIZE  = 256
FIXED_CHUNK_OVERLAP = 30
SEMANTIC_MODEL    = "all-MiniLM-L6-v2"
COSINE_THRESHOLD  = 0.65
MIN_CHUNK_WORDS   = 8
TOP_K             = 5
RRF_K             = 60
CONFIDENCE_THRESHOLD = 0.20
OLLAMA_MODEL      = "llama3.2:3b"
OLLAMA_BASE_URL   = "http://localhost:11434"
MAX_TOKENS        = 512
TEMPERATURE       = 0.0

PROMPT_TEMPLATE = """You are a cybersecurity analyst. Use ONLY the CVE context below.
Do NOT invent details. If context is insufficient, say so.

{context}

Question: {question}

Answer (include: what the vulnerability is, affected component, severity, impact):"""

EVAL_SAMPLE_SIZE    = 200
ROUGE_TYPE          = "rougeL"
FIXED_INDEX_FILE    = INDEX_DIR / "fixed_chunks.pkl"
SEMANTIC_INDEX_FILE = INDEX_DIR / "semantic_chunks.pkl"
FAISS_FIXED_INDEX   = INDEX_DIR / "faiss_fixed.index"
FAISS_FIXED_EMBEDS  = INDEX_DIR / "embeddings_fixed.npy"
FAISS_SEM_INDEX     = INDEX_DIR / "faiss_semantic.index"
FAISS_SEM_EMBEDS    = INDEX_DIR / "embeddings_semantic.npy"
