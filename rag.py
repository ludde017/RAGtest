# RAG Demo with Claude Sonnet 4 (Python + Streamlit)

This is a self‑contained Retrieval‑Augmented Generation (RAG) demo that runs locally. It:

* Ingests text/markdown files from a `docs/` folder
* Chunks & embeds them with a lightweight local model
* Indexes vectors with FAISS
* Retrieves top‑k chunks for a query
* Prompts **Claude Sonnet 4** via the Anthropic API to generate a grounded answer with citations

> **Tip:** It’s ~100 lines and has no external services besides Anthropic. Swap embeddings or the vector DB easily.

---

## 1) Setup

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put a few .txt or .md files into ./docs (create the folder if missing)
export AWS_REGION="us-east-1"                         # choose a Bedrock region with Anthropic access
# Configure AWS credentials via env vars or `aws configure` (profile/role also works)
streamlit run rag_app.py
```

Create a `docs/` folder with any `.txt`/`.md` content. The app will index on first run and cache artifacts in `.rag_cache/`.

---

## 2) `requirements.txt`

````txt
boto3>=1.35.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
numpy>=1.26
pandas>=2.2
streamlit>=1.37
python-dotenv>=1.0
```txt
anthropic>=0.35.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
numpy>=1.26
pandas>=2.2
streamlit>=1.37
python-dotenv>=1.0
````

> You can replace `faiss-cpu` with `faiss-gpu` if you have CUDA.

---

## 3) `rag_app.py`

````python
import os
import glob
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
import faiss
import boto3

CACHE_DIR = ".rag_cache"
DOCS_DIR = "docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # swap to a stronger model anytime
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
META_PATH = os.path.join(CACHE_DIR, "meta.json")

# ---- Bedrock config ----
# Configure via env or AWS profile/role. Common envs:
# AWS_REGION (e.g., "us-east-1"), AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-opus-4-20250514-v1:0"  # Opus 4 (see README notes for 4.1)
)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def read_textlike(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_docs() -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "**", "*.txt"), recursive=True) +
                   glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True))
    docs = []
    for p in paths:
        try:
            docs.append((p, read_textlike(p)))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return docs


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunks.append(text[i:end])
        i = end - overlap
        if i <= 0:
            i = end
    return [c.strip() for c in chunks if c.strip()]


def build_or_load_index() -> Tuple[faiss.IndexFlatIP, List[Chunk], SentenceTransformer]:
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # If cached index exists and docs hash matches, load it
    docs = load_docs()
    doc_fingerprints = [f"{p}:{len(t)}" for p, t in docs]
    combined = "|".join(doc_fingerprints)
    docs_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        meta = json.load(open(META_PATH, "r"))
        if meta.get("docs_hash") == docs_hash and meta.get("embed_model") == EMBED_MODEL_NAME:
            index = faiss.read_index(INDEX_PATH)
            chunks = [Chunk(**c) for c in meta["chunks"]]
            return index, chunks, embedder

    # (Re)build index
    all_chunks: List[Chunk] = []
    for p, t in docs:
        pieces = chunk_text(t)
        for j, piece in enumerate(pieces):
            all_chunks.append(Chunk(text=piece, source=p, chunk_id=j))

    if not all_chunks:
        # Build a tiny dummy index
        index = faiss.IndexFlatIP(384)  # dimension for MiniLM-L6
        return index, [], embedder

    texts = [c.text for c in all_chunks]
    emb = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    vecs = np.array(emb).astype("float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, INDEX_PATH)
    json.dump({
        "docs_hash": docs_hash,
        "embed_model": EMBED_MODEL_NAME,
        "chunks": [c.__dict__ for c in all_chunks]
    }, open(META_PATH, "w"))

    return index, all_chunks, embedder


def retrieve(query: str, index: faiss.IndexFlatIP, embedder: SentenceTransformer, chunks: List[Chunk], k: int = 6):
    if len(chunks) == 0:
        return []
    q = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q).astype("float32"), k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        c = chunks[int(idx)]
        hits.append({
            "score": float(score),
            "text": c.text,
            "source": c.source,
            "chunk_id": c.chunk_id
        })
    return hits


def make_prompt(query: str, contexts: List[dict]) -> tuple[str, list]:
    context_block = "

".join([f"[Source {i+1}: {os.path.basename(c['source'])}#chunk-{c['chunk_id']}]
{c['text']}" for i, c in enumerate(contexts)])
    system_prompt = "You are Claude Opus 4, a careful, grounded assistant. Always cite sources using [filename#chunk-id]."
    user = f"""
You are a meticulous research assistant. Answer the user's question *using only* the context provided. If the answer is not in the context, say you don't know.

Return:
1) A concise answer (2-5 sentences)
2) Bullet citations with [filename#chunk-id]
3) If relevant, a short 'Limitations' note

<question>
{query}
</question>

<context>
{context_block}
</context>
""".strip()

    # Bedrock Converse API expects separate system and messages arrays
    messages = [
        {"role": "user", "content": [{"text": user}]}
    ]
    return system_prompt, messages


def ask_bedrock(system_prompt: str, messages: list, model_id: str = DEFAULT_MODEL_ID, max_tokens: int = 1200) -> str:
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    resp = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=messages,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0.2,
            "topP": 0.95,
        }
    )

    # The response text is in output.message.content (list of blocks)
    out = resp.get("output", {}).get("message", {}).get("content", [])
    texts = []
    for block in out:
        if block.get("text"):
            texts.append(block["text"])
    return "
".join(texts)


# ---------- UI ----------
st.set_page_config(page_title="RAG • Bedrock Opus 4", layout="wide")
st.title("RAG Demo · Amazon Bedrock (Claude Opus 4)")

with st.sidebar:
    st.header("Configuration")
    k = st.slider("Top‑k chunks", 1, 12, 6)
    model_id = st.text_input(
        "Bedrock model ID",
        value=DEFAULT_MODEL_ID,
        help="Use Opus 4 or Opus 4.1 model IDs"
    )
    region = st.text_input("AWS Region", value=BEDROCK_REGION)
    reindex = st.button("Rebuild index")

    st.markdown("""
**How to use**
1. Ensure AWS creds & region are configured
2. Add .txt/.md files to `docs/`
3. (Optional) click **Rebuild index**
4. Ask questions grounded in your files
""")

if reindex:
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(META_PATH):
        os.remove(META_PATH)
    st.experimental_rerun()

# allow region override per-session
if region and region != BEDROCK_REGION:
    # recreate client inside ask_bedrock via env var is simpler; here we just update global
    os.environ["AWS_REGION"] = region

index, chunks, embedder = build_or_load_index()
st.success(f"Indexed {len(chunks)} chunks from {DOCS_DIR}/")

query = st.text_input("Ask a question about your docs")

if query:
    with st.spinner("Retrieving..."):
        ctx = retrieve(query, index, embedder, chunks, k=k)
        df = pd.DataFrame(ctx)
        st.subheader("Context matches")
        if not df.empty:
            st.dataframe(df[["score", "source", "chunk_id", "text"]], use_container_width=True, height=220)
        else:
            st.info("No matches found. Try adding files to ./docs or rebuilding the index.")

    with st.spinner("Calling Bedrock (Opus 4)..."):
        sys_prompt, msgs = make_prompt(query, ctx)
        try:
            answer = ask_bedrock(sys_prompt, msgs, model_id=model_id)
            st.subheader("Answer")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Bedrock error: {e}")
```python
import os
import glob
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic

CACHE_DIR = ".rag_cache"
DOCS_DIR = "docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # swap to a stronger model anytime
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
META_PATH = os.path.join(CACHE_DIR, "meta.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def read_textlike(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_docs() -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "**", "*.txt"), recursive=True) +
                   glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True))
    docs = []
    for p in paths:
        try:
            docs.append((p, read_textlike(p)))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return docs


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunks.append(text[i:end])
        i = end - overlap
        if i <= 0:
            i = end
    return [c.strip() for c in chunks if c.strip()]


def build_or_load_index() -> Tuple[faiss.IndexFlatIP, List[Chunk], SentenceTransformer]:
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # If cached index exists and docs hash matches, load it
    docs = load_docs()
    doc_fingerprints = [f"{p}:{len(t)}" for p, t in docs]
    combined = "|".join(doc_fingerprints)
    docs_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        meta = json.load(open(META_PATH, "r"))
        if meta.get("docs_hash") == docs_hash and meta.get("embed_model") == EMBED_MODEL_NAME:
            index = faiss.read_index(INDEX_PATH)
            chunks = [Chunk(**c) for c in meta["chunks"]]
            return index, chunks, embedder

    # (Re)build index
    all_chunks: List[Chunk] = []
    for p, t in docs:
        pieces = chunk_text(t)
        for j, piece in enumerate(pieces):
            all_chunks.append(Chunk(text=piece, source=p, chunk_id=j))

    if not all_chunks:
        # Build a tiny dummy index
        index = faiss.IndexFlatIP(384)  # dimension for MiniLM-L6
        return index, [], embedder

    texts = [c.text for c in all_chunks]
    emb = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    vecs = np.array(emb).astype("float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, INDEX_PATH)
    json.dump({
        "docs_hash": docs_hash,
        "embed_model": EMBED_MODEL_NAME,
        "chunks": [c.__dict__ for c in all_chunks]
    }, open(META_PATH, "w"))

    return index, all_chunks, embedder


def retrieve(query: str, index: faiss.IndexFlatIP, embedder: SentenceTransformer, chunks: List[Chunk], k: int = 6):
    if len(chunks) == 0:
        return []
    q = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q).astype("float32"), k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        c = chunks[int(idx)]
        hits.append({
            "score": float(score),
            "text": c.text,
            "source": c.source,
            "chunk_id": c.chunk_id
        })
    return hits


def make_prompt(query: str, contexts: List[dict]) -> List[dict]:
    context_block = "\n\n".join([f"[Source {i+1}: {os.path.basename(c['source'])}#chunk-{c['chunk_id']}]\n{c['text']}" for i, c in enumerate(contexts)])
    user = f"""
You are a meticulous research assistant. Answer the user's question *using only* the context provided. If the answer is not in the context, say you don't know.

Return:
1) A concise answer (2-5 sentences)
2) Bullet citations with [filename#chunk-id]
3) If relevant, a short 'Limitations' note

<question>
{query}
</question>

<context>
{context_block}
</context>
""".strip()

    return [
        {"role": "system", "content": "You are Claude Sonnet 4, a careful, grounded assistant. Always cite sources using [filename#chunk-id]."},
        {"role": "user", "content": user}
    ]


def ask_claude(messages: List[dict], model: str = "claude-sonnet-4-20250514", max_output_tokens: int = 1200) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable.")
    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        messages=messages
    )
    # The SDK returns a list of content blocks; join text blocks
    parts = []
    for b in resp.content:
        if b.type == "text":
            parts.append(b.text)
    return "\n".join(parts)


# ---------- UI ----------
st.set_page_config(page_title="RAG • Claude Sonnet 4", layout="wide")
st.title("RAG Demo · Claude Sonnet 4")

with st.sidebar:
    st.header("Configuration")
    k = st.slider("Top‑k chunks", 1, 12, 6)
    model_id = st.text_input("Claude model", value="claude-sonnet-4-20250514", help="Use a snapshot or alias supported by your provider")
    reindex = st.button("Rebuild index")

    st.markdown("""
**How to use**
1. Add .txt/.md files to `docs/`
2. (Optional) click **Rebuild index**
3. Ask questions grounded in your files
""")

if reindex:
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(META_PATH):
        os.remove(META_PATH)
    st.experimental_rerun()

index, chunks, embedder = build_or_load_index()
st.success(f"Indexed {len(chunks)} chunks from {DOCS_DIR}/")

query = st.text_input("Ask a question about your docs")

if query:
    with st.spinner("Retrieving..."):
        ctx = retrieve(query, index, embedder, chunks, k=k)
        df = pd.DataFrame(ctx)
        st.subheader("Context matches")
        if not df.empty:
            st.dataframe(df[["score", "source", "chunk_id", "text"]], use_container_width=True, height=220)
        else:
            st.info("No matches found. Try adding files to ./docs or rebuilding the index.")

    with st.spinner("Asking Claude..."):
        msgs = make_prompt(query, ctx)
        try:
            answer = ask_claude(msgs, model=model_id)
            st.subheader("Answer")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Anthropic API error: {e}")
````

---

## 4) Notes & Tweaks

* **Model IDs (Bedrock):**

  * **Opus 4 (default):** `anthropic.claude-opus-4-20250514-v1:0`
  * **Opus 4.1 (newer):** `anthropic.claude-opus-4-1-20250805-v1:0` (drop‑in upgrade)
  * You must **enable model access** for the chosen region in the Bedrock console.
* **API:** Uses the **Bedrock Converse API** via `boto3` for a consistent messages interface.
* **Embeddings:** Local Sentence‑Transformers keeps the demo private & cheap. For higher quality, try `BAAI/bge-small-en-v1.5`, `BAAI/bge-m3`, or a hosted provider.
* **File types:** Loader reads `.txt`/`.md`. Add PDF/HTML loaders as needed (`pypdf`, `readability-lxml`).
* **Eval:** Paste a query you *know* is not in your corpus; Opus should say it doesn’t know.
* **Prod hardening:** Add caching for query embeddings, a reranker (e.g., bge‑reranker‑base), and guardrails (max prompt size, toxicity filters).
