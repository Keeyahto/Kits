# Kits Monorepo

Monorepo with reusable packages for RAG pipelines:

- kit_common: shared config, logging, models, errors, utils
- kit_llm: chat and embeddings via OpenAI-compatible clients
- kit_chunker: token-based and paragraph splitters + PDF handling
- kit_vector: Qdrant backend abstraction

Русская версия: [README.ru.md](README.ru.md)

## Installation

- Python 3.11+
- Create and activate venv (already present per your setup), then:

```
pip install -e .
```

## Quick Start

See integration examples below.

## Integration Examples

Example 1 — PDF indexing (DocuRAG):

```python
from kit_common import load_settings
from kit_chunker.pdf import split_pdf
from kit_llm.embed import embed_texts
from kit_vector import get_default_backend, CollectionParams

st = load_settings()
chunks = split_pdf("docs/sample.pdf", max_tokens=400, overlap=40, source="sample.pdf", doc_id="doc1")
vectors = embed_texts([c.text for c in chunks], model=st.llm_embed_model or "text-embedding-3-small")
backend = get_default_backend(st)
backend.ensure_collection(CollectionParams(name="docs_docu", vector_size=len(vectors[0]), distance="cosine"))
backend.upsert("docs_docu", vectors, [c.model_dump() for c in chunks], ids=[c.id for c in chunks])
```

Example 2 — QA search and answer:

```python
from kit_llm.embed import embed_texts
from kit_llm.chat import chat

# assume `backend` from previous example
hits = backend.search("docs_docu", embed_texts(["как установить?"])[0], k=3)
context = "\n\n".join(h.payload["text"] for h in hits)
resp = chat([
  {"role":"system","content":"Отвечай кратко и c цитатами (---)."},
  {"role":"user","content": f"Вопрос: как установить?\nКонтекст:\n---\n{context}\n---"}
])
answer = resp["content"]
print(answer)
```

## Testing

Run unit tests:

```
pytest
```

LLM and Qdrant tests are mocked or skipped when external services are not available or not configured.
