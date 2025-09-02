# Набор пакетов «kits»

Моно‑репозиторий с переиспользуемыми компонентами для RAG‑пайплайнов:

- kit_common — общие настройки, логирование, модели, ошибки, утилиты
- kit_llm — чат и эмбеддинги через OpenAI‑совместимые API (поддержка base_url)
- kit_chunker — сплиттеры текста/Markdown/PDF с учётом токенов
- kit_vector — абстракция над Qdrant (индексация и поиск)

## Установка

Требования: Python 3.11+ (у вас уже есть venv). Затем:

```
pip install -e .
```

## Быстрый старт

См. интеграционные примеры ниже.

## Настройки (Settings)

Функция `load_settings()` читает переменные окружения и `.env` (если есть):

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- `LLM_CHAT_MODEL`, `LLM_EMBED_MODEL`
- `QDRANT_URL`, `QDRANT_API_KEY`
- `LOG_LEVEL` (по умолчанию `INFO`)

## Логирование

`kit_common.logging.get_logger(name)` выдаёт логгер, который пишет JSON‑подобные строки в stdout и учитывает `LOG_LEVEL`.

## Интеграционные примеры

Пример 1 — индексация PDF (DocuRAG):

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

Пример 2 — QA‑поиск и ответ:

```python
from kit_llm.embed import embed_texts
from kit_llm.chat import chat

# допустим, backend создан как выше
hits = backend.search("docs_docu", embed_texts(["как установить?"])[0], k=3)
context = "\n\n".join(h.payload["text"] for h in hits)
resp = chat([
  {"role":"system","content":"Отвечай кратко и c цитатами (---)."},
  {"role":"user","content": f"Вопрос: как установить?\nКонтекст:\n---\n{context}\n---"}
])
answer = resp["content"]
print(answer)
```

## Тестирование

Запуск тестов:

```
pytest
```

Тесты, требующие внешних сервисов (LLM/Qdrant), мокируются или пропускаются при отсутствии конфигурации.

## Лицензия

MIT.

