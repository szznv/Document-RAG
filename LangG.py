import json, time, hashlib, shutil, re
from pathlib import Path
from typing import Dict, TypedDict, Tuple, List, Any, Optional

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings

import pdfplumber  # pip install pdfplumber  [web:502]
from langchain_core.documents import Document


# --- CONFIG ---
DOC_DIR = Path("doc")
STATE_PATH = Path("ingest_state.json")

QDRANT_PATH = Path("qdrant_data")
COLLECTION_NAME = "documentation_collection"

MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"
VECTOR_SIZE = 1024

FULL_WIPE_ON_EMPTY_DOC_DIR = False
ENABLE_TABLE_FALLBACK = True

# Retrieval
TOP_K = 5

# Split
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 150

llm = ChatOllama(model=MODEL_NAME, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

CLAUSE_LINE_RE = re.compile(
    r"(?m)^[^\S\r\n]*(?:[-–—•]\s*)?"
    r"(\d{1,3}(?:\.\d{1,3}){0,3})\.?"
    r"(?=[^\d]|$)"
)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


class GraphState(TypedDict):
    question: str
    sources: List[Dict[str, Any]]  # [{"id": int, "text": str, "meta": dict}]
    answer: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> Dict[str, dict]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: Dict[str, dict]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def clear_cache_everything(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if STATE_PATH.exists():
        STATE_PATH.unlink(missing_ok=True)

    if FULL_WIPE_ON_EMPTY_DOC_DIR and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)


def sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in s)
    return s.strip()


def build_section_path(meta: Dict[str, Any]) -> str:

    if not meta:
        return ""

    candidates = []
    for k in ["H1", "H2", "H3", "H4", "header_1", "header_2", "header_3", "header_4"]:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    for k in ["Header 1", "Header 2", "Header 3", "Header 4"]:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    if not candidates:
        for k, v in meta.items():
            if isinstance(k, str) and k.upper().startswith("H") and isinstance(v, str) and v.strip():
                candidates.append(v.strip())

    uniq = []
    seen = set()
    for x in candidates:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    return " > ".join(uniq[:4])


def normalize_clause(raw: str) -> str:
    return raw.strip().rstrip(".")


def annotate_last_clause_for_chunks(docs: List[Document]) -> None:

    last_clause: Optional[str] = None

    for d in docs:
        txt = d.page_content or ""
        # Считаем все пункты, встретившиеся ВНУТРИ чанка, но last_clause для него
        # должен быть тем, что было ДО чанка.
        d.metadata["clause_last"] = last_clause or ""

        # Теперь обновим last_clause по содержимому текущего чанка для следующих чанков.
        for m in CLAUSE_LINE_RE.finditer(txt):
            last_clause = normalize_clause(m.group(1))


def table_to_text(table: List[List[str]], max_cell_len: int = 120) -> str:
    lines = []
    for r_i, row in enumerate(table, start=1):
        cells = []
        for c_i, cell in enumerate(row, start=1):
            cell = "" if cell is None else str(cell)
            cell = sanitize_text(cell)
            if len(cell) > max_cell_len:
                cell = cell[:max_cell_len] + "…"
            cells.append(f"c{c_i}={cell}")
        line = f"row{r_i}: " + "; ".join(cells)
        if sanitize_text(line.replace("c1=", "").replace("c2=", "")):
            lines.append(line)
    return "\n".join(lines).strip()


def extract_tables_pdfplumber(file_path: Path) -> List[Dict[str, Any]]:
    """
    pdfplumber умеет extract_tables() на странице. [web:502]
    """
    out: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if not tables:
                    continue
                for t_idx, table in enumerate(tables, start=1):
                    txt = table_to_text(table)
                    if not txt:
                        continue
                    out.append(
                        {
                            "text": (
                                "ТАБЛИЦА (pdfplumber)\n"
                                f"Источник: {file_path.name}\n"
                                f"Страница: {page_idx}\n"
                                f"Таблица: {t_idx}\n\n"
                                f"{txt}"
                            ),
                            "meta": {
                                "source": file_path.name,
                                "source_path": str(file_path),
                                "kind": "pdfplumber_table",
                                "page": page_idx,
                                "table_id": t_idx,
                            },
                        }
                    )
        log(f"pdfplumber: извлечено таблиц: {len(out)}")
        return out
    except Exception as e:
        log(f"pdfplumber: ошибка извлечения таблиц: {type(e).__name__}: {e}")
        return []


def convert_and_split(file_path: Path) -> List[Document]:
    docs_out: List[Document] = []

    converter = DocumentConverter()
    log(f"Docling: convert() -> {file_path.name}")
    t0 = time.perf_counter()
    result = converter.convert(str(file_path))
    dt = time.perf_counter() - t0
    log(f"Docling: convert() OK ({dt:.1f} c)")

    md = sanitize_text(result.document.export_to_markdown())

    if md:
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_docs = md_splitter.split_text(md)

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.split_documents(header_docs)

        chunk_id = 0
        filtered_docs: List[Document] = []
        for d in docs:
            txt = sanitize_text(d.page_content)
            if len(txt) < 40:
                continue

            chunk_id += 1
            d.page_content = txt

            d.metadata["source"] = file_path.name
            d.metadata["source_path"] = str(file_path)
            d.metadata["kind"] = "docling_md"
            d.metadata["chunk_id"] = chunk_id

            sec = build_section_path(d.metadata)
            if sec:
                d.metadata["section_path"] = sec

            filtered_docs.append(d)

        # Ключевое: проставляем clause_last по порядку чанков
        annotate_last_clause_for_chunks(filtered_docs)

        docs_out.extend(filtered_docs)
    else:
        log("Docling: ВНИМАНИЕ markdown пустой (возможен скан/OCR/таблицы).")

    # --- Fallback: таблицы ---
    if ENABLE_TABLE_FALLBACK and file_path.suffix.lower() == ".pdf":
        for rec in extract_tables_pdfplumber(file_path):
            docs_out.append(Document(page_content=rec["text"], metadata=rec["meta"]))

    log(f"Итого чанков на индекс: {len(docs_out)}")
    return docs_out


def sync_doc_folder(
    vector_store: QdrantVectorStore,
    client: QdrantClient,
    silent_when_no_changes: bool = True,
) -> Tuple[int, int]:
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in DOC_DIR.iterdir() if p.suffix.lower() in (".pdf", ".docx")])
    total = len(files)

    if total == 0:
        clear_cache_everything(client)
        ensure_collection(client)
        log("Папка doc пуста -> индекс и кэш очищены.")
        return (0, 0)

    state = load_state()
    changed = 0

    for path in files:
        file_hash = sha256_file(path)
        prev = state.get(path.name)

        if prev and prev.get("sha256") == file_hash:
            continue

        log(f"Индексация: {path.name}")
        t0 = time.perf_counter()

        docs = convert_and_split(path)
        if not docs:
            log(f"Пропуск {path.name}: нет данных для индексации.")
            state[path.name] = {"sha256": file_hash, "indexed_at": time.time(), "chunks": 0, "status": "empty"}
            changed += 1
            save_state(state)
            continue

        vector_store.add_documents(docs)

        dt = time.perf_counter() - t0
        state[path.name] = {"sha256": file_hash, "indexed_at": time.time(), "chunks": len(docs), "status": "ok"}
        changed += 1
        log(f"Готово: {path.name} (чанков: {len(docs)}, {dt:.1f} c)")

    if changed:
        save_state(state)
        log(f"Синхронизация: добавлено/обновлено документов: {changed} из {total}")
    else:
        if not silent_when_no_changes:
            log("Синхронизация: изменений нет.")

    return (changed, total)


def format_sources_for_prompt(sources: List[Dict[str, Any]]) -> str:
    parts = []
    for s in sources:
        meta = s.get("meta", {}) or {}
        doc = meta.get("source", "unknown")
        kind = meta.get("kind", "text")
        section = meta.get("section_path", "")
        clause_last = meta.get("clause_last", "")
        chunk_id = meta.get("chunk_id", None)
        page = meta.get("page", None)
        table_id = meta.get("table_id", None)

        loc_parts = []
        if clause_last:
            loc_parts.append(f"clause={clause_last}")
        if section:
            loc_parts.append(f"section={section}")
        if chunk_id is not None:
            loc_parts.append(f"chunk={chunk_id}")
        if page is not None:
            loc_parts.append(f"page={page}")
        if table_id is not None:
            loc_parts.append(f"table={table_id}")
        loc_parts.append(f"kind={kind}")
        loc = "; ".join(loc_parts)

        parts.append(
            f"<source id=\"{s['id']}\" doc=\"{doc}\" loc=\"{loc}\">\n"
            f"{s['text']}\n"
            f"</source>"
        )
    return "\n\n".join(parts)


def retrieve(state: GraphState, retriever):
    question = state["question"]

    log("Этап 1/2: поиск релевантных фрагментов...")
    t0 = time.perf_counter()
    docs = retriever.invoke(question)
    dt = time.perf_counter() - t0

    packed = []
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        packed.append({"id": i, "text": d.page_content, "meta": meta})
        src = meta.get("source")
        if src and src not in sources:
            sources.append(src)

    log(f"Найдено фрагментов: {len(docs)} (источники: {', '.join(sources) if sources else 'нет'}, {dt:.1f} c)")
    return {"sources": packed, "question": question}


def generate(state: GraphState):
    question = state["question"]
    sources = state["sources"]

    if not sources:
        return {"answer": "В предоставленных документах нет ответа на этот вопрос."}

    context = format_sources_for_prompt(sources)

    log("Этап 2/2: генерация ответа (LLM) с цитированием...")
    t0 = time.perf_counter()

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО источники ниже.

Правила цитирования (обязательно):
1) Любое утверждение в ответе должно иметь ссылку на источник в формате [Документ id].
2) Рядом с каждой ссылкой [Документ id] добавляй человеко-читаемую форму:
   - (Пункт X) где X = clause из loc.
   - Если clause нет: (пункт неизвестен).
   Пример: "...текст..." [1] (Пункт 5.4.19).
3) В конце добавь раздел "Источники:" и перечисли только те источники, на которые ссылалась:
   - [Документ id] Пункт: X. Документ: Y, где X = clause из loc, Y = section из doc

Если ответа нет в источниках — напиши: "В предоставленных документах нет ответа".

ИСТОЧНИКИ:
{context}

ВОПРОС:
{question}

ОТВЕТ:"""

    response = llm.invoke(prompt)
    dt = time.perf_counter() - t0
    log(f"Генерация завершена ({dt:.1f} c)")
    return {"answer": response.content}


def build_graph(retriever):
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", lambda s: retrieve(s, retriever))
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


if __name__ == "__main__":
    client = QdrantClient(path=str(QDRANT_PATH))
    ensure_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    app = build_graph(retriever)

    log("Старт. Первичная синхронизация папки doc...")
    sync_doc_folder(vector_store, client, silent_when_no_changes=False)

    while True:
        user_query = input("\nВопрос (или 'exit'): ").strip()
        if user_query.lower() == "exit":
            break

        sync_doc_folder(vector_store, client, silent_when_no_changes=True)

        t0 = time.perf_counter()
        result = app.invoke({"question": user_query, "sources": [], "answer": ""})
        dt = time.perf_counter() - t0

        print("\n" + "=" * 50)
        print(result["answer"])
        print("=" * 50)
        log(f"Готово. Общее время запроса: {dt:.1f} c")

