import json, time, hashlib, shutil, re
from pathlib import Path
from typing import Dict, TypedDict, Tuple, List, Any, Optional, Callable

import streamlit as st

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings

import pdfplumber
from langchain_core.documents import Document
import os


# ---------------- CONFIG ----------------
DOC_DIR = Path("doc")
STATE_PATH = Path("ingest_state.json")

QDRANT_PATH = Path("qdrant_data")
COLLECTION_NAME = "documentation_collection"

MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"
VECTOR_SIZE = 1024

FULL_WIPE_ON_EMPTY_DOC_DIR = False
ENABLE_TABLE_FALLBACK = True

TOP_K = 5
CHUNK_SIZE = 2200
CHUNK_OVERLAP = 80

CLAUSE_LINE_RE = re.compile(
    r"(?m)^[^\S\r\n]*(?:[-–—•]\s*)?"
    r"(\d{1,3}(?:\.\d{1,3}){0,3})\.?"
    r"(?=[^\d]|$)"
)
USER_AVATAR = "icon/user.png"
ASSISTANT_AVATAR = "icon/robot.png"
SIDEBAR_LOGO = "icon/logo.svg"

class GraphState(TypedDict):
    question: str
    sources: List[Dict[str, Any]]
    answer: str


# ---------------- UI helpers ----------------
def inject_css() -> None:
    st.markdown(
        """
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 880px; }

/* Title */
.app-title {
  text-align: center;
  font-size: 2.05rem;
  font-weight: 760;
  letter-spacing: 0.2px;
  margin: 0.2rem 0 0.15rem 0;
}
.app-subtitle {
  text-align: center;
  opacity: 0.72;
  margin-bottom: 0.8rem;
}


div.stButton > button[kind="primary"] {
  width: 100%;
  padding: 0.95rem 1.0rem !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
}


.transient-log {
  opacity: 0.60;
  font-size: 0.92em;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  padding: 0.45rem 0.65rem;
  border-left: 3px solid rgba(37, 99, 235, 0.55);
  background: rgba(37, 99, 235, 0.06);
  border-radius: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
/* ширина раскрытого сайдбара */
section[data-testid="stSidebar"] {
  width: 420px !important;
}

/* чтобы контент внутри тоже подстроился */
section[data-testid="stSidebar"] > div {
  width: 420px !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] .stSidebarContent {
  display: none !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
  background: rgba(219, 234, 254, 0.85) !important;
  border-radius: 30px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def ui_log_render(placeholder, text: str) -> None:
    placeholder.markdown(f"<div class='transient-log'>{text}</div>", unsafe_allow_html=True)


def save_uploaded_to_doc(uploaded) -> Path:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOC_DIR / uploaded.name
    with out_path.open("wb") as f:
        f.write(uploaded.getbuffer())
    return out_path


# ---------------- core helpers ----------------
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

    uniq, seen = [], set()
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
        d.metadata["clause_last"] = last_clause or ""
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


def extract_tables_pdfplumber(file_path: Path, logf: Callable[[str], None]) -> List[Dict[str, Any]]:
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
        logf("[pdfplumber] ok")
        return out
    except Exception as e:
        logf(f"[pdfplumber] ошибка: {type(e).__name__}: {e}")
        return []


def convert_and_split(file_path: Path, logf: Callable[[str], None]) -> List[Document]:
    docs_out: List[Document] = []

    converter = DocumentConverter()
    logf(f"[Docling] convert() -> {file_path.name}")
    t0 = time.perf_counter()
    result = converter.convert(str(file_path))
    dt = time.perf_counter() - t0
    logf(f"[Docling] OK ({dt:.1f} c)")

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

        annotate_last_clause_for_chunks(filtered_docs)
        docs_out.extend(filtered_docs)
    else:
        logf("[Docling] markdown пустой (скан/таблицы/OCR).")

    if ENABLE_TABLE_FALLBACK and file_path.suffix.lower() == ".pdf":
        for rec in extract_tables_pdfplumber(file_path, logf=logf):
            docs_out.append(Document(page_content=rec["text"], metadata=rec["meta"]))

    logf(f"[Split] чанков: {len(docs_out)}")
    return docs_out


def sync_doc_folder(
    vector_store: QdrantVectorStore,
    client: QdrantClient,
    logf: Callable[[str], None],
    silent_when_no_changes: bool = True,
) -> Tuple[int, int]:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in DOC_DIR.iterdir() if p.suffix.lower() in (".pdf", ".docx")])
    total = len(files)

    if total == 0:
        clear_cache_everything(client)
        ensure_collection(client)
        logf("[Sync] doc/ пусто -> индекс/кэш очищены.")
        return (0, 0)

    state = load_state()
    changed = 0

    for path in files:
        file_hash = sha256_file(path)
        prev = state.get(path.name)

        if prev and prev.get("sha256") == file_hash:
            continue

        logf(f"[Sync] Индексация: {path.name}")
        t0 = time.perf_counter()

        docs = convert_and_split(path, logf=logf)
        if not docs:
            logf(f"[Sync] Пропуск: {path.name} (пусто).")
            state[path.name] = {"sha256": file_hash, "indexed_at": time.time(), "chunks": 0, "status": "empty"}
            changed += 1
            save_state(state)
            continue

        vector_store.add_documents(docs)

        dt = time.perf_counter() - t0
        state[path.name] = {"sha256": file_hash, "indexed_at": time.time(), "chunks": len(docs), "status": "ok"}
        changed += 1
        logf(f"[Sync] OK: {path.name} ({len(docs)} чанков, {dt:.1f} c)")

    if changed:
        save_state(state)
        logf(f"[Sync] Обновлено: {changed}/{total}")
    else:
        if not silent_when_no_changes:
            logf("[Sync] Изменений нет.")

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


def retrieve(state: GraphState, retriever, logf: Callable[[str], None]):
    q = state["question"]
    logf("Поиск фрагментов…")
    t0 = time.perf_counter()
    docs = retriever.invoke(q)
    dt = time.perf_counter() - t0

    packed = []
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        packed.append({"id": i, "text": d.page_content, "meta": meta})
        src = meta.get("source")
        if src and src not in sources:
            sources.append(src)

    logf(f"Найдено: {len(docs)} (источники: {', '.join(sources) if sources else 'нет'}, {dt:.1f} c)")
    return {"sources": packed, "question": q}


def generate(state: GraphState, llm: ChatOllama, logf: Callable[[str], None]):
    q = state["question"]
    sources = state["sources"]

    if not sources:
        return {"answer": "В предоставленных документах нет ответа на этот вопрос.", "llm_seconds": 0.0}

    context = format_sources_for_prompt(sources)

    logf("Генерация (LLM)…")
    t0 = time.perf_counter()

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО источники ниже.

Правила цитирования (обязательно):
1) Любое утверждение в ответе должно иметь ссылку на источник в формате [Документ id], где id - номер документа по порядку (1,2,3..).
2) Рядом с каждой ссылкой [Документ id] добавляй человеко-читаемую форму:
   - (Пункт X) где X = clause из loc.
   - Если clause нет: (пункт неизвестен).
3) В конце добавь раздел "Источники:" и перечисли только те источники, на которые ссылалась:
   - [Документ id] Пункт: X. Документ: Y, где X = clause из loc, Y = section из doc

Если ответа нет в источниках — напиши: "В предоставленных документах нет ответа".

ИСТОЧНИКИ:
{context}

ВОПРОС:
{q}

ОТВЕТ:"""

    response = llm.invoke(prompt)
    llm_dt = time.perf_counter() - t0
    return {"answer": response.content, "llm_seconds": llm_dt}


def build_graph(retriever, llm: ChatOllama, logf: Callable[[str], None]):
    wf = StateGraph(GraphState)
    wf.add_node("retrieve", lambda s: retrieve(s, retriever, logf=logf))
    wf.add_node("generate", lambda s: generate(s, llm=llm, logf=logf))
    wf.set_entry_point("retrieve")
    wf.add_edge("retrieve", "generate")
    wf.add_edge("generate", END)
    return wf.compile()


# ---------------- Streamlit app ----------------
st.set_page_config(page_title="Поиск по документации", layout="centered")
inject_css()
st.logo(SIDEBAR_LOGO, size="large")  # можно small/medium/large
# Session init
if "started" not in st.session_state:
    st.session_state.started = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.divider()
st.markdown("<div class='app-title'> 🔎 Поиск по документации</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Локальный поиск с цитированием</div>", unsafe_allow_html=True)
st.divider()
# Upload pinned under header
with st.sidebar:
    with st.expander("Загрузка документов", expanded=False):
        st.markdown(
            "<div class='muted-block'>Добавь документы в папку <code>doc/</code></div>",
            unsafe_allow_html=True,
        )

        up = st.file_uploader(
            "или загрузи здесь ↓",
            type=["pdf", "docx"],
            accept_multiple_files=False,
        )
        if up is not None:
            p = save_uploaded_to_doc(up)
            st.success(f"Сохранено: {p.name}")

            if st.session_state.started and st.button("Переиндексировать сейчас", use_container_width=True):
                st.session_state._do_reindex_now = True
                st.rerun()


# Backend init
if "client" not in st.session_state:
    st.session_state.client = QdrantClient(path=str(QDRANT_PATH))
    ensure_collection(st.session_state.client)

if "embeddings" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model=EMBED_MODEL)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = QdrantVectorStore(
        client=st.session_state.client,
        collection_name=COLLECTION_NAME,
        embedding=st.session_state.embeddings,
    )

if "retriever" not in st.session_state:
    st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": TOP_K})

if "llm" not in st.session_state:
    st.session_state.llm = ChatOllama(model=MODEL_NAME, temperature=0)


if not st.session_state.started:
    if st.button("Начать чат", type="primary", use_container_width=True):
        # show transient log under header using st.empty
        tmp = st.empty()
        with st.spinner("Синхронизация документов…"):
            try:
                def logf(msg: str) -> None:
                    ui_log_render(tmp, msg)

                sync_doc_folder(
                    vector_store=st.session_state.vector_store,
                    client=st.session_state.client,
                    logf=logf,
                    silent_when_no_changes=False,
                )
                st.session_state.started = True
            finally:
                tmp.empty()
        st.rerun()

    st.info("Нажми «Начать чат», чтобы проиндексировать документы и открыть чат.")
    st.stop()

# Optional reindex after upload
if st.session_state.get("_do_reindex_now") is True:
    st.session_state._do_reindex_now = False
    tmp = st.empty()
    with st.spinner("Синхронизация…"):
        try:
            def logf(msg: str) -> None:
                ui_log_render(tmp, msg)

            sync_doc_folder(
                vector_store=st.session_state.vector_store,
                client=st.session_state.client,
                logf=logf,
                silent_when_no_changes=False,
            )
        finally:
            tmp.empty()
    st.rerun()

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=m.get("avatar")):
        st.markdown(m["content"])


# Chat input [web:867]
prompt = st.chat_input("Задай вопрос по документации…")

if prompt:
    # Show user message immediately + reserve log slot right under it
    with st.chat_message("user", avatar="icon/user.png"):
        st.markdown(prompt)
        log_slot = st.empty()  # место под логи прямо под запросом [web:941]

    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": USER_AVATAR})

    with st.chat_message("assistant", avatar="icon/robot.png"):
        with st.spinner("Обрабатываю вопрос…"):
            try:
                # logger, writes into log_slot under user message
                def logf(msg: str) -> None:
                    ui_log_render(log_slot, msg)

                # sync before question (as in original)
                sync_doc_folder(
                    vector_store=st.session_state.vector_store,
                    client=st.session_state.client,
                    logf=logf,
                    silent_when_no_changes=True,
                )

                # Build graph with this logger
                graph = build_graph(
                    retriever=st.session_state.retriever,
                    llm=st.session_state.llm,
                    logf=logf,
                )

                t0 = time.perf_counter()
                result = graph.invoke({"question": prompt, "sources": [], "answer": ""})
                total_dt = time.perf_counter() - t0

                answer = result.get("answer", "")
                llm_seconds = result.get("llm_seconds", None)
                if llm_seconds is not None:
                    answer = answer + f"\n\n---\nВремя LLM: {llm_seconds:.1f} c\nВремя запроса (всего): {total_dt:.1f} c"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "avatar": ASSISTANT_AVATAR})
            finally:
                # remove logs after processing
                log_slot.empty()

    st.rerun()
