import json, time, hashlib, shutil
from pathlib import Path
from typing import Dict, TypedDict, Optional, Tuple

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings


# --- CONFIG ---
DOC_DIR = Path("doc")
STATE_PATH = Path("ingest_state.json")

QDRANT_PATH = Path("qdrant_data")
COLLECTION_NAME = "documentation_collection"

MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"
VECTOR_SIZE = 1024


FULL_WIPE_ON_EMPTY_DOC_DIR = False

llm = ChatOllama(model=MODEL_NAME, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


class GraphState(TypedDict):
    question: str
    context: str
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
    # Удаляем коллекцию и state (логический "кэш")
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if STATE_PATH.exists():
        STATE_PATH.unlink(missing_ok=True)

    if FULL_WIPE_ON_EMPTY_DOC_DIR and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)


def convert_and_split(file_path: Path):
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    md = result.document.export_to_markdown()

    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_docs = md_splitter.split_text(md)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=150)
    docs = splitter.split_documents(header_docs)

    for d in docs:
        d.metadata["source"] = file_path.name
        d.metadata["source_path"] = str(file_path)
    return docs


def sync_doc_folder(
    vector_store: QdrantVectorStore,
    client: QdrantClient,
    *,
    silent_when_no_changes: bool = True,
) -> Tuple[int, int]:
    """
    Returns: (added_or_updated_docs_count, total_docs_in_folder)
    """
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in DOC_DIR.iterdir() if p.suffix.lower() in (".pdf", ".docx")])
    total = len(files)

    if total == 0:
        # Папка стала пустой -> чистим индекс и state
        clear_cache_everything(client)
        ensure_collection(client)  # сразу создаём пустую коллекцию, чтобы дальше всё работало
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
        vector_store.add_documents(docs)
        dt = time.perf_counter() - t0

        state[path.name] = {"sha256": file_hash, "indexed_at": time.time(), "chunks": len(docs)}
        changed += 1

        log(f"Готово: {path.name} (чанков: {len(docs)}, {dt:.1f} c)")

    if changed:
        save_state(state)
        log(f"Синхронизация: добавлено/обновлено документов: {changed} из {total}")
    else:
        if not silent_when_no_changes:
            log("Синхронизация: изменений нет.")

    return (changed, total)


def retrieve(state: GraphState, retriever):
    question = state["question"]

    log("Этап 1/2: поиск релевантных фрагментов...")
    t0 = time.perf_counter()
    docs = retriever.invoke(question)
    dt = time.perf_counter() - t0

    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    sources = []
    for d in docs:
        src = d.metadata.get("source")
        if src and src not in sources:
            sources.append(src)

    log(f"Найдено фрагментов: {len(docs)} (источники: {', '.join(sources) if sources else 'нет'}, {dt:.1f} c)")
    return {"context": context, "question": question}


def generate(state: GraphState):
    question = state["question"]
    context = state["context"]


    if not context.strip():
        return {"answer": "В предоставленных документах нет ответа на этот вопрос."}

    log("Этап 2/2: генерация ответа (LLM)...")
    t0 = time.perf_counter()

    prompt = f"""Используй ТОЛЬКО контекст.
Если ответа нет в контексте — так и скажи: "В предоставленных документах нет ответа".

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ (без выдумок):"""

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
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    app = build_graph(retriever)

    log("Старт. Первичная синхронизация папки doc...")
    sync_doc_folder(vector_store, client, silent_when_no_changes=False)

    while True:
        user_query = input("\nВопрос: ").strip()
        if user_query.lower() == "exit":
            break

        sync_doc_folder(vector_store, client, silent_when_no_changes=True)

        t0 = time.perf_counter()
        result = app.invoke({"question": user_query})
        dt = time.perf_counter() - t0

        print("\n" + "=" * 50)
        print(result["answer"])
        print("=" * 50)
        log(f"Готово. Общее время запроса: {dt:.1f} c")
