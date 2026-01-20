import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Tuple


from llmware.configs import LLMWareConfig
from llmware.library import Library
from llmware.retrieval import Query
from langchain_ollama import ChatOllama


PROJECT_DIR = Path(__file__).resolve().parent
DOC_DIR = PROJECT_DIR / "doc"
STATE_PATH = PROJECT_DIR / "ingest_state_llmware.json"

LIBRARY_NAME = "my_rag_lib_cpu"

# Встроенная модель (будет работать на CPU)
EMBEDDING_MODEL = "mini-lm-sbert"

# База векторов
VECTOR_DB = "chromadb"


OLLAMA_MODEL = "qwen2.5:7b-cpu"

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


# === ФУНКЦИИ ===

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> Dict[str, dict]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except:
            return {}
    return {}


def save_state(state: Dict[str, dict]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_environment():
    log("Настройка окружения (Режим CPU)...")
    # Используем пути по умолчанию
    LLMWareConfig().set_active_db("sqlite")
    LLMWareConfig().set_vector_db(VECTOR_DB)
    LLMWareConfig().setup_llmware_workspace()


def get_library() -> Library:
    try:
        return Library().load_library(LIBRARY_NAME)
    except:
        log("Создаем новую библиотеку...")
        try:
            Library().delete_library(LIBRARY_NAME)
        except:
            pass
        return Library().create_new_library(LIBRARY_NAME)


def process_documents(lib: Library) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in DOC_DIR.iterdir() if p.suffix.lower() in (".pdf", ".docx", ".txt")])

    if not files:
        return

    state = load_state()
    new_files = []

    for path in files:
        digest = sha256_file(path)
        if state.get(path.name, {}).get("sha256") != digest:
            new_files.append(path)

    if not new_files:
        return

    log(f"Обработка {len(new_files)} новых файлов...")

    staging = PROJECT_DIR / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    for p in new_files:
        shutil.copy2(p, staging / p.name)

    t0 = time.time()

    # 1. Добавляем файлы
    lib.add_files(input_folder_path=str(staging))

    # 2. Векторизуем (на CPU это займет чуть больше времени)
    log(f"Векторизация (CPU)...")
    lib.install_new_embedding(
        embedding_model_name=EMBEDDING_MODEL,
        vector_db=VECTOR_DB,
        batch_size=50  # Уменьшил размер батча для стабильности на CPU
    )

    log(f"Готово за {time.time() - t0:.1f} сек")

    for p in new_files:
        state[p.name] = {"sha256": sha256_file(p), "indexed_at": time.time()}
    save_state(state)

    shutil.rmtree(staging, ignore_errors=True)


def search_and_answer(lib: Library, query: str):
    # Поиск без указания модели (она подтянется сама)
    search_results = Query(lib).semantic_query(query, result_count=5)

    context_parts = []
    for r in search_results:
        text = r.get("text", "").strip()
        source = r.get("file_source", "doc")
        page = r.get("page_num", "")

        if text:
            meta = f"[{source}" + (f", стр.{page}" if page else "") + "]"
            context_parts.append(f"{meta} {text}")

    context_str = "\n\n".join(context_parts)

    if not context_str:
        print("\n[!] Не найдено информации в документах.")
        return

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО контекст ниже.

КОНТЕКСТ:
{context_str}

ВОПРОС: {query}
ОТВЕТ:"""

    print("\nГенерирую ответ...")
    response = llm.invoke(prompt).content

    print("=" * 60)
    print(response)
    print("=" * 60)


# === ЗАПУСК ===
if __name__ == "__main__":
    try:
        setup_environment()
        lib = get_library()

        # Сначала загружаем документы
        process_documents(lib)

        print("\n--- Система готова (CPU Mode) ---")
        while True:
            try:
                q = input("\nВведите вопрос (или exit): ").strip()
            except EOFError:
                break

            if not q or q.lower() == "exit":
                break

            # Проверяем новые файлы
            process_documents(lib)

            # Ищем и отвечаем
            search_and_answer(lib, q)

    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        input("Нажмите Enter для выхода...")