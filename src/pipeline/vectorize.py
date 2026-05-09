import hashlib
import json
import os
import sys

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline.scope import is_supported_category, normalize_category

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "uni_rag"
BATCH_SIZE = 64

os.makedirs(DB_DIR, exist_ok=True)


def load_embedding_model():
    if os.getenv("DORA_LOCAL_MODELS_ONLY", "0") == "1":
        return SentenceTransformer(MODEL_NAME, local_files_only=True)

    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        return SentenceTransformer(MODEL_NAME, local_files_only=False)


def make_hash(text: str, source: str) -> str:
    return hashlib.md5((source + "\n" + text).encode("utf-8")).hexdigest()


def safe_metadata(metadata: dict) -> dict:
    clean = {}

    for key, value in metadata.items():
        if value is None:
            continue
        clean[str(key)] = str(value)

    return clean


def load_chunks(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks if isinstance(chunks, list) else []


def iter_chunk_files() -> list[str]:
    unified_path = os.path.join(PROCESSED_DIR, "unified_chunks.json")
    if os.path.exists(unified_path):
        return [unified_path]

    return [
        os.path.join(PROCESSED_DIR, filename)
        for filename in os.listdir(PROCESSED_DIR)
        if filename.endswith("_chunks.json")
    ]


def prepare_rows(chunks: list[dict], filename: str):
    texts = []
    ids = []
    metadatas = []
    seen = set()

    for index, item in enumerate(chunks):
        text = str(item.get("text", "")).strip()
        metadata = item.get("metadata", {}) or {}

        if len(text) < 20:
            continue

        category = normalize_category(metadata.get("category"))
        if not is_supported_category(category):
            continue

        metadata = {**metadata, "category": category}

        source = metadata.get("source") or filename
        row_id = item.get("chunk_id") or make_hash(text, f"{source}:{index}")

        if row_id in seen:
            continue

        seen.add(row_id)
        texts.append(text)
        ids.append(str(row_id))
        metadatas.append(safe_metadata(metadata))

    return texts, ids, metadatas


def reset_collection(client):
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_batches(collection, model, texts: list[str], ids: list[str], metadatas: list[dict]):
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]
        embeddings = model.encode(
            [f"passage: {text}" for text in batch_texts],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        collection.upsert(
            documents=batch_texts,
            embeddings=embeddings,
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )


def run_vectorize():
    print("\n[VECTORIZE] Started\n")

    files = iter_chunk_files()
    if not files:
        print("[VECTORIZE] No chunk file found")
        return

    model = load_embedding_model()
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = reset_collection(client)

    total = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        chunks = load_chunks(file_path)
        texts, ids, metadatas = prepare_rows(chunks, filename)

        if not texts:
            continue

        print(f"[VECTORIZE] {filename}: {len(texts)} chunks")
        upsert_batches(collection, model, texts, ids, metadatas)
        total += len(texts)

    print(f"\n[VECTORIZE] Ready. Total vectors: {total}\n")


if __name__ == "__main__":
    run_vectorize()
