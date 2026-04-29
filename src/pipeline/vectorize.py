import os
import json
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")
os.makedirs(DB_DIR, exist_ok=True)

# =========================
# MODEL
# =========================
MODEL_NAME = "intfloat/multilingual-e5-base"
model = SentenceTransformer(MODEL_NAME)

# =========================
# CHROMA
# =========================
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="uni_rag")


# =========================
# HASH (GLOBAL DEDUP)
# =========================
def make_hash(text: str, source: str) -> str:
    return hashlib.md5((text + source).encode("utf-8")).hexdigest()


# =========================
# SAFE METADATA
# =========================
def safe_metadata(meta: dict) -> dict:
    """
    Chroma metadata için güvenli format
    (ama veri kaybetmez, JSON string yapmaz)
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        clean[str(k)] = str(v)
    return clean


# =========================
# PROCESS FILE
# =========================
def process_file(file_path: str):
    filename = os.path.basename(file_path).replace("_chunks.json", "")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        return

    print(f"[EMBED] {filename} -> {len(chunks)} chunks")

    texts = []
    ids = []
    metas = []

    for i, item in enumerate(chunks):
        text = item["text"].strip()
        meta = item.get("metadata", {})

        # boş skip
        if len(text) < 20:
            continue

        # E5 format
        texts.append(f"passage: {text}")

        # stabil ID (collision-proof)
        uid = make_hash(text, filename + str(i))
        ids.append(uid)

        metas.append(safe_metadata(meta))

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        documents=[t.replace("passage: ", "") for t in texts],
        embeddings=embeddings,
        ids=ids,
        metadatas=metas
    )


# =========================
# RUN
# =========================
def run_vectorize():
    print("\n🧠 VECTORIZE STARTED\n")

    files = [
        f for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_chunks.json")
    ]

    if not files:
        print("❌ chunk dosyası yok")
        return

    for file in files:
        process_file(os.path.join(PROCESSED_DIR, file))

    print("\n✅ VECTOR STORE READY\n")


if __name__ == "__main__":
    run_vectorize()