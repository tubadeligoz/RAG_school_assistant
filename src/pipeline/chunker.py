import os
import json
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CLEAN_DIR = os.path.join(BASE_DIR, "data", "processed", "cleaned")
PDF_DIR = os.path.join(BASE_DIR, "data", "processed", "pdf")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# =========================
# SPLITTER (GENEL AMA GÜÇLÜ)
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,        # PDF + yönetmelik için ideal
    chunk_overlap=180,
    separators=[
        "\n\n",   # paragraf
        "\n",     # satır
        ". ",     # cümle
        " ",      # kelime
    ]
)


# =========================
# QUALITY FILTERS
# =========================
def is_low_quality(text: str) -> bool:
    text = text.strip()

    if len(text.split()) < 35:
        return True

    words = text.split()
    if len(set(words)) / max(len(words), 1) < 0.35:
        return True

    return False


def normalize(text: str) -> str:
    return " ".join(text.split())


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# =========================
# CHUNK PROCESSOR
# =========================
def process_document(file_path: str, source_type: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("content", "")
    url = data.get("url", "")

    if not text.strip():
        return []

    text = normalize(text)

    chunks = text_splitter.split_text(text)

    final_chunks = []
    seen = set()

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()

        if is_low_quality(chunk):
            continue

        h = hash_text(chunk)
        if h in seen:
            continue
        seen.add(h)

        final_chunks.append({
            "chunk_id": f"{os.path.basename(file_path)}_{i}",
            "text": chunk,
            "metadata": {
                "source": os.path.basename(file_path),
                "url": url,
                "type": source_type
            }
        })

    return final_chunks


# =========================
# WRITER
# =========================
def save_chunks(all_chunks, name):
    if not all_chunks:
        return

    out_path = os.path.join(PROCESSED_DIR, f"{name}_chunks.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[CHUNKED] {name} -> {len(all_chunks)} chunks")


# =========================
# RUNNER
# =========================
def run_chunker():
    print("\n✂️ PRO CHUNKER STARTED\n")

    all_chunks = []

    # -------------------------
    # HTML CLEANED FILES
    # -------------------------
    if os.path.exists(CLEAN_DIR):
        for file in os.listdir(CLEAN_DIR):
            if file.endswith(".json"):
                path = os.path.join(CLEAN_DIR, file)
                chunks = process_document(path, "html")
                all_chunks.extend(chunks)

    # -------------------------
    # PDF EXTRACTED FILES
    # -------------------------
    if os.path.exists(PDF_DIR):
        for file in os.listdir(PDF_DIR):
            if file.endswith(".json"):
                path = os.path.join(PDF_DIR, file)
                chunks = process_document(path, "pdf")
                all_chunks.extend(chunks)

    save_chunks(all_chunks, "unified")

    print("\n✅ CHUNKING COMPLETED\n")


if __name__ == "__main__":
    run_chunker()