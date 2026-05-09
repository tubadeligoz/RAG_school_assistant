import os
import json
import hashlib
import re
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline.curated_loader import load_curated_chunks
from src.pipeline.scope import detect_scope_category, is_supported_category, normalize_category

CLEAN_DIR = os.path.join(BASE_DIR, "data", "processed", "cleaned")
PDF_DIR = os.path.join(BASE_DIR, "data", "processed", "pdf")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# =========================
# SPLITTER
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1350,
    chunk_overlap=260,
    separators=["\n\n", "\n", ". ", "; ", " "]
)


# =========================
# HELPERS
# =========================
def is_low_quality(text: str) -> bool:
    words = text.split()
    word_count = len(words)

    has_signal = bool(
        re.search(r"\d", text)
        or re.search(
            r"(başvuru|staj|erasmus|değişim|çift anadal|yandal|çap|öğretim|akademik kadro)",
            text,
            re.IGNORECASE,
        )
    )

    if word_count < 20:
        return True

    if word_count < 35 and not has_signal:
        return True

    if len(set(words)) / max(word_count, 1) < 0.30:
        return True

    return False


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", text)


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


TR_MAP = str.maketrans({
    "ç": "c",
    "ğ": "g",
    "ı": "i",
    "ö": "o",
    "ş": "s",
    "ü": "u",
    "Ç": "c",
    "Ğ": "g",
    "İ": "i",
    "I": "i",
    "Ö": "o",
    "Ş": "s",
    "Ü": "u",
})


def fold_text(text: str) -> str:
    return text.casefold().translate(TR_MAP)


def has_any(text: str, needles: tuple[str, ...]) -> bool:
    folded = fold_text(text)
    return any(needle in folded for needle in needles)


def source_has(source_folded: str, needles: tuple[str, ...]) -> bool:
    return any(needle in source_folded for needle in needles)


def is_erasmus_announcement(chunk_folded: str, source_type: str) -> bool:
    if source_type != "html" or "erasmus" not in chunk_folded:
        return False

    return any(
        signal in chunk_folded
        for signal in (
            "basvuru",
            "dil sinavi",
            "hareketliligi",
            "hibe",
            "secim sonucu",
            "sonuclari",
            "ilan",
        )
    )


def infer_category(data: dict, file_path: str, source_type: str, chunk: str) -> str:
    existing = normalize_category(data.get("category"))
    filename = os.path.basename(file_path)
    source_text = " ".join([
        filename,
        str(data.get("url", "")),
        str(data.get("title", "")),
    ])
    source_folded = fold_text(source_text)
    chunk_folded = fold_text(chunk)
    haystack = f"{source_folded} {chunk_folded}"

    erasmus_source = source_has(
        source_folded,
        (
            "erasmus",
            "uluslararasi-ogrenci-degisim",
        ),
    )
    if erasmus_source or is_erasmus_announcement(chunk_folded, source_type):
        return "erasmus"

    if (
        "cift-anadal-ve-yandal" in source_folded
        or (
            ("akademik-takvim" in source_folded or "akademik-takvimi" in source_folded)
            and ("cift anadal" in chunk_folded or "yandal" in chunk_folded)
            and any(signal in chunk_folded for signal in ("tarih", "basvuru", "kayit", "ilan"))
        )
    ):
        return "cift_anadal_yandal"

    if (
        "staj-uygulamalari" in source_folded
        or "uygulamali-egitimler" in source_folded
    ):
        return "staj"

    if source_type == "html" and any(
        signal in chunk_folded
        for signal in (
            "ogretim elemani",
            "ogretim elemanlari",
            "ogretim gorevlisi",
            "ogretim gorevlileri",
            "ogretim uyesi",
            "ogretim uyeleri",
            "akademik kadro",
            "prof. dr",
            "doc. dr",
            "dr. ogretim",
            "arastirma gorevlisi",
        )
    ):
        return "ogretim_gorevlileri"

    if is_supported_category(existing):
        return existing

    return "out_of_scope"


def build_prefix(data: dict) -> str:
    """
    🔥 Chunk başına context ekler
    """
    parts = []

    if data.get("title"):
        parts.append(data["title"])

    if data.get("category"):
        parts.append(data["category"])

    return " > ".join(parts)


# =========================
# PROCESS
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

        # 🔥 CONTEXT EKLE
        category = infer_category(data, file_path, source_type, chunk)
        if not is_supported_category(category):
            continue

        category = normalize_category(category)
        prefix = build_prefix({**data, "category": category})

        if prefix:
            chunk = f"{prefix}\n{chunk}"

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
                "type": source_type,
                "category": category,
                "title": data.get("title", "")
            }
        })

    return final_chunks


# =========================
# RUNNER
# =========================
def run_chunker():
    print("\n[CHUNKER] Started\n")

    all_chunks = []

    # HTML
    if os.path.exists(CLEAN_DIR):
        for file in os.listdir(CLEAN_DIR):
            if file.endswith(".json"):
                path = os.path.join(CLEAN_DIR, file)
                all_chunks.extend(process_document(path, "html"))

    # PDF
    if os.path.exists(PDF_DIR):
        for file in os.listdir(PDF_DIR):
            if file.endswith(".json"):
                path = os.path.join(PDF_DIR, file)
                all_chunks.extend(process_document(path, "pdf"))

    # Curated records
    curated_chunks = load_curated_chunks()
    if curated_chunks:
        print(f"[CHUNKER] Curated chunks: {len(curated_chunks)}")
        all_chunks.extend(curated_chunks)

    if not all_chunks:
        print("[CHUNKER] No chunk generated")
        return

    out_path = os.path.join(PROCESSED_DIR, "unified_chunks.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[CHUNKER] Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    run_chunker()
