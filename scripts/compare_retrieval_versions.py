import json
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TMP_DIR = ROOT / ".rag_compare" / "old"
OLD_PROCESSED_DIR = TMP_DIR / "data" / "processed"
OLD_VECTOR_DIR = TMP_DIR / "data" / "vector_store"
OLD_COLLECTION = "old_uni_rag"

OLD_CHUNK_SIZE = 900
OLD_CHUNK_OVERLAP = 180

CASES = [
    {
        "id": "staj_evrak",
        "query": "staj için gerekli evraklar neler?",
        "expected_source": "staj-uygulamalari-usul-ve-esaslari",
        "required_any": ("Staj Başvuru", "nüfus", "staj defteri"),
    },
    {
        "id": "staj_belge_yeri",
        "query": "staj belgelerine nereden ulaşabilirim?",
        "expected_source": "staj-uygulamalari-usul-ve-esaslari",
        "required_any": ("internet sayfalarında", "Staj Belgeleri", "staj defterini"),
    },
    {
        "id": "staj_sorumlu",
        "query": "sorumlu eğitim elemanını nereden bulabilirim?",
        "expected_source": "staj-uygulamalari-usul-ve-esaslari",
        "required_any": ("Bölüm/Program başkanları", "Sorumlu Öğretim Elemanı", "sorumlu öğretim elemanı"),
    },
    {
        "id": "erasmus_basvuru",
        "query": "erasmus başvuruları hakkında bilgi verir misin?",
        "expected_source": "uluslararasi-ogrenci-degisim",
        "required_any": ("Erasmus Programı için başvurular", "Uluslararası Öğrenci Ofisi", "hareketliliği"),
    },
    {
        "id": "erasmus_dil_sinavi",
        "query": "erasmus dil sınavı ne zaman?",
        "expected_source": "www_dogus_edu_tr_clean",
        "required_any": ("Dil Sınavı", "Student Mobility Language Exam", "24 Nisan 2026"),
    },
    {
        "id": "cap_ortalama",
        "query": "çap için ortalama kaç olmalı?",
        "expected_source": "cift-anadal-ve-yandal",
        "required_any": ("en az 3,00", "en üst %20", "çift anadal programına başvurabilmesi"),
    },
    {
        "id": "yandal_sart",
        "query": "yandal şartları nelerdir?",
        "expected_source": "cift-anadal-ve-yandal",
        "required_any": ("en az 2.50", "yandal programına başvurabilmesi", "Yandal Sertifikası"),
    },
    {
        "id": "ogretim_gorevlileri",
        "query": "öğretim görevlileri kimler?",
        "expected_source": "www_dogus_edu_tr",
        "required_any": ("Öğretim", "Akademik Kadro", "Dr."),
    },
]


def old_normalize(text: str) -> str:
    return " ".join(str(text or "").split())


def old_is_low_quality(text: str) -> bool:
    text = text.strip()
    words = text.split()
    if len(words) < 35:
        return True
    return len(set(words)) / max(len(words), 1) < 0.35


def split_text(text: str, chunk_size: int = OLD_CHUNK_SIZE, overlap: int = OLD_CHUNK_OVERLAP) -> list[str]:
    text = old_normalize(text)
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]

        if end < len(text):
            split_at = max(
                chunk.rfind("\n\n"),
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind(" "),
            )
            if split_at > chunk_size * 0.55:
                end = start + split_at + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def old_process_document(path: Path, source_type: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    text = data.get("content", "")
    if not str(text).strip():
        return []

    final_chunks = []
    seen = set()
    for index, chunk in enumerate(split_text(text)):
        if old_is_low_quality(chunk):
            continue

        key = hash(chunk)
        if key in seen:
            continue
        seen.add(key)
        final_chunks.append(
            {
                "chunk_id": f"{path.name}_{index}",
                "text": chunk,
                "metadata": {
                    "source": path.name,
                    "url": data.get("url", ""),
                    "type": source_type,
                },
            }
        )

    return final_chunks


def build_old_chunks() -> list[dict]:
    chunks = []
    for folder, source_type in (
        (ROOT / "data" / "processed" / "cleaned", "html"),
        (ROOT / "data" / "processed" / "pdf", "pdf"),
    ):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            chunks.extend(old_process_document(path, source_type))

    OLD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (OLD_PROCESSED_DIR / "unified_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunks


def load_new_chunks() -> list[dict]:
    path = ROOT / "data" / "processed" / "unified_chunks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def reset_collection(client, name: str):
    try:
        client.delete_collection(name)
    except NotFoundError:
        pass
    return client.get_or_create_collection(name=name)


def vectorize_old_chunks(chunks: list[dict], embedder):
    if OLD_VECTOR_DIR.exists():
        shutil.rmtree(OLD_VECTOR_DIR)
    OLD_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(OLD_VECTOR_DIR))
    collection = reset_collection(client, OLD_COLLECTION)

    texts = []
    ids = []
    metadatas = []
    for index, item in enumerate(chunks):
        text = str(item.get("text", "")).strip()
        if len(text) < 20:
            continue
        source = item.get("metadata", {}).get("source", "unknown")
        texts.append(text)
        ids.append(f"old_{index}_{abs(hash(source + text))}")
        metadatas.append({k: str(v) for k, v in item.get("metadata", {}).items() if v is not None})

    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        embeddings = embedder.encode(
            [f"passage: {text}" for text in batch_texts],
            batch_size=batch_size,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=ids[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
        )

    return collection


def old_retrieve(query: str, collection, embedder, reranker, top_k: int = 5):
    query_embedding = embedder.encode(f"query: {query}").tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(20, collection.count()),
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    seen = set()
    candidates = []
    for doc, metadata in zip(docs, metas):
        key = doc[:200]
        if key in seen:
            continue
        seen.add(key)
        candidates.append((doc, metadata or {}))

    if not candidates:
        return []

    scores = reranker.predict([[query, doc] for doc, _ in candidates], show_progress_bar=False)
    ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
    return [
        {
            "text": doc,
            "metadata": metadata,
            "score": float(score),
        }
        for score, (doc, metadata) in ranked[:top_k]
    ]


def contains_any(chunks: list[dict], signals: tuple[str, ...], limit: int = 3) -> bool:
    text = "\n".join(chunk.get("text", "") for chunk in chunks[:limit]).casefold()
    return any(signal.casefold() in text for signal in signals)


def top_source(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    metadata = chunks[0].get("metadata", {})
    return str(metadata.get("source", ""))


def source_hit(chunks: list[dict], expected_source: str, limit: int = 3) -> bool:
    expected = expected_source.casefold()
    return any(expected in str(chunk.get("metadata", {}).get("source", "")).casefold() for chunk in chunks[:limit])


def category_distribution(chunks: list[dict]) -> Counter:
    return Counter(str(item.get("metadata", {}).get("category", "uncategorized")) for item in chunks)


def source_distribution(chunks: list[dict]) -> Counter:
    return Counter(str(item.get("metadata", {}).get("source", "")) for item in chunks)


def summarize_chunks(label: str, chunks: list[dict]) -> dict:
    lengths = [len(item.get("text", "")) for item in chunks]
    return {
        "label": label,
        "chunk_count": len(chunks),
        "avg_chars": round(statistics.mean(lengths), 1) if lengths else 0,
        "median_chars": round(statistics.median(lengths), 1) if lengths else 0,
        "categories": dict(category_distribution(chunks)),
        "top_sources": source_distribution(chunks).most_common(12),
    }


def run():
    from src.pipeline.retriever import load_ai_core, retrieve as new_retrieve

    print("[COMPARE] Loading current embedding/reranker models...")
    embedder, reranker, _ = load_ai_core()

    print("[COMPARE] Building old chunks from the same cleaned/pdf data...")
    old_chunks = build_old_chunks()
    new_chunks = load_new_chunks()

    print("[COMPARE] Vectorizing old chunks with old embedding settings...")
    old_collection = vectorize_old_chunks(old_chunks, embedder)

    rows = []
    for case in CASES:
        query = case["query"]

        start = time.perf_counter()
        old_results = old_retrieve(query, old_collection, embedder, reranker)
        old_ms = round((time.perf_counter() - start) * 1000, 1)

        start = time.perf_counter()
        new_results = new_retrieve(query, top_k=5)
        new_ms = round((time.perf_counter() - start) * 1000, 1)

        rows.append(
            {
                "id": case["id"],
                "query": query,
                "old_top_source": top_source(old_results),
                "new_top_source": top_source(new_results),
                "old_source_hit_top3": source_hit(old_results, case["expected_source"]),
                "new_source_hit_top3": source_hit(new_results, case["expected_source"]),
                "old_signal_hit_top3": contains_any(old_results, case["required_any"]),
                "new_signal_hit_top3": contains_any(new_results, case["required_any"]),
                "old_ms": old_ms,
                "new_ms": new_ms,
                "old_top_text": old_results[0]["text"][:260].replace("\n", " ") if old_results else "",
                "new_top_text": new_results[0]["text"][:260].replace("\n", " ") if new_results else "",
            }
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "old_chunks": summarize_chunks("old", old_chunks),
        "new_chunks": summarize_chunks("new", new_chunks),
        "cases": rows,
        "summary": {
            "old_source_hit_top3": sum(row["old_source_hit_top3"] for row in rows),
            "new_source_hit_top3": sum(row["new_source_hit_top3"] for row in rows),
            "old_signal_hit_top3": sum(row["old_signal_hit_top3"] for row in rows),
            "new_signal_hit_top3": sum(row["new_signal_hit_top3"] for row in rows),
            "case_count": len(rows),
            "old_avg_ms": round(statistics.mean(row["old_ms"] for row in rows), 1),
            "new_avg_ms": round(statistics.mean(row["new_ms"] for row in rows), 1),
        },
    }

    out_json = ROOT / "docs" / "rag_retrieval_comparison_2026-05-09.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[COMPARE] Report written to {out_json}")


if __name__ == "__main__":
    run()
