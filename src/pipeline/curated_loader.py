import hashlib
import json
import os
from typing import Iterable

from src.pipeline.scope import is_supported_category, normalize_category


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CURATED_DIR = os.path.join(BASE_DIR, "data", "curated")


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    return [value] if value else []


def _compact(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _hash_record(record: dict) -> str:
    base = "|".join(
        str(record.get(key, ""))
        for key in (
            "category",
            "record_type",
            "source_url",
            "question",
            "name",
            "title",
            "partner_university",
            "country",
        )
    )
    if not base.strip("|"):
        base = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def iter_curated_records(curated_dir: str = CURATED_DIR) -> Iterable[tuple[str, int, dict]]:
    if not os.path.isdir(curated_dir):
        return

    for filename in sorted(os.listdir(curated_dir)):
        if not filename.endswith(".jsonl"):
            continue

        path = os.path.join(curated_dir, filename)
        with open(path, "r", encoding="utf-8") as file:
            for index, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield path, index, record


def render_curated_text(record: dict) -> str:
    record_type = record.get("record_type", "")
    category = normalize_category(record.get("category"))
    title = str(record.get("title", "")).strip()
    subtopic = str(record.get("subtopic", "")).strip()
    source_url = str(record.get("source_url", "")).strip()

    if record_type == "academic_staff":
        parts = [
            "Akademik kadro kaydı",
            f"Ad: {record.get('name', '')}",
            f"Unvan: {record.get('academic_title', '')}",
        ]
        if record.get("role"):
            parts.append(f"Görev: {record.get('role')}")
        if record.get("faculty"):
            parts.append(f"Fakülte/Yüksekokul: {record.get('faculty')}")
        if record.get("department"):
            parts.append(f"Bölüm/Program: {record.get('department')}")
        if record.get("email"):
            parts.append(f"E-posta: {record.get('email')}")
        if record.get("profile_url"):
            parts.append(f"Profil: {record.get('profile_url')}")
        if source_url:
            parts.append(f"Kaynak: {source_url}")
        return _compact("\n".join(parts))

    if record_type in {"erasmus_partner_university", "bilateral_partner_university"}:
        parts = [
            "Anlaşmalı üniversite kaydı",
            f"Kapsam: {record.get('agreement_scope', '')}",
            f"Üniversite: {record.get('partner_university', '')}",
        ]
        if record.get("country"):
            parts.append(f"Ülke: {record.get('country')}")
        if record.get("department") or record.get("programs"):
            parts.append(f"Bölüm/Program: {record.get('department') or record.get('programs')}")
        if record.get("partner_url"):
            parts.append(f"Üniversite bağlantısı: {record.get('partner_url')}")
        if record.get("text"):
            parts.append(str(record.get("text")))
        if source_url:
            parts.append(f"Kaynak: {source_url}")
        return _compact("\n".join(parts))

    if record_type in {"erasmus_partner_overview", "agreement_overview"}:
        parts = [
            record.get("title", "") or "Anlaşmalı üniversiteler",
            f"Kategori: {category}",
            f"Konu: {subtopic}" if subtopic else "",
            f"Kapsam: {record.get('agreement_scope', '')}" if record.get("agreement_scope") else "",
            f"Toplam ülke: {record.get('country_count')}" if record.get("country_count") else "",
            f"Toplam üniversite: {record.get('partner_count')}" if record.get("partner_count") else "",
            record.get("summary", ""),
            record.get("text", ""),
        ]
        if source_url:
            parts.append(f"Kaynak: {source_url}")
        return _compact("\n".join(str(part) for part in parts if part))

    if record_type == "qa":
        parts = [
            title or subtopic or category,
            f"Kategori: {category}",
            f"Konu: {subtopic}" if subtopic else "",
            f"Soru: {record.get('question', '')}",
            f"Cevap: {record.get('answer', '')}",
        ]
        if record.get("source_note"):
            parts.append(f"Kaynak notu: {record.get('source_note')}")
        if source_url:
            parts.append(f"Kaynak: {source_url}")
        return _compact("\n".join(parts))

    bullets = _as_list(record.get("key_points"))
    parts = [
        title or subtopic or category,
        f"Kategori: {category}",
        f"Konu: {subtopic}" if subtopic else "",
        record.get("summary", ""),
        "\n".join(f"- {item}" for item in bullets),
        record.get("text", ""),
    ]
    if source_url:
        parts.append(f"Kaynak: {source_url}")
    return _compact("\n".join(str(part) for part in parts if part))


def curated_record_to_chunk(record: dict, filename: str, index: int) -> dict | None:
    category = normalize_category(record.get("category"))
    if not is_supported_category(category):
        return None

    text = render_curated_text({**record, "category": category})
    if len(text.strip()) < 30:
        return None

    record_id = record.get("record_id") or _hash_record(record)
    source = os.path.basename(filename)
    title = record.get("title") or record.get("question") or record.get("name") or source

    metadata = {
        "source": source,
        "url": record.get("source_url", ""),
        "type": "curated",
        "category": category,
        "title": title,
        "record_type": record.get("record_type", ""),
        "subtopic": record.get("subtopic", ""),
        "faculty": record.get("faculty", ""),
        "department": record.get("department", ""),
        "name": record.get("name", ""),
        "academic_title": record.get("academic_title", ""),
        "role": record.get("role", ""),
        "email": record.get("email", ""),
        "profile_url": record.get("profile_url", ""),
        "source_page_title": record.get("source_page_title", ""),
        "authority": record.get("authority", ""),
        "country": record.get("country", ""),
        "partner_university": record.get("partner_university", ""),
        "partner_url": record.get("partner_url", ""),
        "agreement_scope": record.get("agreement_scope", ""),
        "country_count": record.get("country_count", ""),
        "partner_count": record.get("partner_count", ""),
    }

    return {
        "chunk_id": f"curated_{source}_{index}_{record_id}",
        "text": text,
        "metadata": metadata,
    }


def load_curated_chunks(curated_dir: str = CURATED_DIR) -> list[dict]:
    chunks = []
    for path, index, record in iter_curated_records(curated_dir) or []:
        chunk = curated_record_to_chunk(record, path, index)
        if chunk:
            chunks.append(chunk)
    return chunks
