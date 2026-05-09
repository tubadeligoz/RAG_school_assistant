import json

import pytest

from src.pipeline.curated_loader import load_curated_chunks


@pytest.mark.unit
def test_curated_loader_builds_academic_staff_chunk(tmp_path):
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    record = {
        "category": "ogretim_gorevlileri",
        "record_type": "academic_staff",
        "name": "Mitat UYSAL",
        "academic_title": "Prof. Dr.",
        "role": "Bölüm Başkanı",
        "faculty": "Mühendislik ve Doğa Bilimleri Fakültesi",
        "department": "Yazılım Mühendisliği",
        "email": "muysal@dogus.edu.tr",
        "source_url": "https://muhendislik.dogus.edu.tr/bolumler/yazilim-muhendisligi/akademik-kadro",
        "authority": "official_department_staff_page",
    }
    (curated_dir / "academic_staff.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    chunks = load_curated_chunks(str(curated_dir))

    assert len(chunks) == 1
    assert "Mitat UYSAL" in chunks[0]["text"]
    assert chunks[0]["metadata"]["record_type"] == "academic_staff"
    assert chunks[0]["metadata"]["department"] == "Yazılım Mühendisliği"


@pytest.mark.unit
def test_curated_loader_builds_erasmus_partner_chunk(tmp_path):
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    record = {
        "category": "erasmus",
        "record_type": "erasmus_partner_university",
        "partner_university": "University of West Bohemia",
        "country": "Çek Cumhuriyeti",
        "agreement_scope": "Erasmus+",
        "source_url": "https://www.dogus.edu.tr/uluslararasi-aday/uluslararasi/erasmus-degisim-programi/anlasmalarimiz",
        "authority": "official_erasmus_agreements_page",
    }
    (curated_dir / "erasmus_faq.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    chunks = load_curated_chunks(str(curated_dir))

    assert len(chunks) == 1
    assert "University of West Bohemia" in chunks[0]["text"]
    assert chunks[0]["metadata"]["record_type"] == "erasmus_partner_university"
    assert chunks[0]["metadata"]["country"] == "Çek Cumhuriyeti"
