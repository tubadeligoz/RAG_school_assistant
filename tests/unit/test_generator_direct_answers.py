import pytest

from src.pipeline.generator import condense_question, generate_answer_stream
from src.pipeline.scope import compact_text


def answer_text(chunks, question, history=None):
    return "".join(generate_answer_stream(chunks, question, history or []))


@pytest.mark.unit
def test_staj_document_answer_lists_required_documents(staj_chunks):
    answer = answer_text(staj_chunks, "staj icin gerekli evraklar neler?")
    folded = compact_text(answer)

    assert "staj basvuru ve kabul formu" in folded
    assert "nufus cuzdan" in folded
    assert "staj defteri" in folded
    assert "15 gun" in folded


@pytest.mark.unit
def test_staj_document_location_followup(staj_chunks):
    answer = answer_text(staj_chunks, "staj belgelerine nereden ulasabilirim?")
    folded = compact_text(answer)

    assert "internet sayfalarindan" in folded
    assert "staj defteri" in folded


@pytest.mark.unit
def test_staj_delivery_followup(staj_chunks):
    answer = answer_text(staj_chunks, "kime teslim edecegim peki?")
    folded = compact_text(answer)

    assert "sorumlu ogretim elemani" in folded
    assert "15 gun" in folded


@pytest.mark.unit
def test_staj_responsible_teacher_followup(staj_chunks):
    answer = answer_text(staj_chunks, "sorumlu egitim elamanini nereden bulabilirim?")
    folded = compact_text(answer)

    assert "bolum/program baskani" in folded
    assert "isim uydurmuyorum" in folded


@pytest.mark.unit
def test_contextual_followup_is_rewritten_with_active_topic(staj_history):
    rewritten = condense_question(staj_history, "peki bu belgelere nereden ulasabilirim?")
    assert rewritten.startswith("Staj:")


@pytest.mark.unit
def test_clear_scoped_question_is_not_sent_to_llm(staj_history, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for a scoped deterministic follow-up")

    monkeypatch.setattr("src.pipeline.generator._call_ollama_sync", fail_if_called)
    question = "sorumlu egitim elamanini nereden bulabilirim?"

    assert condense_question(staj_history, question) == question


@pytest.mark.unit
def test_ogretim_gorevlileri_does_not_turn_news_into_staff_list(make_chunk):
    chunks = [
        make_chunk(
            "Akademik Kadro Tüm Programlar. Prof. Dr. Barış Çoban konferansta davetli konuşmacı oldu. Devamını Oku",
            category="ogretim_gorevlileri",
            source="www_dogus_edu_tr_clean.json",
        )
    ]

    answer = answer_text(chunks, "ogretim gorevlileri kimler?")
    folded = compact_text(answer)

    assert "tam ve guvenilir bir ogretim gorevlisi listesi vermiyor" in folded
    assert "tum ogretim gorevlileri listesi gibi sunmam dogru olmaz" in folded


@pytest.mark.unit
def test_ogretim_gorevlileri_uses_curated_staff_records(make_chunk):
    chunk = make_chunk(
        "Akademik kadro kaydı\nAd: Mitat UYSAL\nUnvan: Prof. Dr.\nBölüm/Program: Yazılım Mühendisliği",
        category="ogretim_gorevlileri",
        source="academic_staff.jsonl",
    )
    chunk["metadata"].update(
        {
            "type": "curated",
            "record_type": "academic_staff",
            "name": "Mitat UYSAL",
            "academic_title": "Prof. Dr.",
            "role": "Bölüm Başkanı",
            "faculty": "Mühendislik ve Doğa Bilimleri Fakültesi",
            "department": "Yazılım Mühendisliği",
            "email": "muysal@dogus.edu.tr",
        }
    )

    answer = answer_text([chunk], "yazilim muhendisligi hocalari kimler?")
    folded = compact_text(answer)

    assert "resmi akademik kadro" in folded
    assert "mitat uysal" in folded
    assert "bolum baskani" in folded


@pytest.mark.unit
def test_erasmus_agreement_question_uses_partner_records(make_chunk):
    overview = make_chunk(
        "Toplam 20 ulke ve 90 universite kaydi bulunur.",
        category="erasmus",
        source="erasmus_faq.jsonl",
    )
    overview["metadata"].update(
        {
            "type": "curated",
            "record_type": "erasmus_partner_overview",
            "country_count": 20,
            "partner_count": 90,
            "agreement_scope": "Erasmus+",
        }
    )
    partner = make_chunk(
        "Anlasmali universite kaydi\nUniversite: University of West Bohemia\nUlke: Cek Cumhuriyeti",
        category="erasmus",
        source="erasmus_faq.jsonl",
    )
    partner["metadata"].update(
        {
            "type": "curated",
            "record_type": "erasmus_partner_university",
            "partner_university": "University of West Bohemia",
            "country": "Cek Cumhuriyeti",
            "agreement_scope": "Erasmus+",
        }
    )

    answer = answer_text([overview, partner], "okulun ikili anlasmali oldugu universiteler hangisidir?")
    folded = compact_text(answer)

    assert "resmi erasmus" in folded
    assert "20 ulke" in folded
    assert "university of west bohemia" in folded
