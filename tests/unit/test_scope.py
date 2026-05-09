import pytest

from src.pipeline.scope import (
    detect_scope_category,
    detect_scope_category_from_history,
    is_known_out_of_scope,
    normalize_category,
)


@pytest.mark.unit
def test_detects_supported_categories():
    cases = {
        "staj icin gerekli evraklar": "staj",
        "sorumlu egitim elamanini nereden bulurum": "staj",
        "erasmus basvurulari ne zaman": "erasmus",
        "okulun ikili anlasmali oldugu universiteler hangisidir": "erasmus",
        "anlasmali universiteler hangi ulkelerde": "erasmus",
        "CAP sartlari nelerdir": "cift_anadal_yandal",
        "yan dal icin ortalama kac olmali": "cift_anadal_yandal",
        "ogretim gorevlileri kimler": "ogretim_gorevlileri",
    }

    for query, expected in cases.items():
        assert detect_scope_category(query) == expected


@pytest.mark.unit
def test_known_out_of_scope_questions_are_rejected():
    assert is_known_out_of_scope("final sinav tarihleri ne zaman")
    assert is_known_out_of_scope("okul ucretimi nereye yatiracagim")
    assert is_known_out_of_scope("kampuse nasil ulasirim")


@pytest.mark.unit
def test_history_category_prefers_user_messages():
    history = [
        {"role": "user", "content": "staj icin gerekli evraklar neler?"},
        {"role": "assistant", "content": "Sorumlu ogretim elemani teslim noktasidir."},
    ]

    assert detect_scope_category_from_history(history) == "staj"


@pytest.mark.unit
def test_normalize_category_aliases():
    assert normalize_category("cift anadal yandal") == "cift_anadal_yandal"
    assert normalize_category("ogretim gorevlileri") == "ogretim_gorevlileri"
