import pytest

from src.pipeline.scope import detect_scope_category, is_known_out_of_scope


GOLDEN_INTENTS = [
    ("staj icin gerekli evraklar neler?", "staj"),
    ("staj defteri nasil doldurulur?", "staj"),
    ("sorumlu egitim elamanini nereden bulabilirim?", "staj"),
    ("erasmus hibe sonucu aciklandi mi?", "erasmus"),
    ("erasmus dil sinavi ne zaman?", "erasmus"),
    ("okulun ikili anlasmali oldugu universiteler hangisidir?", "erasmus"),
    ("cap icin ortalama kac olmali?", "cift_anadal_yandal"),
    ("yan dal basvuru sartlari nelerdir?", "cift_anadal_yandal"),
    ("ogretim gorevlileri kimler?", "ogretim_gorevlileri"),
]


@pytest.mark.rag_quality
def test_golden_intent_classification_quality():
    for query, expected in GOLDEN_INTENTS:
        assert detect_scope_category(query) == expected


@pytest.mark.rag_quality
def test_out_of_scope_quality_gate():
    out_of_scope = [
        "final sinav tarihleri ne zaman?",
        "okul ucretimi nereye yatiracagim?",
        "universiteye nasil ulasabilirim?",
        "hangi bolumler var?",
    ]

    for query in out_of_scope:
        assert is_known_out_of_scope(query)
