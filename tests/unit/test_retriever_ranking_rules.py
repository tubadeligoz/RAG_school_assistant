from src.pipeline.retriever import _apply_relevance_filter, _score_candidate


def make_ranked_item(score, source, text, category="cift_anadal_yandal"):
    return (
        score,
        text,
        {
            "category": category,
            "source": source,
            "title": source.replace(".json", ""),
            "type": "pdf",
            "url": "",
        },
    )


def test_cift_conditions_filter_prefers_policy_source_over_calendar():
    ranked = [
        make_ranked_item(
            2.0,
            "2025_2026_onlisans-lisans-akademik-takvimi.json",
            "Cift Anadal ve Yandal Basvuru Tarihleri 10-13 Subat 2026",
        ),
        make_ranked_item(
            1.0,
            "cift-anadal-ve-yandal-yonergesi.json",
            "Basvuru ve Kabul Kosullari: agirlikli genel not ortalamasinin en az 3,00 olmasi gerekir.",
        ),
    ]

    filtered = _apply_relevance_filter(
        ranked,
        "cift_anadal_yandal",
        query="cap icin ortalama kac olmali?",
    )

    assert filtered
    assert all("cift-anadal-ve-yandal" in item[2]["source"] for item in filtered)


def test_erasmus_scoring_penalizes_unrelated_policy_mentions():
    query = "erasmus basvurulari hakkinda bilgi verir misin?"
    noisy_score = _score_candidate(
        query,
        "Danisman ogrencileri Erasmus Farabi Mevlana olanaklari hakkinda bilgilendirir.",
        {
            "category": "erasmus",
            "source": "lisans-onlisans-ogrenci-danismanlik-yonergesi.json",
            "title": "Danismanlik Yonergesi",
            "type": "pdf",
            "url": "",
        },
        1.0,
        "erasmus",
    )
    official_score = _score_candidate(
        query,
        "Madde 6 - Erasmus Programi icin basvurular ve secim hibe sozlesmesi esaslarina gore yapilir.",
        {
            "category": "erasmus",
            "source": "uluslararasi-ogrenci-degisim-yonergesi.json",
            "title": "Uluslararasi Ogrenci Degisim Yonergesi",
            "type": "pdf",
            "url": "",
        },
        1.0,
        "erasmus",
    )

    assert official_score > noisy_score


def test_erasmus_agreement_query_prefers_partner_records():
    query = "okulun ikili anlasmali oldugu universiteler hangisidir?"
    partner_score = _score_candidate(
        query,
        "Anlasmali universite kaydi Kapsam: Erasmus+ Universite: University of West Bohemia Ulke: Cek Cumhuriyeti",
        {
            "category": "erasmus",
            "source": "erasmus_faq.jsonl",
            "title": "University of West Bohemia - Erasmus Anlasmali Universite",
            "type": "curated",
            "record_type": "erasmus_partner_university",
            "country": "Cek Cumhuriyeti",
            "partner_university": "University of West Bohemia",
            "url": "https://www.dogus.edu.tr/uluslararasi-aday/uluslararasi/erasmus-degisim-programi/anlasmalarimiz",
        },
        1.0,
        "erasmus",
    )
    generic_score = _score_candidate(
        query,
        "Erasmus basvuru ve hibe sureci icin basvuru kosullari aciklanir.",
        {
            "category": "erasmus",
            "source": "erasmus_faq.jsonl",
            "title": "Erasmus Basvuru",
            "type": "curated",
            "record_type": "qa",
            "url": "https://www.dogus.edu.tr/duyurular/detay/erasmus",
        },
        1.0,
        "erasmus",
    )

    assert partner_score > generic_score
