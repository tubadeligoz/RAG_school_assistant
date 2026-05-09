from src.pipeline.chunker import infer_category


def test_unrelated_pdf_erasmus_mention_is_not_indexed_as_erasmus():
    data = {
        "title": "Lisans Onlisans Ogrenci Danismanlik Yonergesi",
        "url": "",
    }
    chunk = "Danisman, ogrencileri Erasmus, Farabi ve Mevlana olanaklari hakkinda bilgilendirir."

    assert (
        infer_category(
            data,
            "lisans-onlisans-ogrenci-danismanlik-yonergesi.json",
            "pdf",
            chunk,
        )
        == "out_of_scope"
    )


def test_html_erasmus_announcement_is_indexed_as_erasmus():
    data = {
        "title": "Duyurular",
        "url": "https://www.dogus.edu.tr/",
    }
    chunk = "Erasmus+ 2026 Projesi Ogrenci Hareketliligi Dil Sinavi hakkinda bilgilendirme."

    assert infer_category(data, "www_dogus_edu_tr_clean.json", "html", chunk) == "erasmus"


def test_calendar_cift_yandal_dates_stay_in_scope():
    data = {
        "title": "2025 2026 Onlisans Lisans Akademik Takvimi",
        "url": "",
    }
    chunk = "Cift Anadal ve Yandal Basvuru Tarihleri 10-13 Subat 2026"

    assert (
        infer_category(
            data,
            "2025_2026_onlisans-lisans-akademik-takvimi.json",
            "pdf",
            chunk,
        )
        == "cift_anadal_yandal"
    )
