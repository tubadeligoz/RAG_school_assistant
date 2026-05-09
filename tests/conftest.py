import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_chunk(text: str, category: str = "staj", source: str = "test_source.json"):
    return {
        "text": text,
        "metadata": {
            "category": category,
            "source": source,
            "title": source.replace(".json", ""),
            "type": "test",
            "url": f"mock://{source}",
        },
        "score": 1.0,
    }


@pytest.fixture
def make_chunk_fixture():
    return make_chunk


@pytest.fixture(name="make_chunk")
def make_chunk_factory():
    return make_chunk


@pytest.fixture
def staj_reference_text():
    return """
    staj-uygulamalari-usul-ve-esaslari > staj
    Staj Belgeleri
    Madde 13- Staj icin gerekli tum belgeler, fakulte, yuksekokul, meslek yuksekokulu internet sayfalarinda mevcuttur.
    Staj Oncesi Islemler
    Ogrenci, sorumlu ogretim elemani ile on gorusme yaparak staj yapmaya karar verdigi staj yeri icin
    Staj Basvuru ve Kabul Formu (EK-2) doldurur.
    Formu staj yerine onaylattiktan sonra sorumlu ogretim elemaninin onayina sunar.
    Onaylanan bu form ve gerekli diger belgeler, nufus cuzdan fotokopisi vb., staja baslama tarihinden en az 15 gun once teslim edilir.
    Staj yerleri kesinlesen ogrenciler, staj defterini ilgili birimin internet sayfalarindan cikti almak suretiyle kullanabilir.
    Sorumlu Ogretim Elemani Gorev ve Yetkisi:
    Bolum/Program baskanlari, sorumlu ogretim elemanidir.
    Sorumlu ogretim elemani staj sureci hakkinda ogrencilere aciklayici bilgiler verir.
    Calisan Ogrenciler Icin Staj Muafiyet Dilekcesi
    Hizmet Belgesi
    """


@pytest.fixture
def staj_chunks(staj_reference_text):
    return [make_chunk(staj_reference_text, category="staj", source="staj-uygulamalari-usul-ve-esaslari.json")]


@pytest.fixture
def staj_history():
    return [
        {"role": "user", "content": "staj icin gerekli evraklar neler?"},
        {"role": "assistant", "content": "Staj icin Staj Basvuru ve Kabul Formu, nufus cuzdan fotokopisi ve staj defteri gerekir."},
        {"role": "user", "content": "kime teslim edecegim peki?"},
        {"role": "assistant", "content": "Kaynakta teslim/onay noktasi olarak sorumlu ogretim elemani geciyor."},
    ]
