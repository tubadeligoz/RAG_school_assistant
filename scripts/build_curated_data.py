import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(BASE_DIR, "data", "source_registry.yaml")
CURATED_DIR = os.path.join(BASE_DIR, "data", "curated")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

ACADEMIC_TITLE_RE = (
    r"Emeritus\s+Prof\.\s*Dr\.|Prof\.\s*Dr\.|Doç\.\s*Dr\.|"
    r"Dr\.\s*Öğr\.\s*Üyesi|Dr\.Öğr\.Üyesi|"
    r"Öğr\.\s*Gör\.\s*Dr\.|Öğr\.\s*Gör\.|"
    r"Arş\.\s*Gör\.\s*Dr\.|Arş\.Gör\.Dr\.|Arş\.Gör\.|Arş\.\s*Gör\."
)

EMAIL_RE = re.compile(r"[\w.+-]+@dogus\.edu\.tr", re.IGNORECASE)

FACULTY_MARKERS = (
    "Mühendislik ve Doğa Bilimleri Fakültesi",
    "Mühendislik Fakültesi",
    "Fen-Edebiyat Fakültesi",
    "Fen Edebiyat Fakültesi",
    "İktisadi ,İdari ve Sosyal Bilimler Fakültesi",
    "İktisadi, İdari ve Sosyal Bilimler Fakültesi",
    "İktisadi ve İdari Bilimler Fakültesi",
    "İktisadi Ve İdari Bilimler Fakültesi",
    "Hukuk Fakültesi",
    "Sanat ve Tasarım Fakültesi",
    "Sağlık Bilimleri Yüksekokulu",
    "Meslek Yüksekokulu",
    "Meslek Yüksek Okulu",
    "Lisansüstü Eğitim Enstitüsü",
)

FOOTER_MARKERS = (
    "Paylaş",
    "Sizin İçin",
    "Medya",
    "E-Hizmetler",
    "Yasal Zorunluluklar",
    "Dudullu Kampüsü",
    "Çengelköy Kampüsü",
    "Takip Et",
    "Yukarı Kaydır",
    "© Doğuş Üniversitesi",
)

TR_FOLD_MAP = str.maketrans({
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

PARTNER_COUNTRY_FOLDS = {
    "almanya",
    "belcika",
    "bulgaristan",
    "cek cumhuriyeti",
    "finlandiya",
    "fransa",
    "hollanda",
    "ispanya",
    "isvec",
    "italya",
    "kuzey makedonya",
    "letonya",
    "litvanya",
    "macaristan",
    "norvec",
    "polonya",
    "portekiz",
    "romanya",
    "slovakya",
    "slovenya",
    "yunanistan",
}

PARTNER_LINK_SKIP_SIGNALS = (
    "dogus",
    "basvuru",
    "iletisim",
    "kroki",
    "instagram",
    "twitter",
    "linkedin",
    "facebook",
    "youtube",
    "spotify",
    "erasmus+ anlasmali universiteler",
)


STATIC_QA = {
    "staj": [
        {
            "subtopic": "evraklar",
            "question": "Staj için gerekli evraklar nelerdir?",
            "answer": (
                "Staj başvurusu için temel belge Staj Başvuru ve Kabul Formu'dur (EK-2). "
                "Form staj yerine doldurtulup onaylatıldıktan sonra sorumlu öğretim elemanının onayına sunulur. "
                "Onaylanan form ve gerekli diğer belgeler, örneğin nüfus cüzdanı fotokopisi, staja başlama tarihinden "
                "en az 15 gün önce teslim edilir. Staja başlarken staj defteri/uygulamalı eğitim dosyası da hazırlanır. "
                "Çalışan öğrenciler için staj muafiyeti durumunda ayrıca muafiyet dilekçesi, iş yerinden onaylı hizmet "
                "belgesi, SGK işe giriş bildirgesi ve SGK hizmet dökümü istenir."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/staj-uygulamalari-usul-ve-esaslari.pdf",
            "source_note": "Staj Uygulamaları Usul ve Esasları Madde 9, 13, 14.",
        },
        {
            "subtopic": "belge-konumu",
            "question": "Staj belgelerine nereden ulaşabilirim?",
            "answer": (
                "Staj için gerekli tüm belgeler ilgili fakülte, yüksekokul veya meslek yüksekokulu internet sayfalarında "
                "bulunur. Staj defteri de ilgili birimin internet sayfasından çıktı alınarak kullanılabilir. Bu yüzden "
                "öğrencinin kendi bölümünün veya akademik biriminin Staj/Formlar sayfasını kontrol etmesi gerekir."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/staj-uygulamalari-usul-ve-esaslari.pdf",
            "source_note": "Staj Uygulamaları Usul ve Esasları Madde 13 ve Madde 14.",
        },
        {
            "subtopic": "teslim",
            "question": "Staj evraklarını kime teslim edeceğim?",
            "answer": (
                "Staj başvuru formu staj yeri tarafından onaylandıktan sonra sorumlu öğretim elemanının onayına sunulur. "
                "Staj sonunda eksiksiz doldurulan staj defteri, varsa staj raporu, günlük yoklama çizelgesi ve kapalı zarftaki "
                "stajyer değerlendirme formu ilgili sorumlu öğretim elemanına imza karşılığı teslim edilir."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/staj-uygulamalari-usul-ve-esaslari.pdf",
            "source_note": "Staj Uygulamaları Usul ve Esasları Madde 14 ve Madde 19.",
        },
        {
            "subtopic": "sorumlu-ogretim-elemani",
            "question": "Stajda sorumlu öğretim elemanı kimdir?",
            "answer": (
                "Yönergeye göre bölüm/program başkanları sorumlu öğretim elemanıdır. Bu kişi staj süreci hakkında "
                "bilgilendirme yapar, staj yerinin uygunluğunu değerlendirir, gerektiğinde staj yerleriyle iletişim kurar "
                "ve değerlendirme sonucunun sisteme işlenmesini sağlar."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/staj-uygulamalari-usul-ve-esaslari.pdf",
            "source_note": "Staj Uygulamaları Usul ve Esasları Madde 3.",
        },
        {
            "subtopic": "staj-defteri",
            "question": "Staj defteri nasıl doldurulur?",
            "answer": (
                "Staj defteri günü gününe doldurulur ve çalışma süresi içinde staj yeri staj sorumlusuna onaylatılır. "
                "Yazılar okunaklı olmalı, tahrifat ve karalama yapılmamalı, sadece staj yerinde yapılan işlerle ilgili bilgiler "
                "yazılmalıdır. Çizimler teknik kurallara uygun olmalı; staj yeri izin verirse proje, fotoğraf, broşür veya "
                "kılavuz gibi ekler deftere eklenebilir."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/staj-uygulamalari-usul-ve-esaslari.pdf",
            "source_note": "Staj Uygulamaları Usul ve Esasları Madde 16.",
        },
    ],
    "erasmus": [
        {
            "subtopic": "2026-basvuru",
            "question": "Erasmus 2026 başvuruları ne zaman?",
            "answer": (
                "Doğuş Üniversitesi'nin 13 Mart 2026 tarihli duyurusuna göre Erasmus+ öğrenci öğrenim ve staj hareketliliği "
                "2026 dönemi başvuruları 04.04.2026-17.04.2026 tarihleri arasında başvuru portalı üzerinden yapılır. "
                "2026 sözleşme yılı için en erken proje başlangıç tarihi 01.06.2026 olarak belirtilmiştir."
            ),
            "source_url": "https://www.dogus.edu.tr/duyurular/detay/erasmus-ogrenci-ogrenim-ve-staj-hareketliligi-2026-donemi-projesi-basvurulari",
            "source_note": "Resmi duyuru, 13 Mart 2026.",
        },
        {
            "subtopic": "basvuru",
            "question": "Erasmus başvurusu nasıl yapılır?",
            "answer": (
                "Erasmus öğrenim hareketliliği için başvuru tarihleri Erasmus Ofisi tarafından duyurulur. Öğrenci, ilan edilen "
                "başvuru tarihlerinde Doğuş Üniversitesi Öğrenci Bilgi Sistemine girerek online başvuru formunu eksiksiz doldurur. "
                "Başvuru koşullarını sağlayan öğrenciler Erasmus Dil Sınavına çağırılır. Tarihler için e-posta duyuruları, resmi web "
                "duyuruları ve Erasmus Ofisi Instagram hesabı takip edilmelidir."
            ),
            "source_url": "https://www.dogus.edu.tr/uluslararasi/erasmus-degisim-programi/erasmus-ogrenci-hareketliligi/ogrenim-hareketliligi/giden-ogrenci-hareketliligi",
            "source_note": "Giden Öğrenci Hareketliliği sayfası.",
        },
        {
            "subtopic": "kosullar",
            "question": "Erasmus öğrenim ve staj hareketliliği başvuru koşulları nelerdir?",
            "answer": (
                "Öğrenim hareketliliği için öğrenci hazırlık sınıfı hariç en az 1 yarıyılı tamamlamış olmalıdır. Ön lisans/lisans "
                "öğrencileri için genel not ortalaması en az 2,20; lisansüstü öğrencileri için en az 2,50 olmalıdır. Staj hareketliliğinde "
                "de ön lisans/lisans için en az 2,20, lisansüstü için en az 2,50 şartı bulunur; ayrıca staj yapılacak kurumdan kabul "
                "mektubu alınması ve Doğuş Üniversitesi Erasmus Yabancı Dil Sınavından en az 50 puan alınması gerekir."
            ),
            "source_url": "https://www.dogus.edu.tr/uluslararasi/erasmus-degisim-programi/erasmus-ogrenci-hareketliligi/staj-hareketliligi/giden-ogrenci-staj-hareketliligi",
            "source_note": "Giden Öğrenci Hareketliliği ve Giden Öğrenci Staj Hareketliliği sayfaları.",
        },
        {
            "subtopic": "staj-belgeleri",
            "question": "Erasmus staj hareketliliği için hangi belgeler gerekir?",
            "answer": (
                "Başvuru sırasında güncel öğrenci belgesi ve transkript online başvuru formuna eklenir. Hareketlilik öncesinde staj "
                "anlaşması, seyahat sağlık sigortası, kaza sigortası, mesuliyet sigortası, hibeli katılım için TEB Euro hesap bilgisi, "
                "garantörlük yazısı/vize süreci, hibe sözleşmesi ve çevrimiçi dil desteği (OLS) adımları tamamlanır."
            ),
            "source_url": "https://www.dogus.edu.tr/uluslararasi/erasmus-degisim-programi/erasmus-ogrenci-hareketliligi/staj-hareketliligi/giden-ogrenci-staj-hareketliligi",
            "source_note": "Giden Öğrenci Staj Hareketliliği sayfası.",
        },
        {
            "subtopic": "hibe",
            "question": "Erasmus hibeleri nasıl ödenir?",
            "answer": (
                "Erasmus hibesi yaşam masraflarına katkı niteliğindedir ve tüm masrafları karşılamayabilir. Hibeler iki taksitte ödenir: "
                "gerekli belgeler tamamlandıktan sonra hareketlilik öncesinde yüzde 80, hareketlilik sonrası belgeler ve çevrimiçi anket "
                "tamamlandıktan sonra kalan yüzde 20 ödenir. İkinci ödemenin hak edilmesi için alınan kredilerin üçte ikisinin başarıyla "
                "tamamlanması gerekir."
            ),
            "source_url": "https://www.dogus.edu.tr/uluslararasi/erasmus-degisim-programi/erasmus-ogrenci-hareketliligi/ogrenim-hareketliligi/giden-ogrenci-hareketliligi",
            "source_note": "Giden Öğrenci Hareketliliği sayfası.",
        },
        {
            "subtopic": "anlasmali-universiteler",
            "question": "Doğuş Üniversitesi'nin Erasmus ikili anlaşmalı olduğu üniversiteler hangileridir?",
            "answer": (
                "Doğuş Üniversitesi'nin Erasmus+ anlaşmalı üniversiteleri resmi Anlaşmalarımız sayfasında ülke ülke "
                "listelenir. Veritabanı bu sayfadaki kayıtları ayrı üniversite kayıtları olarak indeksler; ülke veya "
                "bölüm adı verilirse liste daha net daraltılabilir."
            ),
            "source_url": "https://www.dogus.edu.tr/uluslararasi-aday/uluslararasi/erasmus-degisim-programi/anlasmalarimiz",
            "source_note": "Erasmus+ Anlaşmalarımız sayfası.",
        },
    ],
    "cift_anadal_yandal": [
        {
            "subtopic": "cap-kosullari",
            "question": "Çift anadal başvuru şartları nelerdir?",
            "answer": (
                "Çift anadal başvuruları akademik takvimde ilan edilen tarihlerde öğrenci otomasyon sistemi üzerinden yapılır. "
                "Öğrenci, anadal lisans programında en erken 3. ve en geç 5. yarıyılın başında; ön lisans programında ise en erken "
                "2. ve en geç 3. yarıyılın başında başvurabilir. Başvuruya kadar yükümlü olduğu tüm dersleri başarıyla tamamlamış "
                "olması, ağırlıklı genel not ortalamasının en az 3,00 olması ve ilgili sınıfta başarı sıralamasında en üst yüzde 20'de "
                "bulunması gerekir."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/cift-anadal-ve-yandal-yonergesi.pdf",
            "source_note": "Çift Anadal ve Yandal Yönergesi Madde 6.",
        },
        {
            "subtopic": "yandal-kosullari",
            "question": "Yandal başvuru şartları nelerdir?",
            "answer": (
                "Yandal başvuruları akademik takvimde ilan edilen tarihlerde öğrenci otomasyon sistemi üzerinden yapılır. Öğrenci, "
                "kayıtlı olduğu lisans programının en erken 3. ve en geç 6. yarıyılının başında başvurabilir. Başvuru yarıyılına kadar "
                "kayıtlı olduğu lisans programındaki tüm kredili dersleri başarıyla tamamlamış olması ve başvuru sırasındaki ağırlıklı "
                "genel not ortalamasının en az 2,50 olması gerekir."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/cift-anadal-ve-yandal-yonergesi.pdf",
            "source_note": "Çift Anadal ve Yandal Yönergesi Madde 6.",
        },
        {
            "subtopic": "basvuru-tarihleri",
            "question": "ÇAP/Yandal başvuru tarihleri ve kayıt tarihleri nedir?",
            "answer": (
                "Doğuş Üniversitesi Kayıt & Kabul sayfasında 2026 dönemi için başvuru başlangıcı 28.01.2026, başvuru bitişi "
                "06.02.2026, sonuç ilan tarihi 18.02.2026, kesin kayıt başlangıcı 19.02.2026, kesin kayıt bitişi 20.02.2026 ve "
                "yedek kayıt bitişi 23.02.2026 olarak verilmiştir. Bu tarihlerin dönemsel olduğunu ve yeni dönemlerde sayfanın "
                "kontrol edilmesi gerektiğini ayrıca belirtmek gerekir."
            ),
            "source_url": "https://www.dogus.edu.tr/ogrenci/kayit-kabul/cift-anadal-yan-dal",
            "source_note": "Çift Anadal / Yan Dal Kayıt & Kabul sayfası.",
        },
        {
            "subtopic": "yandal-sertifikasi",
            "question": "Yandal sertifikası almak için ne gerekir?",
            "answer": (
                "Kayıtlı olduğu lisans programından mezuniyet hakkını elde eden ve yandal programında zorunlu tanımlanan dersleri "
                "başarıp en az 2,00 ağırlıklı genel not ortalaması ile tamamlayan öğrenciye yandal sertifikası verilir. Yandal programında "
                "ortak veya eşdeğer dersler hariç en az 30 AKTS ve en az 5 ders alınması gerekir. Yandal sertifikası lisans diplomasıyla "
                "verilen hak ve yetkileri sağlamaz."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/cift-anadal-ve-yandal-yonergesi.pdf",
            "source_note": "Çift Anadal ve Yandal Yönergesi Madde 9.",
        },
        {
            "subtopic": "devam-kayit-silme",
            "question": "ÇAP/Yandal devam şartları nelerdir?",
            "answer": (
                "Çift anadal programından mezun olabilmek için birinci anadal ağırlıklı genel not ortalamasının en az 2,72 olması gerekir. "
                "Çift anadal süresince genel not ortalaması bir defaya mahsus 2,50'ye kadar düşebilir; ikinci kez 2,72'nin altına düşerse "
                "ikinci anadal kaydı silinir. Yandal programına devam edebilmek için anadal not ortalamasının en az 2,29 olması gerekir."
            ),
            "source_url": "https://www.dogus.edu.tr/docs/default-source/default-document-library/cift-anadal-ve-yandal-yonergesi.pdf",
            "source_note": "Çift Anadal ve Yandal Yönergesi Madde 8 ve Madde 11.",
        },
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_registry() -> dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def compact_inline(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_lines(text: str) -> list[str]:
    raw_lines = [compact_inline(line) for line in str(text or "").splitlines()]
    lines = []
    previous = ""
    for line in raw_lines:
        if not line or line == previous:
            continue
        previous = line
        lines.append(line)
    return lines


def fold_tr(text: str) -> str:
    folded = str(text or "").translate(TR_FOLD_MAP).casefold()
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def trim_to_main_content(lines: list[str], title: str = "") -> list[str]:
    if not lines:
        return []

    start = 0
    title_folded = title.casefold().strip()
    if title_folded:
        matches = [
            index
            for index, line in enumerate(lines)
            if line.casefold().strip() == title_folded
        ]
        if matches:
            start = matches[-1]

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if any(lines[index].startswith(marker) or lines[index] == marker for marker in FOOTER_MARKERS):
            end = index
            break

    return lines[start:end]


def extract_page_lines(html: str, title: str, max_lines: int = 180) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    body = soup.find("main") or soup.find("article") or soup.body or soup
    lines = clean_lines(body.get_text("\n", strip=True))
    lines = trim_to_main_content(lines, title)

    return lines[:max_lines]


def fetch_html(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=HEADERS, timeout=35, allow_redirects=True)
    response.raise_for_status()
    return response.text, response.url


def soup_title(soup: BeautifulSoup, fallback: str = "") -> str:
    h1 = soup.find("h1")
    if h1:
        title = compact_inline(h1.get_text(" ", strip=True))
        if title:
            return title

    if soup.title:
        title = compact_inline(soup.title.get_text(" ", strip=True))
        if title:
            return title

    return fallback


def extract_page_text(html: str, title: str) -> str:
    return "\n".join(extract_page_lines(html, title, max_lines=180))


def local_processed_text(source: dict) -> str:
    local_path = source.get("local_processed_json")
    if not local_path:
        return ""

    path = os.path.join(BASE_DIR, local_path)
    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return str(data.get("content", "")).strip()


def policy_excerpt(text: str, category: str, max_chars: int = 4800) -> str:
    if not text:
        return ""

    folded = text.casefold()
    if category == "staj":
        markers = ("sorumlu öğretim elemanı", "staj belgeleri", "staj öncesi işlemler", "uygulamalı eğitim dosyası")
    elif category == "erasmus":
        markers = ("erasmus programı", "başvuru şekli", "öğrenci hareketliliği", "hibe")
    else:
        markers = ("başvuru ve kabul koşulları", "çift anadal", "yandal sertifikası", "kayıt silme")

    windows = []
    for marker in markers:
        index = folded.find(marker)
        if index == -1:
            continue
        start = max(0, index - 350)
        end = min(len(text), index + 1300)
        windows.append(text[start:end].strip())

    excerpt = "\n\n".join(windows) if windows else text[:max_chars]
    return excerpt[:max_chars].strip()


def base_record(category: str, source: dict, record_type: str) -> dict:
    return {
        "category": category,
        "record_type": record_type,
        "authority": source.get("authority", ""),
        "source_url": source.get("url", ""),
        "crawled_at": now_iso(),
    }


def make_page_summary_record(category: str, source: dict) -> list[dict]:
    try:
        html, final_url = fetch_html(source["url"])
    except Exception as exc:
        print(f"[CURATED] Fetch failed: {source['url']} -> {exc}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    title = source.get("title") or soup_title(soup, source["url"])
    text = extract_page_text(html, title)
    if len(text.split()) < 25:
        return []

    return [
        {
            **base_record(category, {**source, "url": final_url}, "source_page"),
            "record_id": f"{category}_page_{slugify(title)}",
            "title": title,
            "subtopic": source.get("department") or title,
            "faculty": source.get("faculty", ""),
            "department": source.get("department", ""),
            "summary": f"{title} resmi kaynak sayfasından temizlenmiş bilgi.",
            "text": text[:4200],
        }
    ]


def _partner_name(text: str) -> str:
    name = compact_inline(text)
    name = name.lstrip("-–— ").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _is_partner_country_heading(text: str) -> bool:
    return fold_tr(text).strip(" :") in PARTNER_COUNTRY_FOLDS


def _valid_partner_name(name: str) -> bool:
    folded = fold_tr(name)
    if len(name) < 4 or len(name.split()) < 2:
        return False
    if any(signal in folded for signal in PARTNER_LINK_SKIP_SIGNALS):
        return False
    return any(
        signal in folded
        for signal in (
            "univers",
            "univerz",
            "hochschule",
            "institut",
            "academy",
            "ecole",
            "school",
            "politecnico",
            "politechnika",
            "college",
            "technical",
            "technicka",
            "saxion",
            "thomas more",
        )
    )


def _partner_university_records(
    soup: BeautifulSoup,
    source: dict,
    source_url: str,
    category: str,
) -> list[dict]:
    body = soup.find("main") or soup.find("article") or soup.body or soup
    current_country = ""
    started = False
    records = []
    seen = set()

    for tag in body.find_all(["h2", "h3", "a"]):
        if tag.name in {"h2", "h3"}:
            heading = compact_inline(tag.get_text(" ", strip=True))
            if _is_partner_country_heading(heading):
                current_country = heading
                started = True
                continue

            if started and fold_tr(heading) in {"iyiligin dogusu", "kampusler", "iletisim"}:
                break

        if tag.name != "a" or not started or not current_country:
            continue

        name = _partner_name(tag.get_text(" ", strip=True))
        if not _valid_partner_name(name):
            continue

        key = (fold_tr(current_country), fold_tr(name))
        if key in seen:
            continue
        seen.add(key)

        partner_url = urljoin(source_url, tag.get("href", "")) if tag.get("href") else ""
        records.append(
            {
                **base_record(category, {**source, "url": source_url}, "erasmus_partner_university"),
                "record_id": f"{category}_partner_{slugify(current_country)}_{slugify(name)}",
                "title": f"{name} - Erasmus Anlasmali Universite",
                "subtopic": "erasmus_anlasmali_universiteler",
                "country": current_country,
                "partner_university": name,
                "partner_url": partner_url,
                "agreement_scope": "Erasmus+",
                "summary": "Dogus Universitesi Erasmus+ anlasmali universite kaydi.",
                "text": (
                    f"{name}, Dogus Universitesi Erasmus+ anlasmali universiteler listesinde "
                    f"{current_country} ulkesi altinda yer alir."
                ),
            }
        )

    return records


def _partner_overview_record(category: str, source: dict, source_url: str, records: list[dict]) -> dict | None:
    if not records:
        return None

    grouped: dict[str, list[str]] = {}
    for record in records:
        country = record.get("country", "")
        grouped.setdefault(country, []).append(record.get("partner_university", ""))

    lines = [
        "Dogus Universitesi Erasmus+ anlasmali universiteler listesi resmi Anlasmalarimiz sayfasindan derlenmistir.",
        f"Toplam {len(grouped)} ulke ve {len(records)} universite kaydi bulunur.",
    ]
    for country, names in grouped.items():
        clean_names = [name for name in names if name]
        lines.append(f"{country}: {', '.join(clean_names)}")

    return {
        **base_record(category, {**source, "url": source_url}, "erasmus_partner_overview"),
        "record_id": f"{category}_partner_overview_{slugify(source.get('title', 'anlasmalarimiz'))}",
        "title": source.get("title", "Erasmus Anlasmali Universiteler"),
        "subtopic": "erasmus_anlasmali_universiteler",
        "agreement_scope": "Erasmus+",
        "country_count": len(grouped),
        "partner_count": len(records),
        "summary": "Erasmus+ anlasmali universitelerin ulke bazli ozeti.",
        "text": "\n".join(lines),
    }


def make_partner_university_records(category: str, source: dict) -> list[dict]:
    try:
        html, final_url = fetch_html(source["url"])
    except Exception as exc:
        print(f"[CURATED] Fetch failed: {source['url']} -> {exc}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    records = _partner_university_records(soup, source, final_url, category)
    overview = _partner_overview_record(category, source, final_url, records)
    return ([overview] if overview else []) + records


def _general_bilateral_university_records(lines: list[str], category: str, source: dict, source_url: str) -> list[dict]:
    records = []
    seen = set()
    in_general_list = False

    for line in lines:
        folded = fold_tr(line)
        if "universitemizin ikili anlasma" in folded and "asagida" in folded:
            in_general_list = True
            continue
        if in_general_list and "erasmus" in folded and "anlas" in folded:
            break
        if not in_general_list:
            continue

        name = _partner_name(line)
        if not _valid_partner_name(name):
            continue

        key = fold_tr(name)
        if key in seen:
            continue
        seen.add(key)

        records.append(
            {
                **base_record(category, {**source, "url": source_url}, "bilateral_partner_university"),
                "record_id": f"{category}_bilateral_partner_{slugify(name)}",
                "title": f"{name} - Ikili Isbirligi Anlasmasi",
                "subtopic": "ikili_isbirligi_anlasmalari",
                "partner_university": name,
                "agreement_scope": "Ikili Isbirligi",
                "summary": "Dogus Universitesi ikili isbirligi anlasmasi olan universite kaydi.",
                "text": f"{name}, Dogus Universitesi ikili isbirligi anlasmasi listesinde yer alir.",
            }
        )

    return records


def make_agreement_records(category: str, source: dict) -> list[dict]:
    try:
        html, final_url = fetch_html(source["url"])
    except Exception as exc:
        print(f"[CURATED] Fetch failed: {source['url']} -> {exc}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    title = source.get("title") or soup_title(soup, source["url"])
    lines = extract_page_lines(html, title, max_lines=260)
    section_start = 0
    for index, line in enumerate(lines):
        folded = fold_tr(line)
        if "ikili anlasmalar" in folded or ("erasmus" in folded and "anlas" in folded):
            section_start = index
            break

    section_lines = lines[section_start:260]
    text = "\n".join(section_lines)
    records = _general_bilateral_university_records(section_lines, category, source, final_url)
    if len(text.split()) >= 25:
        records.insert(
            0,
            {
                **base_record(category, {**source, "url": final_url}, "agreement_overview"),
                "record_id": f"{category}_agreement_overview_{slugify(title)}",
                "title": title,
                "subtopic": "ikili_anlasmalar",
                "agreement_scope": "Erasmus+ ve Ikili Isbirligi",
                "summary": "Akademik Isbirlikleri sayfasindaki ikili anlasma ve Erasmus+ anlasma bilgileri.",
                "text": text[:12000],
            },
        )

    return records


def make_local_pdf_record(category: str, source: dict) -> list[dict]:
    text = local_processed_text(source)
    excerpt = policy_excerpt(text, category)
    if len(excerpt.split()) < 25:
        return []

    title = source.get("title", "Resmi yönerge")
    return [
        {
            **base_record(category, source, "source_page"),
            "record_id": f"{category}_policy_{slugify(title)}",
            "title": title,
            "subtopic": "resmi_yonerge",
            "summary": f"{title} içinden RAG için seçilmiş güvenilir yönerge bölümleri.",
            "text": excerpt,
        }
    ]


def normalize_title(title: str) -> str:
    title = compact_inline(title)
    title = re.sub(r"Dr\.Öğr\.Üyesi", "Dr. Öğr. Üyesi", title)
    title = re.sub(r"Arş\.Gör\.Dr\.", "Arş. Gör. Dr.", title)
    title = re.sub(r"Arş\.Gör\.", "Arş. Gör.", title)
    title = re.sub(r"Öğr\.Gör\.", "Öğr. Gör.", title)
    return title


def split_faculty_department(rest: str, fallback_faculty: str = "", fallback_department: str = "") -> tuple[str, str, str]:
    rest = compact_inline(rest).replace(" Remove", "").strip()
    best_marker = ""
    best_index = -1
    for marker in FACULTY_MARKERS:
        index = rest.find(marker)
        if index != -1 and (best_index == -1 or index < best_index):
            best_marker = marker
            best_index = index

    if best_index == -1:
        return "", fallback_faculty, fallback_department

    role = rest[:best_index].strip(" -/")
    faculty = best_marker
    department = rest[best_index + len(best_marker):].strip(" -/")
    return role, faculty or fallback_faculty, department or fallback_department


def parse_staff_entry(text: str, source: dict, source_url: str, category: str) -> dict | None:
    text = normalize_title(text)
    if not EMAIL_RE.search(text):
        return None

    match = re.match(
        rf"(?P<title>{ACADEMIC_TITLE_RE})\s+(?P<name>.+?)\s+"
        rf"(?P<email>{EMAIL_RE.pattern})\s*(?P<rest>.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    academic_title = normalize_title(match.group("title"))
    name = compact_inline(match.group("name"))
    email = match.group("email").lower()
    rest = compact_inline(match.group("rest"))
    role, faculty, department = split_faculty_department(
        rest,
        fallback_faculty=source.get("faculty", ""),
        fallback_department=source.get("department", ""),
    )

    if not name or len(name.split()) < 2:
        return None

    title = f"{name} - {source.get('title', 'Akademik Kadro')}"
    return {
        **base_record(category, {**source, "url": source_url}, "academic_staff"),
        "record_id": f"staff_{slugify(email or name)}_{slugify(source.get('title', ''))}",
        "title": title,
        "subtopic": "akademik_kadro",
        "academic_title": academic_title,
        "name": name,
        "email": email,
        "role": role,
        "faculty": faculty or source.get("faculty", ""),
        "department": department or source.get("department", ""),
        "source_page_title": source.get("title", ""),
    }


def decode_cfemail(encoded: str) -> str:
    try:
        key = int(encoded[:2], 16)
        return "".join(
            chr(int(encoded[index:index + 2], 16) ^ key)
            for index in range(2, len(encoded), 2)
        )
    except Exception:
        return ""


def parse_staff_card(card, source: dict, source_url: str, category: str) -> dict | None:
    title_node = card.select_one(".card-title")
    if not title_node:
        return None

    span = title_node.find("span")
    academic_title = normalize_title(span.get_text(" ", strip=True) if span else "")
    full_title_text = compact_inline(title_node.get_text(" ", strip=True))
    name = full_title_text
    if academic_title and name.startswith(academic_title):
        name = name[len(academic_title):].strip()

    cf_email = card.select_one(".__cf_email__")
    email = ""
    if cf_email and cf_email.get("data-cfemail"):
        email = decode_cfemail(cf_email.get("data-cfemail", "")).lower()
    if not email:
        email_match = EMAIL_RE.search(card.get_text(" ", strip=True))
        email = email_match.group(0).lower() if email_match else ""

    role_node = card.select_one(".card-body p")
    role = compact_inline(role_node.get_text(" ", strip=True) if role_node else "")

    unit_node = card.select_one(".list-group-item")
    unit_lines = clean_lines(unit_node.get_text("\n", strip=True) if unit_node else "")
    faculty = source.get("faculty", "")
    department = source.get("department", "")
    if unit_lines:
        faculty = unit_lines[0] or faculty
    if len(unit_lines) > 1:
        department = unit_lines[1] or department

    link = card.find("a", href=True)
    profile_url = urljoin(source_url, link["href"]) if link else ""

    if not name or len(name.split()) < 2:
        return None

    title = f"{name} - {source.get('title', 'Akademik Kadro')}"
    return {
        **base_record(category, {**source, "url": source_url}, "academic_staff"),
        "record_id": f"staff_{slugify(email or name)}_{slugify(source.get('title', ''))}",
        "title": title,
        "subtopic": "akademik_kadro",
        "academic_title": academic_title,
        "name": name,
        "email": email,
        "role": role,
        "faculty": faculty,
        "department": department,
        "profile_url": profile_url,
        "source_page_title": source.get("title", ""),
    }


def staff_card_records(soup: BeautifulSoup, source: dict, source_url: str, category: str) -> list[dict]:
    records = []
    for card in soup.select(".academician"):
        record = parse_staff_card(card, source, source_url, category)
        if record:
            records.append(record)
    return records


def staff_entry_texts(soup: BeautifulSoup) -> list[str]:
    entries = []
    seen = set()

    for anchor in soup.find_all("a"):
        text = compact_inline(anchor.get_text(" ", strip=True))
        if "@dogus.edu.tr" not in text.lower():
            continue
        if not re.search(ACADEMIC_TITLE_RE, text, flags=re.IGNORECASE):
            continue
        if text not in seen:
            seen.add(text)
            entries.append(text)

    if entries:
        return entries

    text = soup.get_text("\n", strip=True)
    for line in clean_lines(text):
        if "@dogus.edu.tr" in line.lower() and re.search(ACADEMIC_TITLE_RE, line, flags=re.IGNORECASE):
            if line not in seen:
                seen.add(line)
                entries.append(line)

    return entries


def make_academic_staff_records(category: str, source: dict) -> list[dict]:
    try:
        html, final_url = fetch_html(source["url"])
    except Exception as exc:
        print(f"[CURATED] Fetch failed: {source['url']} -> {exc}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    records = staff_card_records(soup, source, final_url, category)
    if records:
        return records

    records = []
    for entry_text in staff_entry_texts(soup):
        record = parse_staff_entry(entry_text, source, final_url, category)
        if record:
            records.append(record)

    return records


def make_static_qa_records(category: str, output_title: str) -> list[dict]:
    records = []
    for index, item in enumerate(STATIC_QA.get(category, []), start=1):
        source = {
            "url": item.get("source_url", ""),
            "authority": "curated_from_official_source",
        }
        records.append(
            {
                **base_record(category, source, "qa"),
                "record_id": f"{category}_qa_{index}_{slugify(item['subtopic'])}",
                "title": output_title,
                **item,
            }
        )
    return records


def slugify(text: str) -> str:
    text = compact_inline(text).casefold()
    char_map = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    text = text.translate(char_map)
    text = re.sub(r"[^0-9a-z]+", "-", text).strip("-")
    return text[:80] or "record"


def dedupe_records(records: list[dict]) -> list[dict]:
    unique = {}
    order = []
    for record in records:
        if record.get("record_type") == "academic_staff":
            key = (
                "staff",
                record.get("email", "").casefold(),
                record.get("source_page_title", "").casefold(),
                record.get("department", "").casefold(),
            )
        else:
            key = (
                record.get("record_type", ""),
                record.get("category", ""),
                record.get("record_id", ""),
            )

        if key not in unique:
            order.append(key)
        unique[key] = record

    return [unique[key] for key in order]


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_category(category: str, config: dict) -> int:
    records = []
    label = config.get("label", category)
    records.extend(make_static_qa_records(category, label))

    for source in config.get("sources", []):
        parser = source.get("parser")
        if parser == "academic_staff":
            records.extend(make_academic_staff_records(category, source))
        elif parser == "partner_university_list":
            records.extend(make_partner_university_records(category, source))
        elif parser == "agreement_list":
            records.extend(make_agreement_records(category, source))
        elif parser == "page_summary":
            records.extend(make_page_summary_record(category, source))
        elif parser == "local_pdf_summary":
            records.extend(make_local_pdf_record(category, source))
        else:
            print(f"[CURATED] Unknown parser {parser!r}: {source.get('url')}")

    records = dedupe_records(records)
    output = os.path.join(CURATED_DIR, config.get("output", f"{category}.jsonl"))
    write_jsonl(output, records)
    print(f"[CURATED] {category}: {len(records)} records -> {output}")
    return len(records)


def main() -> int:
    registry = load_registry()
    categories = registry.get("categories", {})

    total = 0
    for category, config in categories.items():
        total += build_category(category, config or {})

    print(f"[CURATED] Ready. Total records: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
