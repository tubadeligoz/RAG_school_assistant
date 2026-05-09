import re
import unicodedata


SUPPORTED_CATEGORIES = {
    "staj",
    "erasmus",
    "cift_anadal_yandal",
    "ogretim_gorevlileri",
}

CATEGORY_LABELS = {
    "staj": "Staj",
    "erasmus": "Erasmus ve uluslararası değişim",
    "cift_anadal_yandal": "Çift anadal ve yandal",
    "ogretim_gorevlileri": "Öğretim görevlileri ve akademik kadro",
}

CATEGORY_PROMPTS = {
    "staj": "answer_staj",
    "erasmus": "answer_erasmus",
    "cift_anadal_yandal": "answer_cift_anadal_yandal",
    "ogretim_gorevlileri": "answer_ogretim_gorevlileri",
}

CATEGORY_KEYWORDS = {
    "erasmus": (
        "erasmus",
        "exchange",
        "degisim",
        "ogrenci hareketliligi",
        "personel hareketliligi",
        "ogrenim hareketliligi",
        "staj hareketliligi",
        "uluslararasi degisim",
        "hibe",
        "dil sinavi",
        "ikili anlasma",
        "ikili anlasmali",
        "anlasmali universite",
        "anlasmali universiteler",
        "anlasmalarimiz",
        "partner universite",
        "partner university",
        "partner kurum",
        "yurt disi universite",
        "yurtdisi universite",
        "uluslararasi isbirligi",
    ),
    "staj": (
        "staj",
        "staj defteri",
        "staj evraki",
        "staj belgesi",
        "sorumlu ogretim elemani",
        "sorumlu egitim elemani",
        "sorumlu egitim elamani",
        "staj sorumlusu",
        "kime teslim",
        "uygulamali egitim",
        "is yeri egitimi",
    ),
    "cift_anadal_yandal": (
        "cift anadal",
        "cap",
        "c.a.p",
        "yandal",
        "yan dal",
        "ikinci anadal",
    ),
    "ogretim_gorevlileri": (
        "ogretim gorevlisi",
        "ogretim gorevlileri",
        "ogretim elemani",
        "ogretim elemanlari",
        "ogretim uyesi",
        "ogretim uyeleri",
        "akademik kadro",
        "akademik personel",
        "akademisyen",
        "hoca",
        "hocalar",
        "prof dr",
        "doc dr",
        "dr ogretim",
        "arastirma gorevlisi",
    ),
}

OUT_OF_SCOPE_KEYWORDS = (
    "akademik takvim",
    "adres",
    "banka",
    "bolum",
    "butunleme",
    "eposta",
    "fakulte",
    "final",
    "harc",
    "iletisim",
    "iban",
    "kampus",
    "kayit",
    "kvkk",
    "mezuniyet",
    "odeme",
    "program",
    "telefon",
    "ulas",
    "ulasim",
    "ucret",
    "vize",
)

TR_MAP = str.maketrans({
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


def fold_text(text: str) -> str:
    folded = str(text or "").translate(TR_MAP).casefold()
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", fold_text(text)).strip()


def normalize_category(category: str | None) -> str:
    folded = compact_text(category or "")

    aliases = {
        "cift anadal yandal": "cift_anadal_yandal",
        "cift_anadal_yandal": "cift_anadal_yandal",
        "cap_yandal": "cift_anadal_yandal",
        "ogretim gorevlileri": "ogretim_gorevlileri",
        "ogretim_gorevlileri": "ogretim_gorevlileri",
    }

    return aliases.get(folded, folded.replace(" ", "_"))


def is_supported_category(category: str | None) -> bool:
    return normalize_category(category) in SUPPORTED_CATEGORIES


def detect_scope_category(text: str) -> str | None:
    folded = compact_text(text)
    if not folded:
        return None

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in folded for keyword in keywords):
            return category

    return None


def detect_scope_category_from_history(history) -> str | None:
    for message in reversed(history or []):
        if message.get("role") != "user":
            continue

        category = detect_scope_category(str(message.get("content", "")))
        if category:
            return category

    for message in reversed(history or []):
        category = detect_scope_category(str(message.get("content", "")))
        if category:
            return category

    return None


def is_known_out_of_scope(text: str) -> bool:
    folded = compact_text(text)
    if detect_scope_category(folded):
        return False

    return any(keyword in folded for keyword in OUT_OF_SCOPE_KEYWORDS)


def supported_scope_text() -> str:
    return ", ".join(CATEGORY_LABELS[category] for category in (
        "staj",
        "erasmus",
        "cift_anadal_yandal",
        "ogretim_gorevlileri",
    ))


def scope_limit_message() -> str:
    return (
        "Bu asistan şu anda yalnızca "
        f"{supported_scope_text()} konularında cevap verecek şekilde sınırlandırıldı."
    )


def prompt_name_for_category(category: str | None) -> str:
    return CATEGORY_PROMPTS.get(normalize_category(category), "answer_default")
