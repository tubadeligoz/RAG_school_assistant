import os
import re
import unicodedata
from functools import lru_cache
from typing import Iterable

from src.pipeline.scope import (
    CATEGORY_KEYWORDS,
    SUPPORTED_CATEGORIES,
    detect_scope_category,
    is_known_out_of_scope,
    is_supported_category,
    normalize_category,
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")

EMBED_MODEL = "intfloat/multilingual-e5-base"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION_NAME = "uni_rag"

SOURCE_LABELS = {
    "staj-uygulamalari-usul-ve-esaslari": "Staj Uygulamaları Usul ve Esasları",
    "cift-anadal-ve-yandal": "Çift Anadal ve Yandal",
    "uluslararasi-ogrenci-degisim": "Uluslararası Öğrenci Değişimi",
    "erasmus": "Erasmus",
    "ogretim-gorevlileri": "Öğretim Görevlileri",
}

CATEGORIES = {
    "staj": "staj zorunlu staj basvurusu staj suresi evrak belge staj defteri sorumlu ogretim elemani teslim uygulamali egitim",
    "erasmus": "erasmus uluslararasi degisim ogrenci hareketliligi personel hareketliligi hibe dil sinavi basvuru ikili anlasma anlasmali universite partner university",
    "cift_anadal_yandal": "cift anadal yandal cap basvuru sartlari kontenjan not ortalamasi ikinci anadal",
    "ogretim_gorevlileri": "ogretim gorevlisi ogretim elemani ogretim uyesi akademik kadro akademisyen hoca arastirma gorevlisi",
}

INTENT_KEYWORDS = CATEGORY_KEYWORDS
STRICT_INTENTS = set(SUPPORTED_CATEGORIES)

MIN_RELEVANCE_SCORE = 0.35
DEFAULT_CANDIDATE_K = 24

TR_MAP = str.maketrans({
    "\u00e7": "c",
    "\u011f": "g",
    "\u0131": "i",
    "\u00f6": "o",
    "\u015f": "s",
    "\u00fc": "u",
    "\u00c7": "c",
    "\u011e": "g",
    "\u0130": "i",
    "I": "i",
    "\u00d6": "o",
    "\u015e": "s",
    "\u00dc": "u",
})

STOPWORDS = {
    "acaba",
    "ama",
    "bir",
    "bu",
    "buna",
    "bunu",
    "da",
    "de",
    "diye",
    "en",
    "gibi",
    "hangi",
    "icin",
    "ile",
    "kac",
    "mi",
    "mu",
    "nasil",
    "ne",
    "nedir",
    "nelerdir",
    "olan",
    "olarak",
    "ve",
    "veya",
    "ya",
    "var",
    "yok",
}


def _fold(text: str) -> str:
    folded = str(text or "").translate(TR_MAP).casefold()
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[0-9a-z]+", _fold(text))
        if len(token) > 1 and token not in STOPWORDS
    ]


def _norm_key(text: str, max_chars: int = 500) -> str:
    return " ".join(_fold(text[:max_chars]).split())


def _is_public_url(value: str) -> bool:
    value = str(value or "").strip().lower()
    return value.startswith(("http://", "https://", "mock://"))


def _looks_like_local_path(value: str) -> bool:
    value = str(value or "").strip()
    if not value:
        return False

    return (
        bool(re.match(r"^[a-zA-Z]:[\\/]", value))
        or value.startswith(("/", "\\", "file:"))
        or "\\data\\" in value.lower()
        or "/data/" in value.lower()
    )


def _humanize_source_label(value: str) -> str:
    value = os.path.basename(str(value or "").strip())
    value = re.sub(r"\.(json|pdf|html?|txt)$", "", value, flags=re.IGNORECASE)
    value = value.strip(" -_")

    folded_value = _fold(value)
    if folded_value in SOURCE_LABELS:
        return SOURCE_LABELS[folded_value]

    label = value.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label.title() if label else ""


def _source_label(metadata: dict) -> str:
    title = _humanize_source_label(metadata.get("title", ""))
    if title:
        return title

    source = _humanize_source_label(metadata.get("source", ""))
    if source:
        return source

    url = str(metadata.get("url", "") or "")
    if url and _looks_like_local_path(url):
        return _humanize_source_label(url)

    return ""


def format_source_reference(metadata: dict) -> str:
    url = str(metadata.get("url", "") or "").strip()
    label = _source_label(metadata)

    if url.lower().startswith("mock://"):
        return url

    if _is_public_url(url):
        return url if not label else f"{label} - {url}"

    if url and not _looks_like_local_path(url):
        return url if not label else f"{label} - {url}"

    return label


def _load_model(model_class, model_name: str, **kwargs):
    if os.getenv("DORA_LOCAL_MODELS_ONLY", "0") == "1":
        return model_class(model_name, local_files_only=True, **kwargs)

    try:
        return model_class(model_name, local_files_only=True, **kwargs)
    except Exception:
        return model_class(model_name, local_files_only=False, **kwargs)


@lru_cache(maxsize=1)
def load_ai_core():
    import torch
    import numpy as np
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DORA] Device: {device.upper()}")

    embedder = _load_model(SentenceTransformer, EMBED_MODEL, device=device)
    reranker = _load_model(CrossEncoder, RERANK_MODEL, device=device)

    category_embeddings = {
        key: embedder.encode(
            f"passage: {description}",
            normalize_embeddings=True,
        )
        for key, description in CATEGORIES.items()
    }

    return embedder, reranker, category_embeddings


@lru_cache(maxsize=1)
def load_chroma():
    import chromadb

    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def detect_intent_semantic(query: str, embedder, category_embeddings: dict):
    import numpy as np

    keyword_category = detect_scope_category(query)
    if keyword_category:
        return keyword_category

    if is_known_out_of_scope(query):
        return None

    tokens = set(_tokenize(query))
    if not tokens:
        return None

    keyword_scores = {
        category: len(tokens & set(_tokenize(description)))
        for category, description in CATEGORIES.items()
    }
    best_keyword_category, best_keyword_score = max(
        keyword_scores.items(),
        key=lambda item: item[1],
    )
    if best_keyword_score >= 2:
        return best_keyword_category

    query_embedding = embedder.encode(
        f"query: {query}",
        normalize_embeddings=True,
    )

    best_category = None
    best_score = -1.0

    for category, embedding in category_embeddings.items():
        score = float(np.dot(query_embedding, embedding))
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score >= 0.36 else None


def keyword_boost(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0

    text_tokens = set(_tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def _metadata_text(metadata: dict) -> str:
    return " ".join(
        str(metadata.get(key, ""))
        for key in (
            "title",
            "source",
            "category",
            "url",
            "faculty",
            "department",
            "source_page_title",
            "country",
            "partner_university",
            "agreement_scope",
        )
    )


def _has_any(text: str, signals: tuple[str, ...]) -> bool:
    folded = _fold(text)
    return any(signal in folded for signal in signals)


def _source_text(metadata: dict) -> str:
    return _fold(
        " ".join(
            str(metadata.get(key, ""))
            for key in ("source", "title", "url")
        )
    )


DATE_QUERY_SIGNALS = (
    "ne zaman",
    "tarih",
    "takvim",
    "ilan",
    "sonuc",
    "kayit",
)

CONDITION_QUERY_SIGNALS = (
    "sart",
    "kosul",
    "ortalama",
    "not",
    "agno",
    "kac",
    "gereki",
    "basvur",
    "kabul",
)

AGREEMENT_QUERY_SIGNALS = (
    "ikili anlas",
    "ikili isbirligi",
    "anlasmali universite",
    "anlasmali universiteler",
    "anlasmalarimiz",
    "partner universite",
    "partner university",
    "partner universities",
    "partner kurum",
    "yurt disi universite",
    "yurtdisi universite",
    "uluslararasi isbirligi",
)

AGREEMENT_RECORD_TYPES = {
    "erasmus_partner_university",
    "erasmus_partner_overview",
    "bilateral_partner_university",
    "agreement_overview",
    "agreement_university",
    "agreement",
}

ACADEMIC_UNIT_SIGNALS = (
    "agiz ve dis sagligi",
    "anestezi",
    "ascilik",
    "bilgisayar muhendisligi",
    "bilgisayar programciligi",
    "bilisim guvenligi teknolojisi",
    "cocuk gelisimi",
    "dis ticaret",
    "elektrik elektronik muhendisligi",
    "endustri muhendisligi",
    "fizyoterapi",
    "hukuk",
    "iktisat",
    "iletisim bilimleri",
    "ingiliz dili ve edebiyati",
    "ingilizce mutercim",
    "insaat muhendisligi",
    "insaat teknolojisi",
    "internet ag ve teknolojileri",
    "is sagligi ve guvenligi",
    "isletme",
    "makine muhendisligi",
    "mekatronik",
    "psikoloji",
    "siyaset bilimi",
    "sosyoloji",
    "turk dili ve edebiyati",
    "uluslararasi iliskiler",
    "uluslararasi ticaret",
    "yapay zeka operatorlugu",
    "yazilim muhendisligi",
    "yonetim bilisim sistemleri",
)


def _is_date_query(folded_query: str) -> bool:
    return any(signal in folded_query for signal in DATE_QUERY_SIGNALS)


def _is_condition_query(folded_query: str) -> bool:
    return any(signal in folded_query for signal in CONDITION_QUERY_SIGNALS)


def _is_agreement_query(text: str) -> bool:
    folded = _fold(text)
    return any(signal in folded for signal in AGREEMENT_QUERY_SIGNALS)


def _academic_unit_signal(folded_query: str) -> str:
    for signal in ACADEMIC_UNIT_SIGNALS:
        if signal in folded_query:
            return signal
    return ""


def _is_cift_yandal_policy_source(metadata: dict) -> bool:
    return "cift-anadal-ve-yandal" in _source_text(metadata)


def _is_calendar_source(metadata: dict) -> bool:
    source = _source_text(metadata)
    return "akademik-takvim" in source or "akademik-takvimi" in source


def _is_erasmus_source(metadata: dict, doc: str = "") -> bool:
    source = _source_text(metadata)
    folded_doc = _fold(doc[:1200])
    if _is_agreement_record(metadata, doc):
        return True
    if any(
        signal in source
        for signal in (
            "erasmus",
            "uluslararasi-ogrenci-degisim",
            "anlasmalarimiz",
            "akademik-isbirlikleri",
            "uluslararasi-ofis",
        )
    ):
        return True

    return "erasmus" in folded_doc and any(
        signal in folded_doc
        for signal in ("basvuru", "dil sinavi", "hareketliligi", "hibe", "sonuc")
    )


def _is_agreement_record(metadata: dict, doc: str = "") -> bool:
    record_type = str(metadata.get("record_type", ""))
    if record_type in AGREEMENT_RECORD_TYPES:
        return True

    folded = _fold(f"{doc[:1500]} {_metadata_text(metadata)}")
    return any(
        signal in folded
        for signal in (
            "anlasmali universite",
            "anlasmali universiteler",
            "ikili anlasma",
            "ikili isbirligi",
            "erasmus kodu",
            "erasmus code",
            "partner university",
        )
    )


def _score_candidate(
    query: str,
    doc: str,
    metadata: dict,
    rerank_score: float,
    detected_category: str | None,
) -> float:
    score = float(rerank_score)
    metadata_category = normalize_category(metadata.get("category"))

    if metadata.get("type") == "pdf":
        score += 0.08
    if metadata.get("type") == "curated":
        score += 0.18

    if detected_category and metadata_category == detected_category:
        score += 0.45
    elif detected_category in STRICT_INTENTS:
        score -= 0.30

    folded_doc = _fold(f"{doc[:1500]} {_metadata_text(metadata)}")
    if detected_category == "programlar":
        if "tum-programlar" in folded_doc or "tum programlar" in folded_doc or "program adi" in folded_doc:
            score += 0.70
        if "program adi" not in folded_doc and "tum-programlar" not in folded_doc and "tum programlar" not in folded_doc:
            score -= 0.35

    if detected_category == "ulasim_iletisim":
        if any(signal in folded_doc for signal in ("kroki", "nato yolu", "bosna blv", "444 79 97", "info@dogus")):
            score += 0.45
        if not any(signal in folded_doc for signal in ("adres", "kroki", "kampus", "yerleske", "nato yolu", "bosna blv")):
            score -= 0.35

    if detected_category == "akademik_takvim":
        folded_query = _fold(query)
        if "final" in folded_query and "final" in folded_doc and "sinav" in folded_doc:
            score += 0.55
        if "final" in folded_query and "final not" in folded_doc and "final sinav" not in folded_doc:
            score -= 0.20
        if "lisansustu" not in folded_query and "hazirlik" not in folded_query and "onlisans-lisans" in folded_doc:
            score += 0.18

    if detected_category == "ucret_odeme":
        if any(signal in folded_doc for signal in ("online odeme", "ucretler ve odeme kosullari", "odeme")):
            score += 0.50
        if "iade" in folded_doc and "online odeme" not in folded_doc:
            score -= 0.35

    if detected_category == "erasmus":
        agreement_query = _is_agreement_query(query)
        if any(signal in folded_doc for signal in ("erasmus", "degisim", "hareketliligi", "hibe", "dil sinavi")):
            score += 0.50
        if "uluslararasi ogrenci" in folded_doc and "erasmus" not in folded_doc and "degisim" not in folded_doc:
            score -= 0.35
        if _is_erasmus_source(metadata, doc):
            score += 0.30
        else:
            score -= 0.70

        folded_query = _fold(query)
        if "basvur" in folded_query and "basvuru" in folded_doc:
            score += 0.25
        if "dil sinavi" in folded_query and "dil sinavi" in folded_doc:
            score += 0.30
        if agreement_query:
            record_type = str(metadata.get("record_type", ""))
            if record_type == "erasmus_partner_overview":
                score += 1.55
            elif record_type == "erasmus_partner_university":
                score += 1.20
            elif record_type == "bilateral_partner_university":
                score += 0.45
            elif record_type == "agreement_overview":
                score += 0.70
            elif _is_agreement_record(metadata, doc):
                score += 1.10
            else:
                score -= 0.25
            if any(
                signal in folded_doc
                for signal in (
                    "anlasmali universite",
                    "anlasmali universiteler",
                    "ikili anlasma",
                    "ikili isbirligi",
                    "partner university",
                    "erasmus kodu",
                    "erasmus code",
                )
            ):
                score += 0.45

    if detected_category == "ogretim_gorevlileri":
        if any(signal in folded_doc for signal in ("ogretim elemani", "ogretim uyesi", "akademik kadro", "akademik personel", "prof", "doc")):
            score += 0.45
        if metadata.get("record_type") == "academic_staff":
            score += 0.75
        if any(signal in folded_doc for signal in ("haber", "duyuru", "devamini oku", "konferans")) and metadata.get("record_type") != "academic_staff":
            score -= 0.55
        unit_signal = _academic_unit_signal(_fold(query))
        if unit_signal:
            if unit_signal in folded_doc:
                score += 1.20
            else:
                score -= 0.80

    if detected_category == "cift_anadal_yandal":
        folded_query = _fold(query)
        date_query = _is_date_query(folded_query)
        condition_query = _is_condition_query(folded_query)
        policy_source = _is_cift_yandal_policy_source(metadata)
        calendar_source = _is_calendar_source(metadata)

        if policy_source:
            score += 0.35
        if calendar_source and not date_query:
            score -= 0.75
        if calendar_source and date_query:
            score += 0.30

        if condition_query:
            if policy_source:
                score += 0.35
            if any(
                signal in folded_doc
                for signal in (
                    "basvuru ve kabul kosullari",
                    "agirlikli genel not ortalamasi",
                    "not ortalamasinin en az",
                    "en az 3,00",
                    "en az 2.50",
                    "en az 2,50",
                    "en ust %20",
                )
            ):
                score += 0.65

            wants_cap = "cap" in folded_query or "cift anadal" in folded_query
            wants_yandal = "yandal" in folded_query or "yan dal" in folded_query
            if wants_cap and any(signal in folded_doc for signal in ("cift anadal programina basvurabilmesi", "en az 3,00", "en ust %20")):
                score += 0.75
            if wants_yandal and any(signal in folded_doc for signal in ("yandal programina basvurabilmesi", "en az 2.50", "en az 2,50", "2,29")):
                score += 0.85
            if wants_cap and not wants_yandal and any(
                signal in folded_doc
                for signal in (
                    "yandal programina basvurabilmesi",
                    "yandal programina devam",
                    "yandal sertifikasi",
                )
            ):
                score -= 0.60
            if wants_yandal and not wants_cap and "cift anadal programina basvurabilmesi" in folded_doc:
                score -= 0.30

    if detected_category == "staj":
        folded_query = _fold(query)
        source = _source_text(metadata)
        if "staj-uygulamalari" in source:
            score += 0.20
        elif "uygulamali-egitimler" in source:
            score += 0.10
        if "sorumlu" in folded_query and "sorumlu ogretim elemani" in folded_doc:
            score += 0.55
        if "sorumlu" in folded_query and any(
            signal in folded_doc
            for signal in (
                "bolum/program baskanlari, sorumlu ogretim elemanidir",
                "sorumlu ogretim elemani: bolum/program baskanlarini",
            )
        ):
            score += 0.85
        if "teslim" in folded_query and "teslim" in folded_doc:
            score += 0.35
        if any(signal in folded_query for signal in ("belge", "evrak", "defter", "nereden", "ulas")) and any(
            signal in folded_doc
            for signal in ("staj belgeleri", "gerekli tum belgeler", "internet sayfalarinda", "staj defteri")
        ):
            score += 0.45

    score += 0.20 * keyword_boost(query, doc)
    score += 0.12 * keyword_boost(query, _metadata_text(metadata))

    title = _fold(str(metadata.get("title", "")))
    if title and any(token in title for token in _tokenize(query)):
        score += 0.08

    return score


def _unique_candidates(docs: Iterable[str], metas: Iterable[dict]) -> list[tuple[str, dict]]:
    seen = set()
    candidates = []

    for doc, metadata in zip(docs, metas):
        if not doc:
            continue

        key = _norm_key(doc)
        if key in seen:
            continue

        seen.add(key)
        candidates.append((doc, metadata or {}))

    return candidates


def _extend_with_category_candidates(collection, candidates: list[tuple[str, dict]], category: str | None):
    category = normalize_category(category)
    if not is_supported_category(category):
        return candidates

    try:
        extra = collection.get(
            where={"category": category},
            limit=80,
            include=["documents", "metadatas"],
        )
    except Exception:
        return candidates

    combined = candidates + _unique_candidates(
        extra.get("documents", []),
        extra.get("metadatas", []),
    )
    return _unique_candidates(
        [doc for doc, _ in combined],
        [metadata for _, metadata in combined],
    )


def _apply_relevance_filter(
    ranked: list[tuple[float, str, dict]],
    detected_category: str | None,
    query: str = "",
):
    if not ranked:
        return []

    detected_category = normalize_category(detected_category)

    if detected_category in STRICT_INTENTS:
        category_matches = [
            item for item in ranked if normalize_category(item[2].get("category")) == detected_category
        ]
        if category_matches:
            ranked = category_matches
        else:
            return []

    if detected_category == "cift_anadal_yandal":
        folded_query = _fold(query)
        date_query = _is_date_query(folded_query)
        if not date_query:
            policy_matches = [
                item
                for item in ranked
                if _is_cift_yandal_policy_source(item[2])
            ]
            if policy_matches:
                ranked = policy_matches
        else:
            calendar_matches = [
                item
                for item in ranked
                if _is_calendar_source(item[2])
            ]
            if calendar_matches and ("basvur" in folded_query or "kayit" in folded_query):
                ranked = calendar_matches

    if detected_category == "erasmus":
        agreement_query = _is_agreement_query(query)
        erasmus_matches = [
            item
            for item in ranked
            if _is_erasmus_source(item[2], item[1])
        ]
        if erasmus_matches:
            ranked = erasmus_matches
        if agreement_query:
            agreement_matches = [
                item
                for item in ranked
                if _is_agreement_record(item[2], item[1])
            ]
            if agreement_matches:
                ranked = agreement_matches

    if detected_category == "ogretim_gorevlileri":
        staff_matches = [
            item
            for item in ranked
            if item[2].get("record_type") == "academic_staff"
        ]
        if staff_matches:
            ranked = staff_matches

        unit_signal = _academic_unit_signal(_fold(query))
        if unit_signal:
            unit_matches = [
                item
                for item in ranked
                if unit_signal in _fold(f"{item[1]} {_metadata_text(item[2])}")
            ]
            if unit_matches:
                ranked = unit_matches

    if detected_category == "programlar":
        catalog_matches = [
            item
            for item in ranked
            if any(
                signal in _fold(f"{item[1]} {_metadata_text(item[2])}")
                for signal in ("tum-programlar", "tum programlar", "program adi")
            )
        ]
        if catalog_matches:
            ranked = catalog_matches

    if detected_category == "ulasim_iletisim":
        address_matches = [
            item
            for item in ranked
            if any(
                signal in _fold(f"{item[1]} {_metadata_text(item[2])}")
                for signal in ("kroki", "nato yolu", "bosna blv", "444 79 97", "info@dogus")
            )
        ]
        if address_matches:
            ranked = address_matches

    if detected_category == "ucret_odeme":
        payment_matches = [
            item
            for item in ranked
            if any(
                signal in _fold(f"{item[1]} {_metadata_text(item[2])}")
                for signal in ("online odeme", "ucretler ve odeme kosullari", "odeme")
            )
        ]
        if payment_matches:
            ranked = payment_matches

    best_score = ranked[0][0]
    if best_score < MIN_RELEVANCE_SCORE:
        return []

    score_floor = max(MIN_RELEVANCE_SCORE, best_score - 0.70)
    return [item for item in ranked if item[0] >= score_floor]


def _query_collection(collection, query_embedding: list[float], n_results: int, category: str | None):
    where = {"category": normalize_category(category)} if is_supported_category(category) else None

    try:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas"],
        )
    except Exception:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"],
        )


def retrieve(query: str, top_k: int = 6, candidate_k: int = DEFAULT_CANDIDATE_K):
    if not query or not query.strip():
        return []

    embedder, reranker, category_embeddings = load_ai_core()
    collection = load_chroma()

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode(
        f"query: {query}",
        normalize_embeddings=True,
    ).tolist()
    detected_category = detect_intent_semantic(query, embedder, category_embeddings)
    if not is_supported_category(detected_category):
        return []

    agreement_query = detected_category == "erasmus" and _is_agreement_query(query)
    if detected_category == "ogretim_gorevlileri":
        minimum_candidates = 80
    elif agreement_query:
        minimum_candidates = 100
    else:
        minimum_candidates = candidate_k
    n_results = min(max(candidate_k, minimum_candidates, top_k * 4), collection.count())
    results = _query_collection(collection, query_embedding, n_results, detected_category)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    candidates = _unique_candidates(docs, metas)
    if not candidates and os.getenv("DORA_EXPAND_CATEGORY_CANDIDATES", "0") == "1":
        candidates = _extend_with_category_candidates(collection, candidates, detected_category)
    if not candidates:
        return []

    pairs = [[query, doc] for doc, _ in candidates]
    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked = sorted(
        (
            (
                _score_candidate(query, doc, metadata, score, detected_category),
                doc,
                metadata,
            )
            for (doc, metadata), score in zip(candidates, scores)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    ranked = _apply_relevance_filter(ranked, detected_category, query=query)

    return [
        {
            "text": doc,
            "metadata": metadata,
            "score": score,
        }
        for score, doc, metadata in ranked[:top_k]
    ]


def build_context(chunks: list, max_chars: int = 7000) -> str:
    parts = []
    total = 0

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        title = metadata.get("title") or metadata.get("source", "unknown")
        source = metadata.get("source", "unknown")
        url = metadata.get("url", "")
        safe_url = url if _is_public_url(url) and not _looks_like_local_path(url) else ""
        category = normalize_category(metadata.get("category", "unknown"))
        text = chunk.get("text", "")

        block = (
            f"[TITLE: {title}]\n"
            f"[CATEGORY: {category}]\n"
            f"[SOURCE: {source}]\n"
            f"[URL: {safe_url}]\n"
            f"{text}"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)


def get_sources(chunks: list) -> list:
    sources = []
    seen = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = format_source_reference(metadata)
        if source and source not in seen:
            sources.append(source)
            seen.add(source)

    return sources
