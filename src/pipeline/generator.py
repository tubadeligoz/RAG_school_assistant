import json
import os
import re
import unicodedata

import requests

from src.pipeline.prompt_manager import render_answer_prompt, render_prompt
from src.pipeline.retriever import build_context
from src.pipeline.scope import (
    CATEGORY_LABELS,
    detect_scope_category,
    detect_scope_category_from_history,
    normalize_category,
    scope_limit_message,
)


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MAIN_MODEL = os.getenv("DORA_MAIN_MODEL", "qwen3:14b")
FAST_MODEL = os.getenv("DORA_FAST_MODEL", "qwen2.5:0.5b")
ENABLE_FOLLOWUP = os.getenv("DORA_ENABLE_FOLLOWUP", "0") == "1"

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

STANDALONE_TOPIC_SIGNALS = (
    "akademik takvim",
    "akademik kadro",
    "basvuru",
    "bolum",
    "burs",
    "butunleme",
    "cap",
    "cift anadal",
    "ders",
    "erasmus",
    "evrak",
    "fakulte",
    "final",
    "hoca",
    "iade",
    "ikili anlasma",
    "kampus",
    "kayit",
    "kredi",
    "harc",
    "odeme",
    "program",
    "sinav",
    "staj",
    "anlasmali universite",
    "anlasmalarimiz",
    "partner universite",
    "ogretim",
    "ogretim elemani",
    "ogretim gorevlisi",
    "ucret",
    "vize",
    "yandal",
)

DEPENDENCY_MARKERS = (
    "bu",
    "buna",
    "bunda",
    "bunu",
    "bunun",
    "oradaki",
    "orada",
    "onun",
    "peki",
    "sonuncu",
    "nereye",
    "nereden",
    "yatiracagim",
    "yatirilir",
    "odeyecegim",
)


def _fold_text(text: str) -> str:
    folded = str(text or "").translate(TR_MAP).casefold()
    normalized = unicodedata.normalize("NFKD", folded)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _ollama_payload(model: str, prompt: str, stream: bool, temperature: float, num_ctx: int, num_predict: int):
    return {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
        },
    }


def _call_ollama_sync(
    prompt: str,
    temperature: float = 0.1,
    use_fast: bool = True,
    num_predict: int = 160,
):
    model = FAST_MODEL if use_fast else MAIN_MODEL
    payload = _ollama_payload(
        model=model,
        prompt=prompt,
        stream=False,
        temperature=temperature,
        num_ctx=4096 if use_fast else 6144,
        num_predict=num_predict,
    )

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=35 if use_fast else 120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException:
        return ""
    except ValueError:
        return ""


def _call_ollama_stream(prompt: str, temperature: float = 0.1):
    payload = _ollama_payload(
        model=MAIN_MODEL,
        prompt=prompt,
        stream=True,
        temperature=temperature,
        num_ctx=6144,
        num_predict=900,
    )

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line)
            yield data.get("response", "")

            if data.get("done"):
                break
    except Exception as exc:
        yield f"\nModel hatası: {exc}"


def _clean_query(query: str):
    if not query:
        return ""

    query = query.strip()
    query = re.sub(r"^\d+[\.\)\-]\s*", "", query)
    query = query.replace("*", "").strip("- ").strip()

    if len(query.split()) < 2:
        return ""

    return query


def _looks_context_dependent(question: str) -> bool:
    folded_question = _fold_text(question)
    words = set(re.findall(r"[0-9a-z]+", folded_question))

    if words & set(DEPENDENCY_MARKERS):
        return True

    if any(signal in folded_question for signal in STANDALONE_TOPIC_SIGNALS):
        return False

    return False


def _history_text(history) -> str:
    return " ".join(str(message.get("content", "")) for message in history[-4:])


def _latest_user_question(history, fallback: str) -> str:
    fallback_folded = _fold_text(fallback)
    fallback_tokens = set(re.findall(r"[0-9a-z]+", fallback_folded))

    for message in reversed(history or []):
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            candidate = str(message.get("content", "")).strip()
            candidate_folded = _fold_text(candidate)
            candidate_tokens = set(re.findall(r"[0-9a-z]+", candidate_folded))

            if (
                candidate_folded in fallback_folded
                or fallback_folded in candidate_folded
                or (
                    candidate_tokens
                    and len(candidate_tokens & fallback_tokens) >= max(2, min(3, len(candidate_tokens)))
                )
            ):
                return candidate

            break

    return fallback


def _answer_history_text(history, current_question: str = "", max_messages: int = 6) -> str:
    if not history:
        return "Önceki sohbet yok."

    rows = []
    current_folded = _fold_text(current_question)

    for message in (history or [])[-max_messages:]:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            label = "Kullanıcı"
        elif role == "assistant":
            label = "Asistan"
        else:
            label = "Mesaj"

        if current_folded and role == "user" and _fold_text(content) == current_folded:
            continue

        rows.append(f"{label}: {content[:260]}")

    return "\n".join(rows) if rows else "Önceki sohbet yok."


def _deterministic_followup_question(history, question: str) -> str:
    folded_question = _fold_text(question)
    folded_history = _fold_text(_history_text(history))

    current_category = detect_scope_category(question)
    history_category = detect_scope_category_from_history(history)
    if current_category:
        return question

    if not current_category and history_category:
        label = CATEGORY_LABELS.get(history_category, history_category)
        return f"{label}: {question}"

    payment_question = any(signal in folded_question for signal in ("nereye", "yatir", "ode"))
    payment_context = any(signal in folded_history for signal in ("ucret", "odeme", "harc", "yatir"))
    if payment_question and payment_context:
        return "Okul ücreti nereye yatırılır?"

    return ""


def _valid_condensed_question(original: str, candidate: str) -> bool:
    if not candidate:
        return False

    candidate = candidate.strip()
    if len(candidate) > 220 or len(candidate) < 4:
        return False

    if candidate.count("?") > 1:
        return False

    candidate_words = re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+", candidate)
    if len(candidate_words) < 2:
        return False

    original_tokens = set(re.findall(r"[0-9a-z]+", _fold_text(original)))
    candidate_tokens = set(re.findall(r"[0-9a-z]+", _fold_text(candidate)))

    if original_tokens and not (original_tokens & candidate_tokens):
        return False

    return True


def condense_question(history, question):
    if not history or not _looks_context_dependent(question):
        return question

    deterministic_question = _deterministic_followup_question(history, question)
    if deterministic_question:
        return deterministic_question

    history_text = "\n".join(
        f"{message['role']}: {message['content'][:180]}"
        for message in history[-4:]
    )

    prompt = render_prompt(
        "condense_question",
        history_text=history_text,
        question=question,
    )

    result = _call_ollama_sync(prompt, temperature=0.1, use_fast=True, num_predict=80)
    return result if _valid_condensed_question(question, result) else question


def generate_multi_queries(question: str):
    prompt = render_prompt("multi_query", question=question)

    response = _call_ollama_sync(prompt, temperature=0.2, use_fast=True, num_predict=120)
    if not response:
        return [question]

    queries = []
    for line in response.splitlines():
        query = _clean_query(line)
        if query and query.lower() != question.lower():
            queries.append(query)

    return queries[:2] if queries else [question]


def _is_bad_answer(text: str):
    if not text:
        return True

    lower_text = text.lower()
    bad_fragments = (
        "resmi kaynaklara bakın",
        "internet sitesinden",
        "internet sitesine bak",
    )

    if any(fragment in lower_text for fragment in bad_fragments):
        return True

    return len(text.split()) < 3


def _generate_followup(question: str):
    prompt = render_prompt("followup", question=question)

    question_text = _call_ollama_sync(prompt, temperature=0.3, use_fast=True, num_predict=60)
    if not question_text or len(question_text.split()) < 2:
        return ""

    return f"\n\n{question_text}"


def _direct_staj_answer(chunks, question: str) -> str:
    if not chunks:
        return ""

    first_category = normalize_category(chunks[0].get("metadata", {}).get("category"))
    if first_category != "staj":
        return ""

    folded_question = _fold_text(question)
    combined_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    folded_text = _fold_text(combined_text)

    responsible_question = (
        "sorumlu" in folded_question
        and any(signal in folded_question for signal in ("ogretim", "egitim", "eleman", "elaman"))
    )
    if responsible_question:
        lines = []
        if "bolum/program baskanlari" in folded_text and "sorumlu ogretim elemani" in folded_text:
            lines.append(
                "Staj yönergesindeki bilgiye göre bu rolde geçen sorumlu öğretim elemanı, "
                "bölüm/program başkanıdır. Yani kişi adı aramaktan önce kendi bölüm veya "
                "program başkanlığı hattından ilerlemen daha doğru olur."
            )
        elif "sorumlu ogretim elemani" in folded_text:
            lines.append(
                "Kaynakta sorumlu öğretim elemanının staj sürecinde görevli olduğu belirtiliyor; "
                "bu yüzden teslim ve onay adımlarında bu kişiyi/birimi esas almalısın."
            )

        if "ogrencilere aciklayici bilgiler" in folded_text or "bilgilendirme yapilir" in folded_text:
            lines.append("- Sorumlu öğretim elemanı staj süreci hakkında öğrencilere açıklayıcı bilgilendirme yapar.")
        if "staj basvuru ve kabul formu" in folded_text and "onayina sunar" in folded_text:
            lines.append("- Staj Başvuru ve Kabul Formu, staj yeri tarafından onaylandıktan sonra sorumlu öğretim elemanının onayına sunulur.")

        if lines:
            lines.append("- Kaynaklarda kişi adı listesi yok; bu yüzden isim uydurmuyorum.")
            return "\n".join(lines)

    delivery_question = "teslim" in folded_question or "kime" in folded_question
    if delivery_question and "sorumlu ogretim eleman" in folded_text:
        lines = [
            "Staj evraklarını teslim ederken esas muhatap sorumlu öğretim elemanı olarak görünüyor. "
            "Bunu sadece evrak bırakma noktası gibi değil, aynı zamanda başvuru/onay sürecini kontrol eden akademik sorumlu gibi düşünebilirsin."
        ]
        if "staj basvuru ve kabul formu" in folded_text and "onayina sunar" in folded_text:
            lines.append("- Staj Başvuru ve Kabul Formu, staj yeri tarafından onaylandıktan sonra sorumlu öğretim elemanının onayına sunulur.")
        if "sorumlu ogretim elemanina teslim" in folded_text:
            lines.append("- İlgili staj belgeleri/dosyası sorumlu öğretim elemanına teslim edilir.")
        if "15 gun" in folded_text:
            lines.append("- Onaylanan form ve gerekli belgeler staja başlama tarihinden en az 15 gün önce teslim edilir.")
        return "\n".join(lines)

    if not any(
        signal in folded_question
        for signal in (
            "belge",
            "evrak",
            "gerek",
            "basvuru",
            "form",
            "nereden",
            "nerede",
            "ulas",
            "ulaş",
            "bulabilir",
            "alabilir",
            "indir",
        )
    ):
        return ""

    wants_document_location = any(
        signal in folded_question
        for signal in ("nereden", "nerede", "ulas", "ulaş", "bulabilir", "alabilir", "indir")
    )
    if wants_document_location and any(
        signal in folded_text
        for signal in ("internet sayfalarinda", "internet sayfalarında", "cikti almak", "çıktı almak")
    ):
        lines = [
            "Staj belgelerine ilgili fakülte, yüksekokul veya meslek yüksekokulu internet sayfalarından ulaşabilirsiniz. "
            "Yani belgenin merkezi tek bir sayfada değil, bağlı olduğun akademik birimin staj/uygulamalı eğitim sayfasında yer alması bekleniyor.",
        ]
        if "staj defteri" in folded_text:
            lines.append("- Staj defteri de ilgili birimin internet sayfalarından çıktı alınarak kullanılabilir.")
        if "15 gun" in folded_text:
            lines.append("- Onaylanan form ve gerekli belgeler staja başlama tarihinden en az 15 gün önce teslim edilir.")
        return "\n".join(lines)
    lines = [
        "Staj için gereken evraklar temel olarak başvuru/onay formu, kimlik fotokopisi ve staj defteri etrafında toplanıyor. "
        "Kaynakta geçen belgeler şöyle:"
    ]

    if "staj basvuru ve kabul formu" in folded_text:
        lines.append("- Staj Başvuru ve Kabul Formu (EK-2)")

    if "nufus cuzdan" in folded_text:
        lines.append("- Nüfus cüzdanı fotokopisi")

    if "staj defteri" in folded_text:
        lines.append("- Staj defteri")

    worker_docs = []
    if "staj muafiyet dilekcesi" in folded_text:
        worker_docs.append("Çalışan Öğrenciler İçin Staj Muafiyet Dilekçesi")
    if "hizmet belgesi" in folded_text:
        worker_docs.append("Hizmet Belgesi")
    if "ise giris bildirgesi" in folded_text:
        worker_docs.append("SGK İşe Giriş Bildirgesi")
    if "hizmet dokumu" in folded_text:
        worker_docs.append("SGK Hizmet Dökümü")
    if worker_docs:
        lines.append("- Çalışan öğrenciler için ayrıca: " + ", ".join(worker_docs))

    if "15 gun" in folded_text:
        lines.append("- Onaylanan form ve gerekli belgeler staja başlama tarihinden en az 15 gün önce teslim edilir.")

    if len(lines) == 1:
        return ""

    lines.append("Pratikte önce formu doğru şekilde onaylatıp, ardından diğer belgelerle birlikte teslim tarihini kaçırmaman gerekiyor.")
    return "\n".join(lines)


def _direct_contact_answer(chunks) -> str:
    if not chunks:
        return ""

    if chunks[0].get("metadata", {}).get("category") != "ulasim_iletisim":
        return ""

    combined_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    lines = ["Doğuş Üniversitesi için veritabanındaki ulaşım/iletişim bilgileri:"]

    if "Nato Yolu Cad" in combined_text:
        lines.append("- Dudullu Kampüsü: Dudullu Osb Mah. Nato Yolu Cad. 265/1, 34775 Ümraniye / İstanbul")

    if "Bosna Blv No: 140" in combined_text:
        lines.append("- Çengelköy Kampüsü: Bahçelievler Mh., Bosna Blv No: 140, 34680 Üsküdar / İstanbul")

    phone_match = re.search(r"444\s*79\s*97", combined_text)
    if phone_match:
        lines.append(f"- Telefon: {phone_match.group(0)}")

    email_match = re.search(r"[\w\.-]+@dogus\.edu\.tr", combined_text, flags=re.IGNORECASE)
    if email_match:
        lines.append(f"- E-posta: {email_match.group(0)}")

    if "Kroki ve Ulaşım Bilgileri" in combined_text:
        lines.append("- Kaynakta her iki kampüs için \"Kroki ve Ulaşım Bilgileri için tıklayınız\" ifadesi geçiyor.")

    return "\n".join(lines) if len(lines) > 1 else ""


def _direct_payment_answer(chunks, question: str) -> str:
    folded_question = _fold_text(question)
    if not any(signal in folded_question for signal in ("ucret", "odeme", "harc", "yatir")):
        return ""

    if not chunks:
        return "Bu konuda veritabanında net bilgi bulunamadı."

    first_category = chunks[0].get("metadata", {}).get("category")
    if first_category not in {"ucret_odeme", "ucret_iade"}:
        return ""

    combined_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    folded_text = _fold_text(combined_text)

    has_online_payment = "online odeme" in folded_text
    has_fee_terms = "ucretler ve odeme kosullari" in folded_text
    has_account_detail = any(signal in folded_text for signal in ("iban", "banka", "hesap no", "hesap numarasi"))

    if has_account_detail:
        return ""

    if has_online_payment or has_fee_terms:
        lines = [
            "Veritabanında okul ücretinin yatırılacağı banka/IBAN bilgisi net olarak bulunamadı.",
            "Kaynaklarda yalnızca \"Ücretler ve Ödeme Koşulları\" ve \"Online Ödeme\" bağlantı metinleri geçiyor.",
            "Bu yüzden banka hesabı veya ödeme noktası uydurmuyorum.",
        ]
        return "\n".join(lines)

    return "Bu konuda veritabanında net bilgi bulunamadı."


def _calendar_source_label(source: str) -> str:
    folded_source = _fold_text(source)
    if "onlisans-lisans" in folded_source:
        return "Önlisans/Lisans"
    if "lisansustu" in folded_source:
        return "Lisansüstü"
    if "hazirlik" in folded_source:
        return "İngilizce Hazırlık"
    return source.replace("_", " ").replace(".json", "")


def _date_like(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d{1,2}(?:-\d{1,2})?\s+"
            r"(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)"
            r"(?:\s*-\s*\d{1,2}\s+"
            r"(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık))?"
            r"(?:\s+\d{4})?",
            text,
            flags=re.IGNORECASE,
        )
    )


def _direct_final_exam_answer(chunks, question: str) -> str:
    folded_question = _fold_text(question)
    if "final" not in folded_question:
        return ""

    if not chunks or chunks[0].get("metadata", {}).get("category") != "akademik_takvim":
        return ""

    rows = []
    seen = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("category") != "akademik_takvim":
            continue

        source_label = _calendar_source_label(metadata.get("source", ""))
        lines = [line.strip() for line in chunk.get("text", "").splitlines() if line.strip()]

        for index, line in enumerate(lines):
            folded_line = _fold_text(line)
            if "final" not in folded_line or "sinav" not in folded_line:
                continue
            if any(skip in folded_line for skip in ("not", "duiyes", "yeterlik", "munazara")):
                continue

            date = ""
            if _date_like(line):
                date = line
            elif index > 0 and _date_like(lines[index - 1]):
                date = lines[index - 1]

            if not date:
                continue

            label = re.sub(r"^\d{1,2}.*?\d{4}\s*", "", line).strip() or line
            key = (_fold_text(source_label), _fold_text(label), _fold_text(date))
            if key in seen:
                continue

            seen.add(key)
            rows.append((source_label, label, date))

    if not rows:
        return ""

    lines = ["Final sınav tarihleri için veritabanında bulunan bilgiler:"]
    for source_label, label, date in rows[:8]:
        lines.append(f"- {source_label}: {label} - {date}")

    lines.append("Program türünü söylersen sadece ilgili takvimi de gösterebilirim.")
    return "\n".join(lines)


ERASMUS_AGREEMENT_RECORD_TYPES = {
    "erasmus_partner_university",
    "erasmus_partner_overview",
    "bilateral_partner_university",
    "agreement_overview",
}


def _is_erasmus_agreement_question(question: str) -> bool:
    folded_question = _fold_text(question)
    return any(
        signal in folded_question
        for signal in (
            "ikili anlas",
            "ikili isbirligi",
            "anlasmali universite",
            "anlasmali universiteler",
            "anlasmalarimiz",
            "partner universite",
            "partner university",
            "yurt disi universite",
            "yurtdisi universite",
            "uluslararasi isbirligi",
        )
    )


def _requested_country(question: str, chunks) -> str:
    folded_question = _fold_text(question)
    countries = []
    for chunk in chunks:
        country = str(chunk.get("metadata", {}).get("country", "")).strip()
        if country and country not in countries:
            countries.append(country)

    for country in countries:
        if _fold_text(country) in folded_question:
            return country

    return ""


def _direct_erasmus_agreement_answer(chunks, question: str) -> str:
    if not chunks:
        return ""

    first_category = normalize_category(chunks[0].get("metadata", {}).get("category"))
    if first_category != "erasmus" or not _is_erasmus_agreement_question(question):
        return ""

    agreement_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("record_type") in ERASMUS_AGREEMENT_RECORD_TYPES
    ]
    if not agreement_chunks:
        return ""

    partner_rows = []
    seen = set()
    requested_country = _requested_country(question, agreement_chunks)

    for chunk in agreement_chunks:
        metadata = chunk.get("metadata", {})
        record_type = metadata.get("record_type")
        if record_type not in {"erasmus_partner_university", "bilateral_partner_university"}:
            continue

        country = str(metadata.get("country", "")).strip()
        university = str(metadata.get("partner_university", "")).strip()
        scope = str(metadata.get("agreement_scope", "")).strip()
        if not university:
            continue
        if requested_country and _fold_text(country) != _fold_text(requested_country):
            continue

        key = (_fold_text(country), _fold_text(university), _fold_text(scope))
        if key in seen:
            continue
        seen.add(key)
        partner_rows.append((country, university, scope))

    overview = next(
        (
            chunk
            for chunk in agreement_chunks
            if chunk.get("metadata", {}).get("record_type") == "erasmus_partner_overview"
        ),
        None,
    )
    if overview:
        metadata = overview.get("metadata", {})
        country_count = metadata.get("country_count")
        partner_count = metadata.get("partner_count")
    else:
        country_count = ""
        partner_count = ""

    lines = []
    if requested_country:
        lines.append(f"Resmi anlaşma kayıtlarında {requested_country} için bulunan üniversiteler:")
    elif partner_count and country_count:
        lines.append(
            f"Resmi Erasmus+ Anlaşmalarımız sayfasına göre listede {country_count} ülke ve "
            f"{partner_count} anlaşmalı üniversite kaydı bulunuyor."
        )
    else:
        lines.append("Resmi anlaşma kayıtlarında bulunan üniversiteler:")

    if overview:
        skip_prefixes = (
            "kategori:",
            "konu:",
            "kapsam:",
            "toplam ulke:",
            "toplam universite:",
            "kaynak:",
        )
        overview_country_lines = []
        for line in overview.get("text", "").splitlines():
            line = line.strip()
            folded_line = _fold_text(line)
            if ":" not in line or any(folded_line.startswith(prefix) for prefix in skip_prefixes):
                continue
            overview_country_lines.append(line)

        if requested_country:
            requested_folded = _fold_text(requested_country)
            for line in overview_country_lines:
                country_label = line.split(":", 1)[0]
                if _fold_text(country_label) == requested_folded:
                    lines.append(f"- {line}")
                    return "\n".join(lines)

        if overview_country_lines and not requested_country:
            lines.extend(f"- {line}" for line in overview_country_lines[:12])
            if len(overview_country_lines) > 12:
                lines.append(f"- Ayrıca {len(overview_country_lines) - 12} ülke grubu daha var; ülke veya bölüm söylersen daraltabilirim.")
            return "\n".join(lines)

    if partner_rows:
        for country, university, scope in partner_rows[:18]:
            detail = country or scope
            lines.append(f"- {university}" + (f" ({detail})" if detail else ""))
        if len(partner_rows) > 18:
            lines.append(f"- Ayrıca {len(partner_rows) - 18} kayıt daha var; ülke ya da bölüm söylersen daraltabilirim.")
        elif not requested_country and partner_count:
            lines.append("Liste uzun olduğu için ülke ya da bölüm adıyla sorarsan daha hedefli cevap verebilirim.")
        return "\n".join(lines)

    if overview:
        overview_lines = [
            line.strip()
            for line in overview.get("text", "").splitlines()
            if ":" in line and line.strip()
        ]
        if overview_lines:
            lines.extend(f"- {line}" for line in overview_lines[:12])
            lines.append("Ülke veya bölüm belirtirsen listeyi daha okunur şekilde daraltabilirim.")
            return "\n".join(lines)

    return ""


def _direct_ogretim_gorevlileri_answer(chunks, question: str) -> str:
    if not chunks:
        return ""

    first_category = normalize_category(chunks[0].get("metadata", {}).get("category"))
    if first_category != "ogretim_gorevlileri":
        return ""

    folded_question = _fold_text(question)
    if not any(
        signal in folded_question
        for signal in (
            "kim",
            "kimler",
            "hoca",
            "hocalar",
            "nereden",
            "bul",
            "akademik kadro",
            "ogretim gorevlileri",
            "ogretim elemanlari",
        )
    ):
        return ""

    staff_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("record_type") == "academic_staff"
    ]
    if staff_chunks:
        wants_email = any(signal in folded_question for signal in ("mail", "eposta", "e-posta", "iletisim"))
        rows = []
        seen = set()

        for chunk in staff_chunks:
            metadata = chunk.get("metadata", {})
            name = str(metadata.get("name", "")).strip()
            academic_title = str(metadata.get("academic_title", "")).strip()
            role = str(metadata.get("role", "")).strip()
            faculty = str(metadata.get("faculty", "")).strip()
            department = str(metadata.get("department", "")).strip()
            email = str(metadata.get("email", "")).strip()

            if not name:
                continue

            key = (name.casefold(), department.casefold(), role.casefold())
            if key in seen:
                continue
            seen.add(key)

            label = " ".join(part for part in (academic_title, name) if part).strip()
            details = []
            if role:
                details.append(role)
            if department:
                details.append(department)
            elif faculty:
                details.append(faculty)
            if wants_email and email:
                details.append(email)

            rows.append(f"- {label}" + (f" ({', '.join(details)})" if details else ""))

        if rows:
            lines = ["Resmi akademik kadro kayıtlarına göre bulunan kişiler:"]
            lines.extend(rows[:12])
            if len(rows) > 12:
                lines.append(f"- Ayrıca {len(rows) - 12} kayıt daha var; bölüm/program adını söylersen listeyi daraltabilirim.")
            return "\n".join(lines)

    combined_text = "\n".join(chunk.get("text", "") for chunk in chunks)
    folded_text = _fold_text(combined_text)

    has_academic_staff_link = "akademik kadro" in folded_text
    has_only_news_like_people = any(
        signal in folded_text
        for signal in (
            "devamini oku",
            "konferansi",
            "komite uyesi",
            "sempozyumu",
            "davetli konusmaci",
        )
    )

    if has_academic_staff_link or has_only_news_like_people:
        lines = [
            "Bu konuda veritabanındaki parçalar tam ve güvenilir bir öğretim görevlisi listesi vermiyor."
        ]
        if has_academic_staff_link:
            lines.append(
                "- Kaynakta akademik kadroya fakülte/yüksekokul/program sayfalarındaki \"Akademik Kadro\" alanından gidilebildiği görünüyor."
            )
        if has_only_news_like_people:
            lines.append(
                "- Bazı kişi adları haber/duyuru metinlerinde geçtiği için bunları okulun tüm öğretim görevlileri listesi gibi sunmam doğru olmaz."
            )

        lines.append(
            "En sağlıklı yol, ilgili fakülte veya bölüm sayfasını açıp o birimin Akademik Kadro bölümünden isimlere bakmak."
        )
        return "\n".join(lines)

    return ""


def _primary_category(chunks, question: str) -> str | None:
    if chunks:
        category = chunks[0].get("metadata", {}).get("category")
        if category:
            return normalize_category(category)

    return detect_scope_category(question)


def build_answer_prompt(
    context: str,
    question: str,
    category: str | None = None,
    history_text: str = "",
) -> str:
    return render_answer_prompt(context, question, category, history_text=history_text)


def generate_answer_stream(chunks, question, history=None):
    user_question = _latest_user_question(history, question)
    answer_history = _answer_history_text(history, user_question)

    if not chunks:
        if detect_scope_category(question) is None:
            yield scope_limit_message()
        else:
            yield "Bu konuda veritabanında net bilgi bulunamadı."
        return

    direct_answer = _direct_staj_answer(chunks, user_question)
    if direct_answer:
        yield direct_answer
        return

    direct_answer = _direct_payment_answer(chunks, user_question)
    if direct_answer:
        yield direct_answer
        return

    direct_answer = _direct_contact_answer(chunks)
    if direct_answer:
        yield direct_answer
        return

    direct_answer = _direct_final_exam_answer(chunks, user_question)
    if direct_answer:
        yield direct_answer
        return

    direct_answer = _direct_erasmus_agreement_answer(chunks, user_question)
    if direct_answer:
        yield direct_answer
        return

    direct_answer = _direct_ogretim_gorevlileri_answer(chunks, user_question)
    if direct_answer:
        yield direct_answer
        return

    context = build_context(chunks)
    if not context.strip():
        yield "Bu konuda veritabanında net bilgi bulunamadı."
        return

    prompt = build_answer_prompt(
        context,
        user_question,
        _primary_category(chunks, question),
        history_text=answer_history,
    )

    full_answer = ""
    for token in _call_ollama_stream(prompt, temperature=0.2):
        full_answer += token
        yield token

    if _is_bad_answer(full_answer):
        yield "\n\nBu konuda veritabanında yeterli bilgi bulunamadı."
        return

    if ENABLE_FOLLOWUP:
        followup = _generate_followup(question)
        if followup:
            yield followup


def generate_answer(context: str, question: str, category: str | None = None, history_text: str = ""):
    if not str(context or "").strip():
        if detect_scope_category(question) is None:
            return scope_limit_message()
        return "Bu konuda veritabanında net bilgi bulunamadı."

    prompt = build_answer_prompt(
        context,
        question,
        category or detect_scope_category(question),
        history_text=history_text,
    )
    answer = _call_ollama_sync(
        prompt,
        temperature=0.2,
        use_fast=False,
        num_predict=900,
    )

    if _is_bad_answer(answer):
        return "Bu konuda veritabanında net bilgi bulunamadı."

    return answer
