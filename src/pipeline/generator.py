import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1"

# ==========================================
# 🧠 YARDIMCI: OLLAMA ÇAĞRICI (RAW)
# ==========================================
def _call_ollama_raw(prompt: str, temperature: float = 0.0) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 4096}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama Call Error: {e}")
        return ""

# ==========================================
# 🔄 SORGU AKILLANDIRMA (CONDENSATION)
# ==========================================
def condense_question(history: list[dict], current_question: str) -> str:
    if not history:
        return current_question

    history_str = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in history[-3:]])
    
    prompt = f"""Konuşma Geçmişi:
{history_str}

Kullanıcının Son Sorusu: {current_question}

GÖREV: Kullanıcının sorusunu geçmişe bakarak, bağımsız ve tam bir arama sorgusu olarak yeniden yaz. SADECE soruyu döndür.
Örnek: "Peki adresi nedir?" -> "Doğuş Üniversitesi Dudullu Kampüsü'nün açık adresi nedir?" """

    return _call_ollama_raw(prompt, temperature=0.1)

# ==========================================
# 🚀 SORGU GENİŞLETME (MULTI-QUERY)
# ==========================================
def generate_multi_queries(question: str) -> list[str]:
    prompt = f"""Soru: {question}
    GÖREV: Bu soruyu, veritabanında en iyi sonucu bulabilmek için 3 farklı şekilde yeniden yaz.
    Sadece soruları aralarında yeni satır olacak şekilde yaz. Ek açıklama yapma."""
    response = _call_ollama_raw(prompt, temperature=0.4)
    return [q.strip() for q in response.split('\n') if q.strip()][:3]

# ==========================================
# 🛠️ BAĞLAM İNŞASI
# ==========================================
def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, item in enumerate(chunks):
        meta = item.get("metadata", {})
        url = meta.get("url", "Resmi Kaynak")
        parts.append(f"[Kaynak: {url}]\n{item['text']}")
    return "\n\n---\n\n".join(parts)

# ==========================================
# 🌊 STREAMING GENERATOR (ULTRA-LOGIC)
# ==========================================
def generate_answer_stream(chunks: list[dict], question: str, history: list[dict] = None):
    
    context = _build_context(chunks)

    history_context = ""
    if history:
        last_msgs = [f"{m['role'].upper()}: {m['content'][:150]}" for m in history[-2:]]
        history_context = "Yakın Hafıza:\n" + "\n".join(last_msgs)

    # 🧠 MANTIKSAL KUSURSUZLUK (LOGICAL FLAWLESSNESS) PROMPT'U
    prompt = f"""Sen Doğuş Üniversitesi'nin otonom akademik asistanı DORA'sın.

HAYATİ MANTIK KURALLARI (BUNLARI ASLA İHLAL ETME):
1. MANTIKSAL ÇELİŞKİ YAPMA: "Evet, yok" veya "Evet, bulunmuyor" gibi saçma cümleler kurma. Bir bölüm, özellik veya bilgi BAĞLAMDA YOKSA, doğrudan "Hayır, [X] bulunmamaktadır" veya "Bu konuda bir bilgiye ulaşamadım" de.
2. GEÇMİŞİ KOPYALAMA (ODAKLAN): YAKIN HAFIZA sadece konuyu anlaman içindir. Cevabı SADECE KULLANICININ SON SORUSUNA ver. Soru sadece "Telefon numarası nedir?" ise, YALNIZCA telefonu ver. Önceki mesajlarda konuştuğunuz adresleri veya krokileri tekrar anlatma.
3. TEMBEL KAYNAK METİNLERİNİ FİLTRELE: Eğer sana verilen BAĞLAM metninin içinde "Ulaşım için tıklayınız", "Kroki sayfasında yer almaktadır" gibi yönlendirme cümleleri varsa, BUNLARI KULLANICIYA SÖYLEME. Metinde somut bilgi (otobüs no, metro adı) yoksa, "Ulaşım detayları veritabanımda somut olarak yer almıyor" de.

YAKIN HAFIZA:
{history_context}

BAĞLAM (Kullanabileceğin Tek Veri Kaynağı):
{context}

SON SORU: {question}

YANIT:""".strip()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0, # Halüsinasyonu sıfırlamak için en katı ayar
            "num_ctx": 8192
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        for line in response.iter_lines():
            if line:
                body = json.loads(line)
                chunk = body.get("response", "")
                if chunk:
                    yield chunk
                if body.get("done"):
                    break
    except Exception as e:
        yield f"❌ Akış Hatası: {e}"