import base64
import html
import os
import sys
import time
import unicodedata
from pathlib import Path

import streamlit as st


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ASSETS_DIR = Path(ROOT) / "assets"
LOGO_PATH = ASSETS_DIR / "dogus-logo-tr.png"

from src.pipeline.generator import (  # noqa: E402
    condense_question,
    generate_answer_stream,
    generate_multi_queries,
)
from src.pipeline.retriever import get_sources, retrieve  # noqa: E402
from src.pipeline.scope import detect_scope_category  # noqa: E402


st.set_page_config(
    page_title="Doğuş Asistanı",
    page_icon="📚",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def image_to_data_uri(path):
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_logo_src():
    if not LOGO_PATH.exists():
        return ""
    return image_to_data_uri(str(LOGO_PATH))


@st.cache_resource(show_spinner=False)
def warmup_retrieval_stack():
    from src.pipeline.retriever import load_ai_core, load_chroma

    start = time.perf_counter()
    load_ai_core()
    collection = load_chroma()
    return {
        "elapsed": round(time.perf_counter() - start, 1),
        "chunk_count": collection.count(),
    }


@st.cache_resource(show_spinner=False)
def warmup_answer_model():
    from src.pipeline.generator import MAIN_MODEL, _call_ollama_sync

    start = time.perf_counter()
    _call_ollama_sync(
        "Kısa bir Türkçe hazır yanıtı ver: hazırım.",
        temperature=0.0,
        use_fast=False,
        num_predict=8,
    )
    return {
        "elapsed": round(time.perf_counter() - start, 1),
        "model": MAIN_MODEL,
    }


SUGGESTED_PROMPTS = [
    {
        "id": "staj-evraklari",
        "title": "Staj evrakları",
        "body": "Gerekli belgeleri ve teslim sürecini öğren.",
        "prompt": "Staj için gerekli evraklar neler?",
    },
    {
        "id": "staj-sorumlusu",
        "title": "Staj sorumlusu",
        "body": "Kime teslim edeceğini ve sorumlu kişiyi sor.",
        "prompt": "Staj evraklarını kime teslim edeceğim?",
    },
    {
        "id": "erasmus",
        "title": "Erasmus",
        "body": "Başvuru, sınav ve hareketlilik bilgilerini kontrol et.",
        "prompt": "Erasmus başvuruları hakkında bilgi verir misin?",
    },
    {
        "id": "cap-yandal",
        "title": "ÇAP ve yandal",
        "body": "Başvuru şartları ve ortalama koşullarını sor.",
        "prompt": "ÇAP ve yandal şartları nelerdir?",
    },
]


THEMES = {
    "Aydınlık": {
        "bg": "#fbfbfd",
        "panel": "#ffffff",
        "panel_soft": "#fff3f3",
        "text": "#161316",
        "muted": "#67606a",
        "border": "#eadbdd",
        "accent": "#c5162e",
        "accent_dark": "#9e1023",
        "assistant": "#ffffff",
        "shadow": "0 18px 45px rgba(127, 18, 34, 0.10)",
        "input": "#ffffff",
    },
    "Koyu": {
        "bg": "#121214",
        "panel": "#1c1c20",
        "panel_soft": "#2a171b",
        "text": "#f7f2f3",
        "muted": "#b9afb2",
        "border": "#3a2d31",
        "accent": "#ff465d",
        "accent_dark": "#e33249",
        "assistant": "#1d1d22",
        "shadow": "0 18px 45px rgba(0, 0, 0, 0.28)",
        "input": "#1d1d22",
    },
}


def inject_theme(theme_name):
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
            :root {{
                --dora-bg: {theme["bg"]};
                --dora-panel: {theme["panel"]};
                --dora-panel-soft: {theme["panel_soft"]};
                --dora-text: {theme["text"]};
                --dora-muted: {theme["muted"]};
                --dora-border: {theme["border"]};
                --dora-accent: {theme["accent"]};
                --dora-accent-dark: {theme["accent_dark"]};
                --dora-assistant: {theme["assistant"]};
                --dora-shadow: {theme["shadow"]};
                --dora-input: {theme["input"]};
            }}

            html, body, [data-testid="stAppViewContainer"], .stApp {{
                background:
                    linear-gradient(
                        180deg,
                        color-mix(in srgb, var(--dora-panel-soft) 42%, var(--dora-bg)) 0%,
                        var(--dora-bg) 360px
                    );
                color: var(--dora-text);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            .block-container {{
                max-width: 1080px;
                padding-top: 2rem;
                padding-bottom: 7rem;
            }}

            [data-testid="stSidebar"] {{
                background: var(--dora-panel);
                border-right: 1px solid var(--dora-border);
            }}

            [data-testid="stSidebar"] * {{
                color: var(--dora-text);
            }}

            .dora-sidebar-logo {{
                display: flex;
                align-items: center;
                gap: 0.7rem;
                margin: 0.15rem 0 1rem;
                padding: 0.75rem;
                border: 1px solid var(--dora-border);
                border-radius: 12px;
                background: color-mix(in srgb, var(--dora-panel-soft) 42%, transparent);
            }}

            .dora-sidebar-logo img {{
                width: 54px;
                height: 54px;
                object-fit: contain;
                border-radius: 50%;
                background: #ffffff;
                border: 1px solid rgba(197, 22, 46, 0.18);
            }}

            .dora-sidebar-title {{
                margin: 0;
                font-size: 1rem;
                line-height: 1.2;
                font-weight: 780;
            }}

            .dora-sidebar-caption {{
                margin: 0.18rem 0 0;
                color: var(--dora-muted);
                font-size: 0.82rem;
                line-height: 1.35;
            }}

            .dora-welcome {{
                padding: 2.4rem 0 1rem;
            }}

            .dora-welcome-inner {{
                display: grid;
                grid-template-columns: 128px minmax(0, 1fr);
                align-items: center;
                gap: 1.45rem;
            }}

            .dora-hero-logo {{
                width: 128px;
                height: 128px;
                display: grid;
                place-items: center;
                border-radius: 28px;
                background: #ffffff;
                border: 1px solid rgba(197, 22, 46, 0.16);
                box-shadow: 0 20px 45px rgba(197, 22, 46, 0.16);
                overflow: hidden;
            }}

            .dora-hero-logo img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                padding: 7px;
            }}

            .dora-eyebrow {{
                display: inline-flex;
                align-items: center;
                width: fit-content;
                margin-bottom: 0.85rem;
                padding: 0.36rem 0.68rem;
                border: 1px solid color-mix(in srgb, var(--dora-accent) 34%, var(--dora-border));
                border-radius: 999px;
                background: var(--dora-panel-soft);
                color: var(--dora-accent);
                font-size: 0.8rem;
                font-weight: 760;
            }}

            .dora-welcome h1 {{
                margin: 0;
                max-width: 780px;
                font-size: 3.05rem;
                line-height: 1.04;
                color: var(--dora-text);
                letter-spacing: 0;
            }}

            .dora-welcome p {{
                margin: 1rem 0 0;
                max-width: 680px;
                color: var(--dora-muted);
                font-size: 1rem;
                line-height: 1.65;
            }}

            .dora-card-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.9rem;
                margin-top: 1.35rem;
            }}

            .dora-card-link {{
                color: inherit;
                display: block;
                text-decoration: none;
                cursor: pointer;
            }}

            .dora-card {{
                min-height: 138px;
                position: relative;
                padding: 1.05rem 1.1rem 1rem 1.2rem;
                border: 1px solid var(--dora-border);
                border-radius: 8px;
                background:
                    linear-gradient(
                        180deg,
                        color-mix(in srgb, var(--dora-panel-soft) 24%, var(--dora-panel)) 0%,
                        var(--dora-panel) 78%
                    );
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.05);
                transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
            }}

            .dora-card::before {{
                content: "";
                position: absolute;
                left: 0;
                top: 0.9rem;
                bottom: 0.9rem;
                width: 3px;
                border-radius: 999px;
                background: var(--dora-accent);
            }}

            .dora-card:hover {{
                border-color: color-mix(in srgb, var(--dora-accent) 42%, var(--dora-border));
                transform: translateY(-1px);
                box-shadow: var(--dora-shadow);
            }}

            .dora-card-link:focus-visible .dora-card {{
                outline: 3px solid color-mix(in srgb, var(--dora-accent) 28%, transparent);
                outline-offset: 3px;
            }}

            .dora-card-title {{
                color: var(--dora-text);
                font-weight: 800;
                font-size: 1rem;
                margin-bottom: 0.45rem;
            }}

            .dora-card-body {{
                color: var(--dora-muted);
                font-size: 0.9rem;
                line-height: 1.45;
            }}

            .dora-card-prompt {{
                margin-top: 1rem;
                padding-top: 0.8rem;
                border-top: 1px solid color-mix(in srgb, var(--dora-border) 72%, transparent);
                color: var(--dora-accent);
                font-size: 0.83rem;
                font-weight: 700;
                line-height: 1.35;
            }}

            .dora-scope {{
                margin: 1.1rem 0 0;
                padding: 0.85rem 1rem;
                border: 1px solid var(--dora-border);
                border-radius: 8px;
                color: var(--dora-muted);
                background: var(--dora-panel-soft);
                font-size: 0.9rem;
                text-align: center;
            }}

            [data-testid="stChatMessage"] {{
                border: 1px solid var(--dora-border);
                border-radius: 14px;
                padding: 0.75rem 0.95rem;
                background: var(--dora-assistant);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
            }}

            [data-testid="stChatMessage"] p {{
                color: var(--dora-text);
                line-height: 1.65;
            }}

            [data-testid="stChatInput"] {{
                width: min(860px, calc(100vw - 2rem));
                margin: 0 auto 0.65rem;
            }}

            [data-testid="stChatInput"] > div {{
                min-height: 58px;
                display: flex;
                align-items: center;
                border: 1px solid var(--dora-border);
                border-radius: 18px;
                background:
                    linear-gradient(
                        180deg,
                        color-mix(in srgb, var(--dora-panel) 94%, var(--dora-panel-soft)) 0%,
                        var(--dora-input) 100%
                    );
                box-shadow: 0 16px 42px rgba(0, 0, 0, 0.10);
                transition: border-color 160ms ease, box-shadow 160ms ease;
            }}

            [data-testid="stChatInput"] > div:focus-within {{
                border-color: color-mix(in srgb, var(--dora-accent) 58%, var(--dora-border));
                box-shadow:
                    0 18px 46px rgba(0, 0, 0, 0.12),
                    0 0 0 4px color-mix(in srgb, var(--dora-accent) 14%, transparent);
            }}

            [data-testid="stChatInput"] textarea {{
                min-height: 48px !important;
                max-height: 132px !important;
                padding: 0.8rem 3.15rem 0.8rem 1rem !important;
                background: transparent !important;
                color: var(--dora-text) !important;
                border: 0 !important;
                border-radius: 18px !important;
                box-shadow: none !important;
                font-size: 0.96rem !important;
                line-height: 1.45 !important;
            }}

            [data-testid="stChatInput"] textarea::placeholder {{
                color: color-mix(in srgb, var(--dora-muted) 78%, transparent);
                opacity: 1;
            }}

            [data-testid="stChatInput"] button {{
                width: 38px !important;
                height: 38px !important;
                min-width: 38px !important;
                min-height: 38px !important;
                margin-right: 0.46rem !important;
                border: 0 !important;
                border-radius: 12px !important;
                background: var(--dora-accent) !important;
                color: #ffffff !important;
                box-shadow: 0 10px 22px rgba(197, 22, 46, 0.26);
                transition: background 160ms ease, transform 160ms ease, box-shadow 160ms ease;
            }}

            [data-testid="stChatInput"] button:hover {{
                background: var(--dora-accent-dark) !important;
                transform: translateY(-1px);
                box-shadow: 0 12px 26px rgba(197, 22, 46, 0.32);
            }}

            [data-testid="stChatInput"] button svg {{
                width: 18px !important;
                height: 18px !important;
            }}

            .stButton > button {{
                width: 100%;
                min-height: 2.8rem;
                border-radius: 10px;
                border: 1px solid var(--dora-border);
                color: var(--dora-text);
                background: var(--dora-panel);
                font-weight: 650;
            }}

            .stButton > button:hover {{
                border-color: var(--dora-accent);
                background: var(--dora-panel-soft);
                color: var(--dora-accent);
            }}

            div[data-testid="stExpander"] {{
                border-color: var(--dora-border);
                background: var(--dora-panel);
                border-radius: 12px;
            }}

            .dora-history-item {{
                padding: 0.55rem 0.65rem;
                margin-bottom: 0.45rem;
                border-radius: 8px;
                border: 1px solid var(--dora-border);
                background: var(--dora-panel-soft);
                color: var(--dora-muted);
                font-size: 0.86rem;
            }}

            @media (max-width: 760px) {{
                .block-container {{
                    padding-left: 1rem;
                    padding-right: 1rem;
                }}

                .dora-welcome-inner {{
                    grid-template-columns: 1fr;
                    text-align: center;
                }}

                .dora-hero-logo {{
                    width: 104px;
                    height: 104px;
                    margin: 0 auto;
                    border-radius: 24px;
                }}

                .dora-eyebrow {{
                    margin-left: auto;
                    margin-right: auto;
                }}

                .dora-welcome h1 {{
                    font-size: 2.2rem;
                }}

                .dora-card-grid {{
                    grid-template-columns: 1fr;
                }}

                [data-testid="stChatInput"] {{
                    width: calc(100vw - 1rem);
                    margin-bottom: 0.35rem;
                }}

                [data-testid="stChatInput"] > div {{
                    min-height: 54px;
                    border-radius: 16px;
                }}

                [data-testid="stChatInput"] textarea {{
                    min-height: 44px !important;
                    padding-left: 0.9rem !important;
                    font-size: 0.94rem !important;
                }}

                [data-testid="stChatInput"] button {{
                    width: 36px !important;
                    height: 36px !important;
                    min-width: 36px !important;
                    min-height: 36px !important;
                    border-radius: 11px !important;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_query(value):
    if isinstance(value, list):
        return value[0] if value else ""
    if value is None:
        return ""
    return str(value).strip()


def clean_query_list(queries):
    return [clean_query(query) for query in queries if clean_query(query)]


def looks_like_agreement_query(query):
    folded = str(query or "").casefold()
    replacements = str.maketrans({
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
    folded = folded.translate(replacements)
    folded = "".join(
        char for char in unicodedata.normalize("NFKD", folded) if not unicodedata.combining(char)
    )
    return any(
        signal in folded
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
        )
    )


def dedupe_chunks(chunks, limit=8):
    seen = set()
    unique_chunks = []

    for chunk in sorted(chunks, key=lambda item: item.get("score", 0), reverse=True):
        text = chunk.get("text", "")
        key = " ".join(text[:500].lower().split())

        if not text or key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

        if len(unique_chunks) >= limit:
            break

    return unique_chunks


def retrieve_for_question(question, smart_question):
    queries = [smart_question]
    if question.lower() != smart_question.lower():
        queries.append(question)

    queries = clean_query_list(queries)
    detected_category = next((detect_scope_category(query) for query in queries if detect_scope_category(query)), None)
    if not detected_category:
        return [], queries, []

    all_chunks = []
    errors = []
    if detected_category == "ogretim_gorevlileri":
        primary_top_k = 12
    elif detected_category == "erasmus" and looks_like_agreement_query(smart_question):
        primary_top_k = 20
    else:
        primary_top_k = 6
    unique_limit = 20 if primary_top_k > 12 else 8

    for index, query in enumerate(queries):
        try:
            all_chunks.extend(retrieve(query, top_k=primary_top_k if index == 0 else 3))
        except Exception as exc:
            errors.append(str(exc))

    unique_chunks = dedupe_chunks(all_chunks, limit=unique_limit)

    if len(unique_chunks) >= 3:
        return unique_chunks, queries, errors

    fallback_queries = [
        query
        for query in generate_multi_queries(smart_question)
        if query.lower() not in {item.lower() for item in queries}
    ]

    for query in fallback_queries[:2]:
        try:
            all_chunks.extend(retrieve(query, top_k=3, candidate_k=16))
            queries.append(query)
        except Exception as exc:
            errors.append(str(exc))

    return dedupe_chunks(all_chunks, limit=unique_limit), queries, errors


def render_sources(chunks):
    sources = get_sources(chunks)
    if not sources:
        return

    with st.expander("Kaynaklar"):
        for source in sources[:8]:
            st.caption(source)


def render_welcome():
    logo_src = get_logo_src()
    logo_markup = (
        f'<div class="dora-hero-logo"><img src="{logo_src}" alt="Doğuş Üniversitesi logosu"></div>'
        if logo_src
        else '<div class="dora-hero-logo">DORA</div>'
    )
    st.markdown(
        f"""
        <div class="dora-welcome">
            <div class="dora-welcome-inner">
                {logo_markup}
                <div>
                    <div class="dora-eyebrow">Doğuş Üniversitesi asistanı</div>
                    <h1>Hoş geldin, nasıl yardımcı olabilirim?</h1>
                    <p>
                        Staj, Erasmus, ÇAP/yandal ve öğretim görevlileriyle ilgili sorularını
                        kaynaklara dayanarak yanıtlayabilirim.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards_markup = []
    for item in SUGGESTED_PROMPTS:
        title = html.escape(item["title"])
        body = html.escape(item["body"])
        prompt = html.escape(item["prompt"])
        suggested_id = html.escape(item["id"])
        cards_markup.append(
            f"""
            <a class="dora-card-link" href="?suggested={suggested_id}">
                <div class="dora-card">
                    <div class="dora-card-title">{title}</div>
                    <div class="dora-card-body">{body}</div>
                    <div class="dora-card-prompt">{prompt}</div>
                </div>
            </a>
            """
        )

    st.html(
        f'<div class="dora-card-grid">{"".join(cards_markup)}</div>',
    )

    st.markdown(
        """
        <div class="dora-scope">
            Kapsam dışı konularda bilgi uydurmak yerine sınırımı açıkça söylerim.
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_retrieval_ready():
    if st.session_state.get("retrieval_ready"):
        return

    with st.spinner("DORA kaynak motoru ve cevap modeli hazırlanıyor... İlk açılış biraz sürebilir."):
        warmup_retrieval_stack()
        warmup_answer_model()

    st.session_state.retrieval_ready = True


if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug" not in st.session_state:
    st.session_state.debug = False

if "theme" not in st.session_state:
    st.session_state.theme = "Aydınlık"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if "retrieval_ready" not in st.session_state:
    st.session_state.retrieval_ready = False


suggested_id = clean_query(st.query_params.get("suggested"))
if suggested_id:
    suggested_prompt = next(
        (
            item["prompt"]
            for item in SUGGESTED_PROMPTS
            if item["id"] == suggested_id
        ),
        "",
    )
    if suggested_prompt:
        st.session_state.pending_prompt = suggested_prompt
    st.query_params.clear()
    st.rerun()


with st.sidebar:
    sidebar_logo_src = get_logo_src()
    if sidebar_logo_src:
        st.markdown(
            f"""
            <div class="dora-sidebar-logo">
                <img src="{sidebar_logo_src}" alt="Doğuş Üniversitesi logosu">
                <div>
                    <p class="dora-sidebar-title">Doğuş Üniversitesi</p>
                    <p class="dora-sidebar-caption">Staj, Erasmus, ÇAP/Yandal</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Doğuş Üniversitesi")
        st.caption("Staj, Erasmus, ÇAP/Yandal")

    st.session_state.theme = st.radio(
        "Tema",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        horizontal=True,
    )

    if st.button("Yeni sohbet"):
        st.session_state.messages = []
        st.session_state.pending_prompt = ""
        st.rerun()

    st.session_state.debug = st.toggle("Debug", value=st.session_state.debug)
    st.markdown("#### Son sorular")

    recent_user_messages = [
        message["content"]
        for message in st.session_state.messages
        if message["role"] == "user"
    ][-8:]

    if recent_user_messages:
        for message in reversed(recent_user_messages):
            safe_message = html.escape(message[:72])
            st.markdown(
                f'<div class="dora-history-item">{safe_message}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Henüz sohbet yok.")


inject_theme(st.session_state.theme)
ensure_retrieval_ready()

if not st.session_state.messages:
    render_welcome()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def handle_query(question):
    question = clean_query(question)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        smart_question = clean_query(
            condense_question(st.session_state.messages[:-1], question)
        )

        with st.spinner("Kaynaklar aranıyor..."):
            chunks, queries, errors = retrieve_for_question(question, smart_question)

        if st.session_state.debug:
            with st.expander("Debug"):
                st.markdown(f"**Orijinal soru:** {question}")
                st.markdown(f"**Arama sorusu:** {smart_question}")
                st.markdown(f"**Kullanılan aramalar:** {queries}")
                if errors:
                    st.json({"errors": errors})
                st.json(chunks)

        response = st.write_stream(
            generate_answer_stream(
                chunks,
                smart_question,
                st.session_state.messages,
            )
        )

        render_sources(chunks)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )


prompt = st.chat_input("Mesajını yaz...")
queued_prompt = st.session_state.pop("pending_prompt", "")

if queued_prompt:
    handle_query(queued_prompt)
elif prompt:
    handle_query(prompt)
