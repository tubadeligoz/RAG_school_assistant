import os
import sys
import streamlit as st
import time

# 🔥 PROJE ROOT AYARI
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.retriever import retrieve
# Generator'dan yeni akıllı fonksiyonları çekiyoruz
from src.pipeline.generator import (
    generate_answer_stream, 
    condense_question, 
    generate_multi_queries
)

# ==========================================
# 🎨 PREMIUM UI CONFIG (OLED DARK)
# ==========================================
st.set_page_config(page_title="DORA | AI Orchestra", page_icon="📟", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        .stApp { background-color: #000000 !important; color: #FFFFFF; font-family: 'Inter', sans-serif; }
        header[data-testid="stHeader"] { visibility: hidden; }
        [data-testid="stToolbar"] { visibility: hidden !important; }

        /* ── STICKY HEADER ── */
        .sticky-header {
            position: fixed; top: 0; left: 0; width: 100%; height: 65px;
            background: rgba(0, 0, 0, 0.8); backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            display: flex; align-items: center; padding: 0 40px; z-index: 99999;
        }
        .brand { font-family: 'Syne', sans-serif !important; font-size: 20px; font-weight: 800; letter-spacing: -1px; }

        /* ── SIDEBAR (HISTORY) ── */
        section[data-testid="stSidebar"] {
            background-color: #050505 !important;
            border-right: 1px solid #111 !important;
            width: 280px !important;
        }
        .history-card { 
            padding: 12px; border-radius: 10px; font-size: 13px; color: #666; 
            margin-bottom: 8px; transition: 0.2s; border: 1px solid transparent;
        }
        .history-card:hover { background: #111; color: #EEE; border-color: #222; }

        /* ── CHAT AREA ── */
        .block-container { 
            max-width: 850px !important; 
            padding-top: 100px !important; 
            padding-bottom: 160px !important;
            margin: 0 auto !important; 
        }
        
        [data-testid="stChatMessage"] { background: transparent !important; margin-bottom: 40px !important; }
        .stMarkdown p { font-size: 16px !important; line-height: 1.8; color: #D1D1D1; }

        /* ── SOURCE PILLS ── */
        .source-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 25px; }
        .source-link {
            text-decoration: none; background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;
            padding: 6px 14px; font-size: 11px; color: #888 !important; transition: 0.3s;
            font-family: 'JetBrains Mono', monospace !important;
        }
        .source-link:hover { background: #FFF; color: #000 !important; border-color: #FFF; }

        /* ── COMPACT INPUT (GEMINI STYLE) ── */
        .stChatInputContainer { 
            position: fixed !important; bottom: 35px !important; left: 50% !important;
            transform: translateX(-50%) !important; width: 90% !important; max-width: 750px !important;
            background: #0F0F0F !important; border: 1px solid #222 !important; border-radius: 28px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.9) !important;
        }
        
        /* Suggestion Grid */
        .s-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 40px; }
    </style>
    
    <div class="sticky-header">
        <div class="brand">DORA <span style="color:#555; font-weight:400; font-size:15px;">&nbsp;//&nbsp; Doğuş AI Orchestra</span></div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_list" not in st.session_state:
    st.session_state.history_list = ["Akademik Takvim", "Burs Yönetmeliği", "Erasmus Başvurusu"]

# ==========================================
# 📂 SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("＋ Yeni Sohbet", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("<p style='color:#333; font-size:10px; margin-top:40px; letter-spacing:2px; font-weight:600;'>GEÇMİŞ</p>", unsafe_allow_html=True)
    for chat in st.session_state.history_list:
        st.markdown(f'<div class="history-card">💬 {chat}</div>', unsafe_allow_html=True)

# ==========================================
# 📟 MAIN CHAT FLOW
# ==========================================
if not st.session_state.messages:
    # Karşılama Ekranı
    st.markdown("""
        <div style="margin-top:60px;">
            <h1 style="font-family:'Syne'; font-size:52px; font-weight:800; letter-spacing:-2.5px; background: linear-gradient(90deg, #FFF, #555); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Merhaba Tuba,</h1>
            <p style="color:#444; font-size:24px; margin-top:-10px; font-weight:500;">Bugün neyi simüle edelim?</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="s-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    cards = [
        ("🏛️ Kurumsal Yapı", "Üniversite vizyonu ve akademik kadro."),
        ("📚 Lisans Programları", "Bölümler ve eğitim içerikleri."),
        ("📍 Kampüs Rehberi", "Ulaşım ve sosyal imkanlar."),
        ("💳 Finansal Bilgi", "Burslar ve ödeme detayları.")
    ]
    for i, (t, d) in enumerate(cards):
        with (col1 if i % 2 == 0 else col2):
            if st.button(f"**{t}**\n\n{d}", key=f"btn_{i}"):
                st.session_state.pending = t
                st.rerun()
else:
    # Sohbet Geçmişi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("urls"):
                links_html = "".join([f'<a href="{url}" target="_blank" class="source-link">🔗 {url.split("/")[-1]}</a>' for url in msg["urls"]])
                st.markdown(f'<div class="source-container">{links_html}</div>', unsafe_allow_html=True)

# ==========================================
# 💬 AGENTIC PROCESSING (STRATEGIC RAG)
# ==========================================
def handle_query(q):
    # 1. Kullanıcı Mesajı
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)

    with st.chat_message("assistant"):
        # 2. DÜŞÜNME VE ANALİZ (Multi-Query Retrieval)
        unique_chunks = []
        urls = []
        
        with st.status("🧠 Stratejik tarama yapılıyor...", expanded=False) as status:
            # Önce soruyu geçmişe göre akıllandır
            smart_q = condense_question(st.session_state.messages[:-1], q)
            st.write(f"🔍 Analiz Edilen Sorgu: *{smart_q}*")
            
            # Soruyu 3 farklı varyasyona genişlet (Multi-Query)
            queries = generate_multi_queries(smart_q)
            st.write(f"🚀 Genişletilmiş Aramalar: {', '.join(queries)}")
            
            # Her sorgu için ayrı arama yap
            all_retrieved = []
            for sq in queries:
                all_retrieved.extend(retrieve(sq, final_k=3))
            
            # Mükerrer sonuçları temizle
            seen_text = set()
            for c in all_retrieved:
                if c['text'] not in seen_text:
                    unique_chunks.append(c)
                    seen_text.add(c['text'])
            
            urls = list(set([c.get('metadata', {}).get('url', 'https://www.dogus.edu.tr') for c in unique_chunks]))
            status.update(label="✅ Bilgi derinliği sağlandı.", state="complete")

        # 3. STREAMING GENERATION (Typewriter Effect)
        # st.write_stream, generator'dan gelen parçaları anlık ekrana basar
        full_response = st.write_stream(generate_answer_stream(unique_chunks, smart_q, history=st.session_state.messages))
        
        # Kaynak Butonları
        if urls:
            links_html = "".join([f'<a href="{url}" target="_blank" class="source-link">🔗 {url.split("/")[-1]}</a>' for url in urls])
            st.markdown(f'<div class="source-container">{links_html}</div>', unsafe_allow_html=True)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response, "urls": urls})

# Logic Triggers
if "pending" in st.session_state and st.session_state.pending:
    query = st.session_state.pending
    st.session_state.pending = None
    handle_query(query)
    st.rerun()

if prompt := st.chat_input("DORA'ya bir soru sor..."):
    handle_query(prompt)
    st.rerun()