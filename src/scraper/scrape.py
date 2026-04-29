import sys
import os
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# =========================
# PATH
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from urls import URLS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# =========================
# LOW VALUE FILTER
# =========================
LOW_VALUE_KEYWORDS = [
    "etkinlik",
    "duyuru",
    "haber",
    "tüm etkinlikler",
    "tüm duyurular",
    "programları keşfet",
    "anasayfa"
]

def is_low_value(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in LOW_VALUE_KEYWORDS)

# =========================
# INFO DENSITY (ÇOK KRİTİK)
# =========================
def info_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0
    return len(set(words)) / len(words)

# =========================
# CLEAN TEXT
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)

    noise_patterns = [
        r"cookie.*?",
        r"tüm hakları saklıdır",
        r"gizlilik politikası",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return text.strip()

# =========================
# HTML EXTRACT (SMART)
# =========================
def extract_main_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # script vs sil
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # linkleri koru (ÇOK ÖNEMLİ)
    for a in soup.find_all("a"):
        a.replace_with(a.get_text())

    # öncelik main
    main = soup.find("main") or soup.find("article")

    if main:
        text = main.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    return text

# =========================
# SCRAPE FUNCTION
# =========================
def scrape_url(page, url: str):
    print(f"[SCRAPING] {url}")

    try:
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        html = page.content()
        text = extract_main_content(html)

        # fallback
        if len(text) < 200:
            print("[FALLBACK] inner_text")
            text = page.inner_text("body")

        text = clean_text(text)

        # 🔴 LOW VALUE FILTER
        if is_low_value(text):
            print(f"[SKIP LOW VALUE] {url}")
            return

        # 🔴 LENGTH CHECK
        if len(text.split()) < 80:
            print(f"[SKIP SHORT] {url}")
            return

        # 🔴 INFO DENSITY CHECK
        if info_density(text) < 0.3:
            print(f"[SKIP LOW DENSITY] {url}")
            return

        # 🔴 DUPLICATE LINE CLEAN
        lines = text.split(". ")
        unique_lines = []
        seen = set()

        for line in lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)

        text = ". ".join(unique_lines)

        # filename
        filename = (
            url.replace("https://", "")
               .replace("http://", "")
               .replace("/", "_")
               .replace(".", "_")
        )

        file_path = os.path.join(RAW_DIR, f"{filename}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[SAVED] {filename}.txt ({len(text)} chars)")

    except Exception as e:
        print(f"[ERROR] {url} -> {e}")

# =========================
# MAIN
# =========================
def run_scraper():
    print("\n🚀 RAG SCRAPER STARTED\n")

    if not URLS:
        print("❌ URL listesi boş")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        )

        page = context.new_page()

        for url in URLS:
            scrape_url(page, url)

        browser.close()

    print("\n✅ SCRAPING COMPLETED\n")

if __name__ == "__main__":
    run_scraper()