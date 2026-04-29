import json
import os
import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md

RAW_DIR = "data/raw"
CLEAN_DIR = "data/processed/cleaned"

# =========================
# LOW VALUE SATIR FİLTRESİ
# =========================
LOW_VALUE_PATTERNS = [
    r"tümünü görüntüle",
    r"devamını oku",
    r"anasayfa",
    r"menu",
    r"haber",
    r"duyuru",
    r"etkinlik",
    r"copyright",
]

def is_low_value(line: str) -> bool:
    line = line.lower()
    return any(re.search(p, line) for p in LOW_VALUE_PATTERNS)

# =========================
# INFO DENSITY
# =========================
def info_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0
    return len(set(words)) / len(words)

# =========================
# HTML → MARKDOWN
# =========================
def clean_html_to_markdown(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # 🔥 TITLE
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # 🔥 ANA CONTENT BUL
    main = soup.find("main") or soup.find("article")

    if not main:
        main = soup.find("div", {"id": "content"}) or soup.find("div", {"class": "content"})

    if main:
        soup = main

    # 🔥 TAG TEMİZLİĞİ
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside", "button"]):
        tag.decompose()

    # 🔥 LINK TEXT KORU
    for a in soup.find_all("a"):
        a.replace_with(a.get_text())

    # 🔥 MARKDOWN
    markdown_text = md(
        str(soup),
        heading_style="ATX",
        strip=["img"]
    )

    # 🔥 SATIR TEMİZLİĞİ
    lines = markdown_text.split("\n")

    cleaned_lines = []
    seen = set()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # low value satır at
        if is_low_value(line):
            continue

        # duplicate satır at
        if line in seen:
            continue

        seen.add(line)
        cleaned_lines.append(line)

    clean_text = "\n".join(cleaned_lines)

    # 🔥 GLOBAL CLEAN
    clean_text = re.sub(r"\n{2,}", "\n", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)

    return title, clean_text.strip()

# =========================
# DOSYA İŞLEYİCİ
# =========================
def clean_file(input_path, output_path):
    if os.path.getsize(input_path) == 0:
        print(f"⚠️ [SKIP] boş dosya: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    if len(html_content.strip()) < 20:
        print(f"⚠️ [SKIP] çok kısa içerik: {input_path}")
        return

    title, clean_md = clean_html_to_markdown(html_content)

    # 🔴 GLOBAL KALİTE KONTROL
    if len(clean_md.split()) < 80:
        print(f"⚠️ [SKIP] içerik yetersiz")
        return

    if info_density(clean_md) < 0.3:
        print(f"⚠️ [SKIP] düşük bilgi yoğunluğu")
        return

    filename = os.path.basename(input_path).replace(".txt", "")
    url = filename.replace("_", "/")

    # 🔥 TITLE BOOST (çok önemli)
    final_content = f"# {title}\n\n{clean_md}" if title else clean_md

    cleaned_data = {
        "url": url,
        "title": title,
        "content": final_content
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ CLEANED: {filename}")

# =========================
# MAIN
# =========================
def run_cleaning():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]

    if not files:
        print("❌ RAW klasöründe veri yok")
        return

    print("\n🧹 CLEANING STARTED\n")

    for file in files:
        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(CLEAN_DIR, f"{file.replace('.txt', '')}_clean.json")

        print(f"🔄 {file}")
        clean_file(input_path, output_path)

    print("\n✅ CLEANING DONE\n")

if __name__ == "__main__":
    run_cleaning()