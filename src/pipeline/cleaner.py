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

    # Satır yapısını koru; chunker başlıkları ve listeleri bu sayede daha iyi ayırır.
    clean_text = re.sub(r"[ \t]+", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    return title, clean_text.strip()


def filename_to_url(filename: str) -> str:
    parts = filename.split("_")
    if parts[:4] == ["www", "dogus", "edu", "tr"]:
        suffix = "/".join(parts[4:])
        return "https://www.dogus.edu.tr" + (f"/{suffix}" if suffix else "")

    return filename.replace("_", "/")


def detect_category(filename: str, title: str) -> str:
    text = f"{filename} {title}".lower()

    if "aday" in text or "program" in text or "fakulte" in text or "fakülte" in text:
        return "programlar"

    if "ogrenci" in text or "öğrenci" in text:
        return "ogrenci"

    if "uluslararasi" in text or "international" in text:
        return "uluslararasi"

    if "akademik" in text:
        return "akademik"

    return "web"

# =========================
# DOSYA İŞLEYİCİ
# =========================
def clean_file(input_path, output_path):
    if os.path.getsize(input_path) == 0:
        print(f"[CLEANER] Skipped empty file: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    if len(html_content.strip()) < 20:
        print(f"[CLEANER] Skipped short content: {input_path}")
        return

    title, clean_md = clean_html_to_markdown(html_content)

    # 🔴 GLOBAL KALİTE KONTROL
    if len(clean_md.split()) < 80:
        print("[CLEANER] Skipped insufficient content")
        return

    if info_density(clean_md) < 0.3:
        print("[CLEANER] Skipped low information density")
        return

    filename = os.path.basename(input_path).replace(".txt", "")
    url = filename_to_url(filename)

    # 🔥 TITLE BOOST (çok önemli)
    final_content = f"# {title}\n\n{clean_md}" if title else clean_md

    cleaned_data = {
        "url": url,
        "title": title,
        "category": detect_category(filename, title),
        "content": final_content
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"[CLEANER] Cleaned: {filename}")

# =========================
# MAIN
# =========================
def run_cleaning():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]

    if not files:
        print("[CLEANER] No raw data found")
        return

    print("\n[CLEANER] Started\n")

    for file in files:
        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(CLEAN_DIR, f"{file.replace('.txt', '')}_clean.json")

        print(f"[CLEANER] Processing: {file}")
        clean_file(input_path, output_path)

    print("\n[CLEANER] Done\n")

if __name__ == "__main__":
    run_cleaning()
