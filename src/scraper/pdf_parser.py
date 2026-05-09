import os
import json
import pdfplumber

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PDF_DIR = os.path.join(BASE_DIR, "data", "raw", "pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "cleaned")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"

    except Exception as e:
        print(f"[ERROR] PDF okunamadı: {pdf_path} -> {e}")

    return text.strip()


def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path).replace(".pdf", "")

    print(f"[PDF] işleniyor: {filename}")

    text = extract_text_from_pdf(pdf_path)

    if len(text.split()) < 50:
        print(f"[SKIP] çok kısa: {filename}")
        return

    # 🔥 kategoriyi klasörden alıyoruz
    category = os.path.basename(os.path.dirname(pdf_path))

    data = {
        "url": pdf_path,
        "title": filename,
        "content": text,
        "category": category   # 🔥 EN ÖNEMLİ EKLEME
    }

    out_path = os.path.join(OUTPUT_DIR, f"{filename}_pdf.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {filename}_pdf.json")


def run_pdf_parser():
    print("\n[PDF PARSER] Basladi\n")

    for root, dirs, files in os.walk(PDF_DIR):
        for file in files:
            if file.endswith(".pdf"):
                process_pdf(os.path.join(root, file))

    print("\n[PDF PARSER] Tamamlandi\n")


if __name__ == "__main__":
    run_pdf_parser()
