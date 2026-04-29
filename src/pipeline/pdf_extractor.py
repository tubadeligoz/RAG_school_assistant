import os
import json
import fitz  # PyMuPDF

PDF_DIR = "data/raw/pdf"
OUT_DIR = "data/processed/pdf"

os.makedirs(OUT_DIR, exist_ok=True)


def pdf_to_text(path):
    doc = fitz.open(path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    return text


def run():
    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not files:
        print("❌ PDF yok")
        return

    for file in files:
        path = os.path.join(PDF_DIR, file)

        text = pdf_to_text(path)

        if len(text.strip()) < 50:
            print(f"⚠️ çok boş PDF: {file}")
            continue

        out = {
            "url": file,
            "content": text
        }

        out_path = os.path.join(OUT_DIR, file.replace(".pdf", ".json"))

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"✅ PDF işlendi: {file}")


if __name__ == "__main__":
    run()