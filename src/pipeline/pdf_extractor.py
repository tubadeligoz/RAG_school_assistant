import os
import json
import fitz  # PyMuPDF

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PDF_DIR = os.path.join(BASE_DIR, "data", "raw", "pdf")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "pdf")

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# CLEAN TEXT
# =========================
def clean_text(text: str) -> str:
    lines = text.split("\n")

    cleaned = []
    for line in lines:
        line = line.strip()

        if not line:
            continue

        # sayfa numarası / footer / header noise
        if line.isdigit():
            continue

        if len(line) < 3:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# =========================
# TITLE
# =========================
def extract_title(file_path: str):
    return os.path.basename(file_path).replace(".pdf", "").replace("_", " ")


# =========================
# CATEGORY DETECTOR
# =========================
def detect_category(file_path: str):
    normalized_path = file_path.lower()
    parent = os.path.basename(os.path.dirname(file_path)).lower()
    filename = os.path.basename(file_path).lower()
    text = f"{parent} {filename} {normalized_path}"

    if "kvkk" in text or "kvk" in text or "kişisel-veri" in text:
        return "kvkk"

    if "staj" in text or "uygulamali-egitim" in text or "uygulamalı-eğitim" in text:
        return "staj"

    if "cift-anadal" in text or "çift-anadal" in text or "yandal" in text:
        return "cift_anadal_yandal"

    if "ucret" in text or "ücret" in text or "iade" in text or parent == "ucret_iade":
        return "ucret_iade"

    if "takvim" in text:
        return "akademik_takvim"

    if "yonetmelik" in text or "yönetmelik" in text or parent == "yonetmelikler":
        return "yonetmelikler"

    if "yonerge" in text or "yönerge" in text or parent == "yonergeler":
        return "yonergeler"

    return "pdf"


# =========================
# PDF TO TEXT (PAGE SAFE)
# =========================
def pdf_to_text(path: str) -> str:
    doc = fitz.open(path)

    pages_text = []

    for page in doc:
        text = page.get_text("text")
        if text:
            pages_text.append(text)

    return clean_text("\n".join(pages_text))


# =========================
# RECURSIVE FILE COLLECTOR
# =========================
def get_all_pdfs():
    pdf_files = []

    for root, _, files in os.walk(PDF_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    return pdf_files


# =========================
# MAIN
# =========================
def run():
    files = get_all_pdfs()

    if not files:
        print("[PDF] No PDF found")
        return

    print(f"\n[PDF] Total files: {len(files)}\n")

    for path in files:

        try:
            text = pdf_to_text(path)

            if len(text.split()) < 80:
                print(f"[PDF] Skipped short PDF: {path}")
                continue

            out = {
                "url": path,  # full path (çok önemli)
                "title": extract_title(path),
                "category": detect_category(path),
                "type": "pdf",
                "content": text
            }

            # output name safety
            safe_name = os.path.basename(path).replace(".pdf", "")
            out_path = os.path.join(OUT_DIR, f"{safe_name}.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            print(f"[PDF] Processed: {safe_name}")

        except Exception as e:
            print(f"[PDF] Error: {path} -> {e}")


if __name__ == "__main__":
    run()
