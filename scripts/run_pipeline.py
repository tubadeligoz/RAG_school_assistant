import subprocess
import sys
import os

# =========================
# PROJECT ROOT
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# =========================
# RUN STEP HELPER
# =========================
def run_step(name, command):
    print(f"\n🚀 {name} BAŞLIYOR...\n" + "-" * 50)

    result = subprocess.run(command, shell=True, cwd=BASE_DIR)

    if result.returncode != 0:
        print(f"\n❌ {name} HATA VERDİ! Pipeline durduruldu.")
        sys.exit(1)

    print(f"\n✅ {name} TAMAMLANDI\n")


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("\n🔥 RAG PIPELINE STARTED 🔥\n")

    # 1. SCRAPER
    run_step(
        "1. SCRAPER (Web + PDF URL Detection)",
        "python src/scraper/scrape.py"
    )

    # 2. HTML CLEANER
    run_step(
        "2. HTML CLEANER (HTML → JSON)",
        "python src/pipeline/cleaner.py"
    )

    # 3. PDF EXTRACTOR (NEW STEP)
    run_step(
        "3. PDF EXTRACTOR (PDF → TEXT → JSON)",
        "python src/pipeline/pdf_extractor.py"
    )

    # 4. CHUNKER (UNIFIED)
    run_step(
        "4. CHUNKER (HTML + PDF → Chunks)",
        "python src/pipeline/chunker.py"
    )

    # 5. VECTORIZE
    run_step(
        "5. VECTORIZE (Embedding → ChromaDB)",
        "python src/pipeline/vectorize.py"
    )

    print("\n🎉 PIPELINE SUCCESSFULLY COMPLETED 🎉\n")

    print("📌 Test etmek için:")
    print("👉 streamlit run src/app/streamlit_app.py\n")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()