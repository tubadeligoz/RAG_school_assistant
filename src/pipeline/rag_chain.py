from pipeline.retriever import retrieve, build_context, get_sources


def ask(question: str):
    # 1. Chunk'ları getir
    chunks = retrieve(question, k=5)

    # 2. Context string'e çevir
    context = build_context(chunks)

    # 3. LLM'e gönder
    from pipeline.generator import generate_answer
    answer = generate_answer(context, question)

    # 4. Kaynakları al
    sources = get_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,   # debug için
    }


if __name__ == "__main__":
    while True:
        q = input("\n❓ Soru: ")
        if q.lower() in ("exit", "quit", "q"):
            break
        result = ask(q)
        print("\n🤖 Cevap:\n", result["answer"])
        print("\n🔗 Kaynaklar:", result["sources"])