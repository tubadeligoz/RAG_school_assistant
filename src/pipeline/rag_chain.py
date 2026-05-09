from src.pipeline.generator import generate_answer
from src.pipeline.retriever import build_context, get_sources, retrieve


def ask(question: str):
    chunks = retrieve(question, top_k=6)
    context = build_context(chunks)
    answer = generate_answer(context, question)

    return {
        "answer": answer,
        "sources": get_sources(chunks),
        "chunks": chunks,
    }


if __name__ == "__main__":
    while True:
        query = input("\nSoru: ")
        if query.lower() in ("exit", "quit", "q"):
            break

        result = ask(query)
        print("\nCevap:\n", result["answer"])
        print("\nKaynaklar:", result["sources"])
