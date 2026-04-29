import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")

# =========================
# MODELS
# =========================
EMBED_MODEL = "intfloat/multilingual-e5-base"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# =========================
# CHROMA
# =========================
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="uni_rag")


# =========================
# RETRIEVE
# =========================
def retrieve(query: str, top_k: int = 5):
    if not query or not query.strip():
        return []

    # e5 format (ÇOK ÖNEMLİ)
    query_emb = embedder.encode(f"query: {query}").tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=20,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # =========================
    # deduplicate (çok kritik)
    # =========================
    seen = set()
    candidates = []

    for doc, meta in zip(docs, metas):
        key = doc[:200]  # full hash yerine stabil truncation
        if key not in seen:
            seen.add(key)
            candidates.append((doc, meta))

    if not candidates:
        return []

    # =========================
    # rerank
    # =========================
    pairs = [[query, doc] for doc, _ in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    # =========================
    # final output
    # =========================
    output = []
    for score, (doc, meta) in ranked[:top_k]:
        output.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "metadata": meta,
            "score": float(score)
        })

    return output


# =========================
# CONTEXT BUILDER (CLEAN)
# =========================
def build_context(chunks):
    if not chunks:
        return ""

    parts = []
    for i, c in enumerate(chunks):
        source = c.get("source", "unknown")

        # 🔥 kritik: çok uzun context LLM'i bozar
        text = c["text"][:1200]

        parts.append(
            f"[Source {i+1} | {source}]\n{text}"
        )

    return "\n\n---\n\n".join(parts)


# =========================
# SOURCES
# =========================
def get_sources(chunks):
    return list(set([c.get("source") for c in chunks if c.get("source")]))