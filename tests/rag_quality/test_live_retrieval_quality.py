import os

import pytest


@pytest.mark.rag_quality
@pytest.mark.slow
def test_live_retrieval_top_category_when_enabled():
    if os.getenv("DORA_RUN_LIVE_RAG") != "1":
        pytest.skip("Set DORA_RUN_LIVE_RAG=1 to run live retrieval quality checks")

    from src.pipeline.retriever import retrieve

    cases = [
        ("staj icin gerekli evraklar neler?", "staj"),
        ("erasmus basvurulari ne zaman?", "erasmus"),
        ("cap sartlari nelerdir?", "cift_anadal_yandal"),
    ]

    for query, expected_category in cases:
        chunks = retrieve(query, top_k=3)
        assert chunks, query
        assert chunks[0]["metadata"]["category"] == expected_category
