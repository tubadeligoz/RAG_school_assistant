import os

import pytest


@pytest.mark.system
@pytest.mark.slow
def test_live_rag_staj_smoke_when_enabled():
    if os.getenv("DORA_RUN_LIVE_RAG") != "1":
        pytest.skip("Set DORA_RUN_LIVE_RAG=1 to run live Chroma/model checks")

    from src.pipeline.retriever import retrieve

    chunks = retrieve("staj icin gerekli evraklar neler?", top_k=3)
    assert chunks
    assert chunks[0]["metadata"]["category"] == "staj"
