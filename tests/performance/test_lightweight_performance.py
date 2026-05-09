import time

import pytest

from src.pipeline.generator import generate_answer_stream
from src.pipeline.prompt_manager import render_answer_prompt
from src.pipeline.scope import detect_scope_category


@pytest.mark.performance
def test_intent_detection_latency_budget():
    queries = [
        "staj icin gerekli evraklar neler?",
        "erasmus basvurulari ne zaman?",
        "cap sartlari nelerdir?",
        "ogretim gorevlileri kimler?",
        "final sinavlari ne zaman?",
    ]

    start = time.perf_counter()
    for _ in range(300):
        for query in queries:
            detect_scope_category(query)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0


@pytest.mark.performance
def test_prompt_rendering_latency_budget():
    start = time.perf_counter()
    for _ in range(300):
        render_answer_prompt("baglam", "soru", "staj")
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0


@pytest.mark.performance
def test_direct_staj_answer_latency_budget(staj_chunks):
    start = time.perf_counter()
    for _ in range(100):
        "".join(generate_answer_stream(staj_chunks, "sorumlu egitim elamanini nereden bulabilirim?", []))
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
