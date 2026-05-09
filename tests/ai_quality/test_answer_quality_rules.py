import pytest

from src.pipeline.generator import generate_answer_stream
from src.pipeline.scope import compact_text


def answer_text(chunks, question):
    return "".join(generate_answer_stream(chunks, question, []))


@pytest.mark.ai_quality
def test_answer_does_not_invent_responsible_teacher_name(staj_chunks):
    answer = answer_text(staj_chunks, "sorumlu egitim elamanini nereden bulabilirim?")
    folded = compact_text(answer)

    assert "isim uydurmuyorum" in folded
    assert "bolum/program baskani" in folded


@pytest.mark.ai_quality
def test_out_of_scope_answer_uses_scope_message():
    answer = answer_text([], "final sinav tarihleri ne zaman?")
    folded = compact_text(answer)

    assert "yalnizca" in folded
    assert "staj" in folded
    assert "final sinav tarihleri" not in folded


@pytest.mark.ai_quality
def test_empty_in_scope_context_does_not_hallucinate():
    answer = answer_text([], "staj ucreti var mi?")
    folded = compact_text(answer)

    assert "net bilgi bulunamadi" in folded
