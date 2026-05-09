import pytest

from src.pipeline.generator import condense_question, generate_answer_stream
from src.pipeline.scope import compact_text


def ask_with_chunks(history, chunks, question):
    smart_question = condense_question(history, question)
    answer = "".join(generate_answer_stream(chunks, smart_question, history))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return smart_question, answer


@pytest.mark.beta
def test_beta_staj_followup_acceptance_flow(staj_chunks):
    history = []
    _, answer_1 = ask_with_chunks(history, staj_chunks, "staj icin gerekli evraklar neler?")
    _, answer_2 = ask_with_chunks(history, staj_chunks, "kime teslim edecegim peki?")
    _, answer_3 = ask_with_chunks(history, staj_chunks, "sorumlu egitim elamanini nereden bulabilirim?")

    combined = compact_text("\n".join([answer_1, answer_2, answer_3]))
    assert "staj basvuru ve kabul formu" in combined
    assert "sorumlu ogretim elemani" in combined
    assert "bolum/program baskani" in combined


@pytest.mark.beta
def test_beta_scope_boundary_acceptance():
    answer = "".join(generate_answer_stream([], "okul ucretimi nereye yatiracagim?", []))
    folded = compact_text(answer)

    assert "yalnizca" in folded
    assert "staj" in folded
    assert "erasmus" in folded
