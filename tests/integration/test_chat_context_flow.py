import pytest

from src.pipeline.generator import condense_question, generate_answer_stream
from src.pipeline.scope import compact_text


def run_direct_answer(chunks, question, history):
    smart_question = condense_question(history, question)
    answer = "".join(generate_answer_stream(chunks, smart_question, history))
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return smart_question, answer


@pytest.mark.integration
def test_staj_multi_turn_context_flow(staj_chunks):
    history = []

    first_smart, first_answer = run_direct_answer(staj_chunks, "staj icin gerekli evraklar neler?", history)
    second_smart, second_answer = run_direct_answer(staj_chunks, "kime teslim edecegim peki?", history)
    third_smart, third_answer = run_direct_answer(staj_chunks, "sorumlu egitim elamanini nereden bulabilirim?", history)

    assert first_smart == "staj icin gerekli evraklar neler?"
    assert compact_text(first_answer).find("staj basvuru ve kabul formu") >= 0
    assert compact_text(second_answer).find("sorumlu ogretim elemani") >= 0
    assert compact_text(third_answer).find("bolum/program baskani") >= 0


@pytest.mark.integration
def test_prompt_selection_for_non_direct_answer(monkeypatch, make_chunk):
    import src.pipeline.generator as generator

    captured = {}

    def fake_stream(prompt, temperature=0.1):
        captured["prompt"] = prompt
        yield "Erasmus cevabi yeterli metin"

    monkeypatch.setattr(generator, "_call_ollama_stream", fake_stream)

    chunks = [make_chunk("Erasmus basvuru ve hibe bilgisi.", category="erasmus", source="erasmus.json")]
    answer = "".join(generator.generate_answer_stream(chunks, "Erasmus basvurusu nasil yapilir?", []))

    assert answer == "Erasmus cevabi yeterli metin"
    assert "Erasmus" in captured["prompt"]
    assert "Erasmus basvuru" in captured["prompt"]


@pytest.mark.integration
def test_answer_prompt_uses_visible_user_question_and_history(monkeypatch, make_chunk):
    import src.pipeline.generator as generator

    captured = {}

    def fake_stream(prompt, temperature=0.1):
        captured["prompt"] = prompt
        yield "Takip cevabi yeterli metin"

    monkeypatch.setattr(generator, "_call_ollama_stream", fake_stream)

    chunks = [make_chunk("Erasmus sonuclari ve basvuru bilgisi.", category="erasmus", source="erasmus.json")]
    history = [
        {"role": "user", "content": "Erasmus basvurulari hakkinda bilgi verir misin?"},
        {"role": "assistant", "content": "Basvuru surecinde sinav ve sonuc adimlari var."},
        {"role": "user", "content": "Peki sonuclari nereden takip edecegim?"},
    ]

    answer = "".join(
        generator.generate_answer_stream(
            chunks,
            "Erasmus ve uluslararasi degisim: Peki sonuclari nereden takip edecegim?",
            history,
        )
    )

    assert answer == "Takip cevabi yeterli metin"
    assert "Peki sonuclari nereden takip edecegim?" in captured["prompt"]
    assert "Erasmus ve uluslararasi degisim: Peki" not in captured["prompt"]
    assert "Basvuru surecinde sinav" in captured["prompt"]
