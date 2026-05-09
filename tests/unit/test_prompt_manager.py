import pytest

from src.pipeline.prompt_manager import load_prompt, render_answer_prompt, render_prompt


@pytest.mark.unit
def test_load_prompt_reads_prompt_file():
    prompt = load_prompt("answer_staj")
    assert "{context}" in prompt
    assert "{question}" in prompt


@pytest.mark.unit
def test_render_prompt_keeps_missing_variables_readable():
    rendered = render_prompt("answer_default", context="baglam", question="soru", unknown="x")
    assert "baglam" in rendered
    assert "soru" in rendered


@pytest.mark.unit
def test_answer_prompt_is_category_specific():
    rendered = render_answer_prompt("staj baglami", "staj sorusu", "staj")
    lowered = rendered.casefold()
    assert "staj" in lowered
    assert "staj baglami" in rendered
    assert "staj sorusu" in rendered


@pytest.mark.unit
def test_answer_prompt_includes_conversation_context():
    rendered = render_answer_prompt(
        "staj baglami",
        "kime teslim edecegim?",
        "staj",
        history_text="Kullanıcı: Staj evrakları neler?",
    )

    assert "SOHBET BAĞLAMI" in rendered
    assert "Staj evrakları neler" in rendered
