import pytest

from src.pipeline.prompt_manager import load_prompt


@pytest.mark.ai_quality
def test_answer_prompts_require_context_grounding():
    for name in [
        "answer_default",
        "answer_staj",
        "answer_erasmus",
        "answer_cift_anadal_yandal",
        "answer_ogretim_gorevlileri",
    ]:
        prompt = load_prompt(name).casefold()
        assert "bağlam" in prompt or "baglam" in prompt
        assert "uydurma" in prompt or "net bilgi" in prompt
