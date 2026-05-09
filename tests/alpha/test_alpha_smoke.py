from pathlib import Path

import pytest


@pytest.mark.alpha
def test_core_modules_import():
    import src.pipeline.generator  # noqa: F401
    import src.pipeline.prompt_manager  # noqa: F401
    import src.pipeline.retriever  # noqa: F401
    import src.pipeline.scope  # noqa: F401


@pytest.mark.alpha
def test_prompt_files_exist():
    prompt_dir = Path("prompts")
    expected = {
        "answer_default.txt",
        "answer_staj.txt",
        "answer_erasmus.txt",
        "answer_cift_anadal_yandal.txt",
        "answer_ogretim_gorevlileri.txt",
        "condense_question.txt",
        "multi_query.txt",
        "followup.txt",
    }

    missing = [name for name in expected if not (prompt_dir / name).exists()]
    assert not missing


@pytest.mark.alpha
def test_processed_chunks_are_scope_limited_when_present():
    import json

    path = Path("data/processed/unified_chunks.json")
    if not path.exists():
        pytest.skip("processed chunks are not generated yet")

    chunks = json.loads(path.read_text(encoding="utf-8"))
    categories = {item.get("metadata", {}).get("category") for item in chunks}
    assert categories <= {"staj", "erasmus", "cift_anadal_yandal", "ogretim_gorevlileri"}
