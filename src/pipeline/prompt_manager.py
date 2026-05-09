from functools import lru_cache
from pathlib import Path

from src.pipeline.scope import prompt_name_for_category


BASE_DIR = Path(__file__).resolve().parents[2]
PROMPT_DIR = BASE_DIR / "prompts"


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists() and name != "answer_default":
        path = PROMPT_DIR / "answer_default.txt"

    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs) -> str:
    return load_prompt(name).format_map(SafeDict(kwargs))


def render_answer_prompt(
    context: str,
    question: str,
    category: str | None = None,
    history_text: str = "",
) -> str:
    return render_prompt(
        prompt_name_for_category(category),
        context=context,
        question=question,
        history_text=history_text,
    )
