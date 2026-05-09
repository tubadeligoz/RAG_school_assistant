import pytest

from src.pipeline.retriever import build_context, get_sources


@pytest.mark.integration
def test_build_context_includes_metadata(make_chunk):
    chunks = [
        make_chunk("Staj metni", category="staj", source="staj.json"),
        make_chunk("Erasmus metni", category="erasmus", source="erasmus.json"),
    ]

    context = build_context(chunks)

    assert "[CATEGORY: staj]" in context
    assert "[SOURCE: staj.json]" in context
    assert "Erasmus metni" in context


@pytest.mark.integration
def test_get_sources_deduplicates(make_chunk):
    chunks = [
        make_chunk("A", source="same.json"),
        make_chunk("B", source="same.json"),
        make_chunk("C", source="other.json"),
    ]

    assert get_sources(chunks) == ["mock://same.json", "mock://other.json"]


@pytest.mark.integration
def test_get_sources_hides_local_file_paths():
    chunks = [
        {
            "text": "Staj metni",
            "metadata": {
                "category": "staj",
                "source": "staj-uygulamalari-usul-ve-esaslari.json",
                "title": "staj-uygulamalari-usul-ve-esaslari",
                "type": "pdf",
                "url": r"C:\Users\tubad\Desktop\RAG_school_assistant\data\raw\pdf\yonergeler\staj-uygulamalari-usul-ve-esaslari.pdf",
            },
            "score": 1.0,
        }
    ]

    assert get_sources(chunks) == ["Staj Uygulamaları Usul ve Esasları"]
