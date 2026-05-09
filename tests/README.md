# Test Suite

Bu klasor DORA/RAG uygulamasi icin farkli test katmanlarini ayirir.

## Hizli testler

```powershell
python -m pytest
```

## Kategori bazli calistirma

```powershell
python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest tests/rag_quality
python -m pytest tests/ai_quality
python -m pytest tests/security
```

## Opsiyonel canli testler

Bu testler Streamlit, Chroma veya model sureclerini kullanabilir.

```powershell
$env:DORA_RUN_SYSTEM_TESTS="1"; python -m pytest tests/system
$env:DORA_RUN_LIVE_RAG="1"; python -m pytest tests/rag_quality -m slow
```

## Klasorler

- `alpha`: erken smoke ve kurulum kontrolleri
- `unit`: fonksiyon seviyesinde hizli testler
- `integration`: moduller arasi akisi mock'larla test eder
- `system`: canli uygulama saglik kontrolleri
- `performance`: hafif performans butceleri
- `rag_quality`: retrieval ve grounding kalitesi
- `ai_quality`: cevap kalitesi ve halusinasyon guard'lari
- `beta`: kullanici kabul senaryolari
- `security`: prompt injection ve veri siniri testleri
