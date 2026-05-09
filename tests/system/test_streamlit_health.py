import os

import pytest
import requests


@pytest.mark.system
def test_streamlit_health_endpoint_when_enabled():
    if os.getenv("DORA_RUN_SYSTEM_TESTS") != "1":
        pytest.skip("Set DORA_RUN_SYSTEM_TESTS=1 to run live Streamlit checks")

    response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
    assert response.status_code == 200
    assert response.text.strip() == "ok"
