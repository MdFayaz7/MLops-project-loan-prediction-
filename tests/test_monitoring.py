# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import patch
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app
from src.components.model_monitoring import ModelMonitoring

client = TestClient(app)

def test_model_monitoring_run():
    """Test that ModelMonitoring runs and creates the files."""
    monitoring = ModelMonitoring()
    # Check that paths are configured
    assert monitoring.reference_path.exists()
    assert monitoring.current_path.exists()
    
    # Run monitoring
    success = monitoring.run()
    assert success is True
    assert monitoring.report_html_path.exists()
    assert monitoring.metrics_json_path.exists()


def test_monitoring_endpoint():
    """Test that the /monitoring endpoint returns the report successfully."""
    response = client.get("/monitoring?refresh=true")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Evidently" in response.text or "drift" in response.text.lower()
