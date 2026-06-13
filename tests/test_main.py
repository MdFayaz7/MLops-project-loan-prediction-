# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import patch
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app
from src.utils import load_config

client = TestClient(app)

def test_load_config():
    """Test that configuration is loaded successfully and contains expected keys."""
    config = load_config()
    assert isinstance(config, dict)
    assert "data" in config
    assert "artifacts" in config


@patch("app.main.run_prediction_pipeline")
@patch("app.main.log_prediction")
def test_predict_endpoint_approved(mock_log, mock_predict):
    """Test /predict endpoint when loan prediction returns 0 (approved)."""
    # Mocking prediction pipeline return: (prediction_val, model_version)
    mock_predict.return_value = (0, "v1.0.0")
    
    payload = {
        "credit_policy": 1.0,
        "purpose": "debt_consolidation",
        "int_rate": 0.1189,
        "installment": 829.1,
        "log_annual_inc": 11.3504,
        "dti": 19.48,
        "fico": 737.0,
        "days_with_cr_line": 5639.95,
        "revol_bal": 28854.0,
        "revol_util": 52.1,
        "inq_last_6mths": 0.0,
        "delinq_2yrs": 0.0,
        "pub_rec": 0.0
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["label"] == "Approved"
    assert data["model_version"] == "v1.0.0"
    
    # Verify the mocked function calls
    mock_predict.assert_called_once()
    mock_log.assert_called_once()


@patch("app.main.run_prediction_pipeline")
@patch("app.main.log_prediction")
def test_predict_endpoint_rejected(mock_log, mock_predict):
    """Test /predict endpoint when loan prediction returns 1 (rejected)."""
    mock_predict.return_value = (1, "v1.0.0")
    
    payload = {
        "credit_policy": 1.0,
        "purpose": "debt_consolidation",
        "int_rate": 0.1189,
        "installment": 829.1,
        "log_annual_inc": 11.3504,
        "dti": 19.48,
        "fico": 737.0,
        "days_with_cr_line": 5639.95,
        "revol_bal": 28854.0,
        "revol_util": 52.1,
        "inq_last_6mths": 0.0,
        "delinq_2yrs": 0.0,
        "pub_rec": 0.0
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["label"] == "Rejected"
    assert data["model_version"] == "v1.0.0"
