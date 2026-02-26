import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

from src.api.api import app

client = TestClient(app)


def test_home_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Sales Forecasting Drift API is running!"
    assert data["status"] == "healthy"


@patch("src.api.api.run_all_drift_checks")
def test_run_drift_checks_success(mock_run_all_drift_checks):
    """Test successful drift checks with numpy type conversion"""
    mock_reports = [
        {
            "model_name": "weekly_forecast",
            "drift_detected": np.bool_(True),
            "p_value": np.float64(0.0001),
            "statistic": np.float64(12.5),
            "features_with_drift": ["net_sales", "lag_2"],
        }
    ]
    mock_run_all_drift_checks.return_value = mock_reports

    response = client.post("/run-drift-checks")

    assert response.status_code == 200
    data = response.json()
    assert "reports" in data
    report = data["reports"][0]
    assert isinstance(report["drift_detected"], bool)
    assert report["drift_detected"] is True
    assert isinstance(report["p_value"], float)


@patch("src.api.api.run_all_drift_checks")
def test_run_drift_checks_error(mock_run_all_drift_checks):
    """Test proper 500 status code on exception"""
    mock_run_all_drift_checks.side_effect = Exception("Database connection failed")

    response = client.post("/run-drift-checks")

    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "Failed to run drift checks"
    assert "Database connection failed" in data["details"]