import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.ml.utils_weekly import (
    resample_weekly,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    check_last_week_completeness,
    get_feature_columns,
    calculate_forecast_metrics,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def daily_data_complete_weeks():
    """Data up to Sunday Jan 19 (exactly 2 full weeks + partial 3rd week)"""
    dates = pd.date_range("2025-01-06", "2025-01-19", freq="D")  # 14 days
    np.random.seed(42)
    net_sales = np.random.uniform(1000, 5000, len(dates))
    return pd.DataFrame({"date": dates, "net_sales": net_sales})


@pytest.fixture
def daily_data_incomplete_last_week():
    """Data up to Friday Jan 17 (missing Sat Jan 18 & Sun Jan 19)"""
    dates = pd.date_range("2025-01-06", "2025-01-17", freq="D")  # 12 days
    np.random.seed(42)
    net_sales = np.random.uniform(1000, 5000, len(dates))
    return pd.DataFrame({"date": dates, "net_sales": net_sales})


@pytest.fixture
def weekly_data():
    """25 weeks of data for reliable lag/rolling tests"""
    weeks = pd.date_range("2025-01-06", periods=25, freq="W-MON")
    np.random.seed(123)
    sales = np.random.uniform(25000, 45000, len(weeks))
    return pd.DataFrame({"week": weeks, "weekly_sales": sales})


# ============================================
# CORE UTILITY TESTS
# ============================================

def test_resample_weekly(daily_data_complete_weeks):
    df = daily_data_complete_weeks.copy()
    result = resample_weekly(df)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["week", "weekly_sales"]
    assert len(result) == 3  # 2 full + partial week starting Jan 20
    assert (result["week"].dt.weekday == 0).all()
    assert pytest.approx(result["weekly_sales"].sum()) == df["net_sales"].sum()


def test_add_calendar_features(weekly_data):
    result = add_calendar_features(weekly_data.copy())

    assert "weekofyear" in result.columns
    assert "week_sin" in result.columns
    assert result["weekofyear"].dtype in [np.int32, np.int64]
    assert (result["weekofyear"] >= 1).all() and (result["weekofyear"] <= 53).all()
    assert (result["week_sin"].abs() <= 1.0001).all()


def test_add_lag_features(weekly_data):
    result = add_lag_features(weekly_data.copy())

    assert "lag_2" in result.columns
    assert "lag_12" in result.columns
    assert pd.isna(result["lag_2"].iloc[0])
    assert pd.isna(result["lag_2"].iloc[1])
    assert not pd.isna(result["lag_2"].iloc[2])
    assert result["lag_2"].iloc[2] == weekly_data["weekly_sales"].iloc[0]


def test_add_rolling_features(weekly_data):
    result = add_rolling_features(weekly_data.copy())

    assert "roll_mean_4" in result.columns
    assert "roll_mean_8" in result.columns
    assert "roll_mean_12" in result.columns
    assert pd.isna(result["roll_mean_4"].iloc[:3]).all()
    assert not pd.isna(result["roll_mean_4"].iloc[3])


def test_check_last_week_completeness():
    """Test based on the actual function logic: checks for Saturday, not full week"""
    
    # Test 1: Ends on Sunday (has Saturday) → False
    dates_sunday = pd.date_range("2025-01-13", "2025-01-19", freq="D")
    df_sunday = pd.DataFrame({"date": dates_sunday, "net_sales": np.ones(7)})
    weekly_sunday = pd.DataFrame({"week": [pd.Timestamp("2025-01-13")], "weekly_sales": [7.0]})
    
    assert check_last_week_completeness(df_sunday, weekly_sunday) is False
    
    # Test 2: Ends on Saturday (has Saturday) → False
    dates_saturday = pd.date_range("2025-01-13", "2025-01-18", freq="D")
    df_saturday = pd.DataFrame({"date": dates_saturday, "net_sales": np.ones(6)})
    weekly_saturday = pd.DataFrame({"week": [pd.Timestamp("2025-01-13")], "weekly_sales": [6.0]})
    
    assert check_last_week_completeness(df_saturday, weekly_saturday) is False
    
    # Test 3: Ends on Friday (missing Saturday) → True
    dates_friday = pd.date_range("2025-01-13", "2025-01-17", freq="D")
    df_friday = pd.DataFrame({"date": dates_friday, "net_sales": np.ones(5)})
    weekly_friday = pd.DataFrame({"week": [pd.Timestamp("2025-01-13")], "weekly_sales": [5.0]})
    
    assert check_last_week_completeness(df_friday, weekly_friday) is True


def test_get_feature_columns():
    features = get_feature_columns()
    expected = [
        "weekofyear", "week_sin",
        "lag_2", "lag_12",
        "roll_mean_4", "roll_mean_8", "roll_mean_12"
    ]
    assert sorted(features) == sorted(expected)


def test_calculate_forecast_metrics(weekly_data):
    y_true = weekly_data["weekly_sales"].iloc[:10].values
    y_pred = y_true * 1.05

    metrics = calculate_forecast_metrics(y_true, y_pred)

    assert "mape" in metrics and 4.9 < metrics["mape"] < 5.1
    assert "mae" in metrics and metrics["mae"] > 0
    assert "rmse" in metrics and metrics["rmse"] > 0
    assert metrics["n_samples"] == 10


def test_full_feature_pipeline(weekly_data):
    df = weekly_data.copy()

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df_clean = df.dropna()

    expected_cols = ["week", "weekly_sales"] + get_feature_columns()
    for col in expected_cols:
        assert col in df_clean.columns

    assert len(df_clean) < len(weekly_data)
    assert len(df_clean) >= 13


@pytest.fixture
def temp_dir():
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path)


def test_prediction_saving(temp_dir, weekly_data):
    future_weeks = pd.date_range("2025-07-07", periods=12, freq="W-MON")
    predictions = np.random.uniform(30000, 50000, 12)

    df_forecast = pd.DataFrame({
        "week": future_weeks,
        "predicted_profit": predictions
    })

    parquet_path = temp_dir / "forecast.parquet"
    csv_path = temp_dir / "forecast.csv"

    df_forecast.to_parquet(parquet_path, index=False)
    df_forecast.to_csv(csv_path, index=False)

    assert parquet_path.exists()
    assert csv_path.exists()

    loaded = pd.read_parquet(parquet_path)
    assert len(loaded) == 12
    assert "predicted_profit" in loaded.columns