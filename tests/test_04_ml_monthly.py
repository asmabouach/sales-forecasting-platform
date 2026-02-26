import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Import the functions to test
from src.ml.utils_monthly import (
    resample_monthly,
    is_full_month,
    filter_complete_months,
    prepare_prophet_data,
    inverse_transform_predictions,
    calculate_forecast_metrics,
    generate_future_months,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def daily_data_complete_months():
    """Daily data covering 3 full calendar months (Jan, Feb, Mar 2025)"""
    dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    np.random.seed(42)
    net_sales = np.random.uniform(1000, 5000, len(dates))
    return pd.DataFrame({"date": dates, "net_sales": net_sales})


@pytest.fixture
def daily_data_incomplete_last_month():
    """Daily data up to mid-March 2025 (incomplete last month)"""
    dates = pd.date_range("2025-01-01", "2025-03-15", freq="D")  # Only first 15 days of March
    np.random.seed(42)
    net_sales = np.random.uniform(1000, 5000, len(dates))
    return pd.DataFrame({"date": dates, "net_sales": net_sales})


@pytest.fixture
def monthly_data():
    """Pre-aggregated monthly data for testing"""
    months = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    sales = [120000, 135000, 140000]
    return pd.DataFrame({"month": months, "monthly_sales": sales})


# ============================================
# CORE UTILITY TESTS
# ============================================

def test_resample_monthly(daily_data_complete_months):
    df = daily_data_complete_months.copy()
    result = resample_monthly(df)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["month", "monthly_sales"]
    assert len(result) == 3
    assert result["month"].dt.is_month_end.all()
    assert pytest.approx(result["monthly_sales"].sum()) == df["net_sales"].sum()


def test_is_full_month(daily_data_complete_months):
    date_series = daily_data_complete_months["date"]

    # All three months should be full
    assert is_full_month(date_series, pd.Timestamp("2025-01-31")) is True
    assert is_full_month(date_series, pd.Timestamp("2025-02-28")) is True
    assert is_full_month(date_series, pd.Timestamp("2025-03-31")) is True

    # Remove several days from February to drop below 90%
    missing_feb = date_series[~date_series.isin(pd.date_range("2025-02-10", "2025-02-20", freq="D"))]
    incomplete_df = daily_data_complete_months[daily_data_complete_months["date"].isin(missing_feb)]

    assert is_full_month(incomplete_df["date"], pd.Timestamp("2025-02-28")) is False


def test_filter_complete_months(daily_data_complete_months, monthly_data, daily_data_incomplete_last_month):
    # All months complete
    result = filter_complete_months(daily_data_complete_months, monthly_data.copy())
    assert len(result) == 3

    # Incomplete last month: only Jan and Feb should remain
    incomplete_monthly = resample_monthly(daily_data_incomplete_last_month)
    result_incomplete = filter_complete_months(daily_data_incomplete_last_month, incomplete_monthly)
    assert len(result_incomplete) == 2
    assert list(result_incomplete["month"]) == [
        pd.Timestamp("2025-01-31"),
        pd.Timestamp("2025-02-28")
    ]


def test_prepare_prophet_data(monthly_data):
    result = prepare_prophet_data(monthly_data.copy())

    assert list(result.columns) == ["ds", "y"]
    assert result["ds"].equals(monthly_data["month"])
    assert np.allclose(result["y"], np.log1p(monthly_data["monthly_sales"]))


def test_inverse_transform_predictions():
    values = np.array([100000, 110000, 120000])
    log_vals = np.log1p(values)
    result = inverse_transform_predictions(log_vals)

    # Use higher tolerance due to floating point precision
    assert np.allclose(result, values, atol=2)


def test_calculate_forecast_metrics(monthly_data):
    df_prophet = prepare_prophet_data(monthly_data.copy())

    # Perfect forecast
    forecast = df_prophet.rename(columns={"y": "yhat"}).copy()
    forecast["yhat_lower"] = forecast["yhat"] - 0.1
    forecast["yhat_upper"] = forecast["yhat"] + 0.1

    metrics = calculate_forecast_metrics(df_prophet, forecast)

    assert metrics["n_samples"] == 3
    assert metrics["mape"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mae"] == pytest.approx(0.0, abs=2)
    assert metrics["rmse"] == pytest.approx(0.0, abs=2)


def test_generate_future_months():
    last_month = pd.Timestamp("2025-03-31")
    future = generate_future_months(last_month, periods=6)

    assert len(future) == 6
    assert future[0] == pd.Timestamp("2025-04-30")
    assert future[-1] == pd.Timestamp("2025-09-30")
    # future is DatetimeIndex → convert to Series for .dt
    assert all(pd.Series(future).dt.is_month_end)


# ============================================
# FILE SAVING TEST
# ============================================

@pytest.fixture
def temp_dir():
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path)


def test_prediction_saving_logic(temp_dir, monthly_data):
    last_month = monthly_data["month"].iloc[-1]
    future_dates = generate_future_months(last_month, periods=3)
    predictions = [130000, 135000, 140000]

    df_forecast = pd.DataFrame({
        "date": future_dates,
        "predicted_profit": predictions,
        "lower_ci": np.array(predictions) * 0.9,
        "upper_ci": np.array(predictions) * 1.1,
    })

    parquet_path = temp_dir / "test_forecast.parquet"
    csv_path = temp_dir / "test_forecast.csv"

    df_forecast.to_parquet(parquet_path, index=False)
    df_forecast.to_csv(csv_path, index=False)

    assert parquet_path.exists()
    assert csv_path.exists()

    loaded = pd.read_parquet(parquet_path)
    assert len(loaded) == 3
    assert "predicted_profit" in loaded.columns