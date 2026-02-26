import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def resample_weekly(df):
    """Resample to weekly frequency (Monday as week start)"""
    logger.info("Resampling data to weekly frequency...")
    df_weekly = (
        df.resample("W-MON", on="date")
          .sum()
          .reset_index()
          .rename(columns={"date": "week", "net_sales": "weekly_sales"})
    )
    logger.info(f"Weekly resampling complete. Shape: {df_weekly.shape}")
    return df_weekly

def add_calendar_features(df):
    """Add week-level time-based features"""
    logger.info("Adding calendar features...")
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    logger.info("Calendar features added: weekofyear, week_sin")
    return df

def add_lag_features(df):
    """Add weekly lag features"""
    logger.info("Adding lag features...")
    lags = [2, 12]
    for lag in lags:
        df[f"lag_{lag}"] = df["weekly_sales"].shift(lag)
    logger.info(f"Lag features added: {[f'lag_{l}' for l in lags]}")
    return df

def add_rolling_features(df):
    """Add rolling window statistics"""
    logger.info("Adding rolling window features...")
    windows = [4, 8, 12]
    for w in windows:
        df[f"roll_mean_{w}"] = df["weekly_sales"].rolling(w).mean()
    logger.info(f"Rolling features added: {[f'roll_mean_{w}' for w in windows]}")
    return df

def check_last_week_completeness(df_cleaned, df_weekly):
    """Check if last week has all 7 days"""
    logger.info("Checking last week data completeness...")
    
    last_date = df_cleaned['date'].max()
    last_monday = last_date - pd.Timedelta(days=last_date.weekday())
    expected_saturday = last_monday + pd.Timedelta(days=5)
    
    last_week_start = df_weekly['week'].iloc[-1]
    last_week_dates = df_cleaned[
        (df_cleaned['date'] >= last_week_start) &
        (df_cleaned['date'] < last_week_start + pd.Timedelta(days=7))
    ]['date']
    
    if expected_saturday not in last_week_dates.values:
        logger.warning("Last week is incomplete (missing Saturday) - dropping last week from dataset")
        logger.info(f"Last date in data: {last_date}, Expected Saturday: {expected_saturday}")
        return True
    
    logger.info("Last week is complete (has Saturday data)")
    return False

def get_feature_columns():
    """Return list of feature columns for model prediction"""
    return [
        "weekofyear",
        "week_sin", 
        "lag_2", "lag_12",
        "roll_mean_4",
        "roll_mean_8",
        "roll_mean_12",
    ]

def calculate_forecast_metrics(y_true, y_pred, logger=None):
    """
    Calculate forecast accuracy metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    logger : logging.Logger, optional
        Logger instance for logging metrics
    
    Returns:
    --------
    dict: Dictionary containing metrics
    """
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out zeros to avoid division by zero in MAPE
    mask = y_true != 0
    
    if mask.sum() > 0:  # Only calculate MAPE if we have non-zero values
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    metrics = {
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(y_true)
    }
    
    if logger:
        logger.info(f"Metrics calculated on {len(y_true)} samples:")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  MAE: {mae:,.2f}")
        logger.info(f"  RMSE: {rmse:,.2f}")
    
    return metrics