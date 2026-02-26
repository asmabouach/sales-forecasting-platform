"""
Monthly forecasting utility functions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import logging

logger = logging.getLogger(__name__)

def resample_monthly(df):
    """
    Resample to monthly frequency (calendar month-end)
    and aggregate by summing the net_sales.
    """
    logger.info("Resampling data to monthly frequency...")
    df_monthly = (
        df.resample("M", on="date")  # M = Month-End frequency
          .sum()
          .reset_index()
          .rename(columns={"date": "month", "net_sales": "monthly_sales"})
    )
    logger.info(f"Monthly resampling complete. Shape: {df_monthly.shape}")
    return df_monthly

def is_full_month(date_series, month_end_date):
    """
    Check if a month is complete by verifying we have data for all days.
    
    Args:
        date_series: Series of dates from the daily data
        month_end_date: The month-end date to check
        
    Returns:
        bool: True if month is complete, False otherwise
    """
    # Get year and month from month_end_date
    year = month_end_date.year
    month = month_end_date.month
    
    # Get number of days in that month
    days_in_month = calendar.monthrange(year, month)[1]
    
    # Create expected dates for the month
    expected_dates = pd.date_range(
        start=f"{year}-{month:02d}-01",
        end=f"{year}-{month:02d}-{days_in_month}",
        freq='D'
    )
    
    # Get actual dates from data for this month
    actual_dates = date_series[
        (date_series.dt.year == year) & 
        (date_series.dt.month == month)
    ]
    
    # Count unique dates
    unique_dates = actual_dates.dt.date.unique()
    
    # Month is considered full if we have at least 90% of days
    # (accounts for possible missing data on weekends/holidays)
    completeness_ratio = len(unique_dates) / days_in_month
    
    logger.debug(f"Month {year}-{month:02d}: {len(unique_dates)}/{days_in_month} days = {completeness_ratio:.1%}")
    
    # Return True if we have at least 90% of days
    return completeness_ratio >= 0.9

def filter_complete_months(df_daily, df_monthly):
    """
    Filter monthly data to keep only complete months.
    
    Args:
        df_daily: Daily sales data
        df_monthly: Monthly aggregated data
        
    Returns:
        DataFrame with only complete months
    """
    logger.info("Filtering for complete months only...")
    
    complete_months = []
    
    for idx, row in df_monthly.iterrows():
        month_end = row['month']
        
        if is_full_month(df_daily['date'], month_end):
            complete_months.append(row)
        else:
            logger.warning(f"Dropping incomplete month: {month_end.strftime('%Y-%m')}")
    
    if complete_months:
        df_complete = pd.DataFrame(complete_months)
        logger.info(f"Complete months: {len(df_complete)}/{len(df_monthly)} months retained")
        logger.info(f"Complete date range: {df_complete['month'].min()} to {df_complete['month'].max()}")
    else:
        df_complete = pd.DataFrame(columns=df_monthly.columns)
        logger.warning("No complete months found!")
    
    return df_complete

def prepare_prophet_data(df_monthly):
    """
    Prepare data for Prophet model format and apply log1p transformation
    """
    logger.info("Preparing data for Prophet model...")
    
    if len(df_monthly) == 0:
        raise ValueError("No monthly data available after filtering complete months")
    
    df_prophet = df_monthly[['month', 'monthly_sales']].copy()
    df_prophet = df_prophet.rename(columns={'month': 'ds', 'monthly_sales': 'y'})
    
    # Apply log1p transformation for Prophet
    df_prophet['y'] = np.log1p(df_prophet['y'])
    logger.info(f"Applied log1p transformation to target variable")
    logger.info(f"Prophet data shape: {df_prophet.shape}")
    logger.info(f"Prophet date range: {df_prophet['ds'].min()} to {df_prophet['ds'].max()}")
    
    return df_prophet

def inverse_transform_predictions(predictions_log):
    """
    Reverse log1p transformation: exp(predictions) - 1
    """
    logger.info("Applying inverse transformation (expm1) to predictions...")
    predictions = np.expm1(predictions_log)
    return predictions

# Add this function to utils_monthly.py after the existing functions

def calculate_forecast_metrics(df_prophet, forecast):
    """
    Calculate forecast accuracy metrics (MAPE, MAE, RMSE) 
    on historical data using Prophet predictions.
    
    Args:
        df_prophet: DataFrame with actual values (log-transformed)
        forecast: Prophet forecast DataFrame with predictions
    
    Returns:
        dict: Dictionary containing metrics
    """
    logger.info("Calculating forecast metrics...")
    
    # Merge actual values with predictions
    merged = pd.merge(df_prophet, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                     on='ds', how='inner')
    
    # Inverse transform both actual and predicted values
    y_true = np.expm1(merged['y'])  # Actual values (original scale)
    y_pred = np.expm1(merged['yhat'])  # Predicted values (original scale)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Calculate MAPE (avoid division by zero)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    metrics = {
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(y_true)
    }
    
    logger.info(f"Metrics calculated on {len(y_true)} samples:")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  MAE: {mae:,.2f}")
    logger.info(f"  RMSE: {rmse:,.2f}")
    
    return metrics

def generate_future_months(last_month, periods=12):
    """
    Generate future month dates for forecasting
    """
    logger.info(f"Generating {periods} future month dates...")
    
    # Prophet uses month-end dates
    future_months = pd.date_range(
        start=last_month + pd.DateOffset(months=1),
        periods=periods,
        freq='M'  # Month End
    )
    
    logger.info(f"Future months: {future_months[0].strftime('%Y-%m')} to {future_months[-1].strftime('%Y-%m')}")
    return future_months

def get_current_month_completeness(df_daily):
    """
    Check completeness of current month
    """
    current_date = datetime.now()
    current_month = current_date.replace(day=1)
    
    # Get days in current month
    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
    days_so_far = current_date.day
    
    # Get data for current month
    current_month_data = df_daily[
        (df_daily['date'].dt.year == current_date.year) & 
        (df_daily['date'].dt.month == current_date.month)
    ]
    
    unique_days = current_month_data['date'].dt.day.nunique()
    completeness = unique_days / days_in_month
    
    logger.info(f"Current month ({current_date.strftime('%Y-%m')}): {unique_days}/{days_in_month} days = {completeness:.1%}")
    
    return completeness >= 0.9  # Consider complete if we have 90% of days