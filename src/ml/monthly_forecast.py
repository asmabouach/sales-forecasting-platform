import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

# Import utility functions
from utils_monthly import (
    resample_monthly,
    filter_complete_months,
    prepare_prophet_data,
    inverse_transform_predictions,
    generate_future_months,
    calculate_forecast_metrics
)

# Load environment variables
load_dotenv()

# ============================================
# PATHS & LOGGING
# ============================================
with open(Path(__file__).resolve().parents[2] / "configs" / "paths.yml", "r") as f:
    paths = yaml.safe_load(f)

# Paths
model_path = Path(__file__).resolve().parents[2] / paths["monthly_model_path"]
prediction_path = Path(__file__).resolve().parents[2] / paths["monthly_prediction"]
logs_dir = Path(__file__).resolve().parents[2] / paths["logs_dir_ml"]

# Create logs directory
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"monthly_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Logging setup
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ============================================
# DATABASE FUNCTIONS
# ============================================
def connect_db(env="dw"):
    """Connect to PostgreSQL database"""
    logger.info(f"Connecting to {env} database...")
    with open(Path(__file__).resolve().parents[2] / "configs" / "db.yml", "r") as f:
        db_config = yaml.safe_load(f)
    
    conn = psycopg2.connect(
        host=db_config[env]["host"],
        database=db_config[env]["database"],
        user=db_config[env]["user"],
        password=os.getenv(db_config[env]["password_env"]),
        port=db_config[env]["port"]
    )
    logger.info(f"Connected to database: {db_config[env]['database']}")
    return conn

def query_df(sql, env="dw"):
    """Query database and return DataFrame"""
    logger.info(f"Executing SQL query on {env} database...")
    logger.debug(f"Query: {sql[:200]}...")  # Log first 200 chars of query
    
    conn = connect_db(env)
    df = pd.read_sql(sql, conn)
    conn.close()
    
    logger.info(f"Query returned {len(df)} rows, {len(df.columns)} columns")
    logger.debug(f"DataFrame columns: {list(df.columns)}")
    return df

# ============================================
# MAIN FORECASTING PIPELINE
# ============================================
def main():
    logger.info("=" * 60)
    logger.info("STARTING MONTHLY FORECASTING PIPELINE")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    logger.info(f"Pipeline start time: {start_time}")
    
    try:
        # ========== 1. LOAD DATA ========== 
        df = query_df("""SELECT 
            d.date,
            s.net_sales
        FROM fact_sales AS s
        JOIN dim_date AS d
            ON s.date_id = d.date_id
        JOIN dim_invoice As i
            ON s.invoice_number = i.invoice_number
        WHERE i.is_return = 'No'
        ORDER BY s.date_id;
        """, env="dw")
        
        logger.info(f"Raw data loaded: {len(df)} rows")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Preprocess
        df['date'] = pd.to_datetime(df['date'])
        df['net_sales'] = df['net_sales'].round(3)
        logger.info(f"Total net sales: {df['net_sales'].sum():,.2f}")
        
        # ========== 2. CHECK LAST MONTH COMPLETENESS ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 2: Checking last month completeness")
        logger.info("-" * 40)
        
        # Get last date in data
        last_date = df['date'].max()
        last_day_of_month = pd.Timestamp(year=last_date.year, 
                                        month=last_date.month, 
                                        day=pd.Timestamp(last_date.year, last_date.month, 1).daysinmonth)
        
        logger.info(f"Last data date: {last_date.date()}")
        logger.info(f"Last day of month: {last_day_of_month.date()}")
        
        # ========== 3. RESAMPLE TO MONTHLY ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 3: Resampling to monthly frequency")
        logger.info("-" * 40)
        
        df_monthly = resample_monthly(df)
        logger.info(f"Monthly data shape: {df_monthly.shape}")
        
        # ========== 4. DROP INCOMPLETE LAST MONTH ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 4: Dropping incomplete last month if needed")
        logger.info("-" * 40)
        
        # Drop last month if incomplete
        if last_date != last_day_of_month:
            logger.warning(
                f"Last month {last_date.strftime('%Y-%m')} is incomplete (last data date: {last_date.date()}); dropping last month.")
            print(f"⚠️ Last month {last_date.strftime('%Y-%m')} is incomplete – dropping last month.")
            df_monthly = df_monthly.iloc[:-1].copy()
            logger.info(f"After dropping last month: {df_monthly.shape[0]} months remaining")

        # If no full months left, stop here
        if df_monthly.empty:
            logger.warning("No full months available after dropping incomplete data – aborting forecast.")
            print("⚠️ No full months available – skipping forecast.")
            return
        
        # ========== 5. FILTER FOR COMPLETE MONTHS (additional check) ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 5: Filtering for complete months")
        logger.info("-" * 40)
        
        df_monthly_complete = filter_complete_months(df, df_monthly)
        
        if len(df_monthly_complete) == 0:
            logger.error("No complete months available for forecasting")
            raise ValueError("No complete monthly data available for forecasting")
        
        logger.info(f"Complete monthly data shape: {df_monthly_complete.shape}")
        logger.info(f"Complete monthly date range: {df_monthly_complete['month'].min()} to {df_monthly_complete['month'].max()}")
        logger.info(f"Complete months sales - Mean: {df_monthly_complete['monthly_sales'].mean():,.2f}, "
                   f"Std: {df_monthly_complete['monthly_sales'].std():,.2f}")
        
        # ========== 6. PREPARE DATA FOR PROPHET ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 6: Preparing data for Prophet model")
        logger.info("-" * 40)
        
        df_prophet = prepare_prophet_data(df_monthly_complete)
        
        # ========== 7. LOAD PROPHET MODEL ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 7: Loading trained Prophet model")
        logger.info("-" * 40)
        
        logger.info(f"Loading Prophet model from: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model type: {type(model).__name__}")
        
        # ========== 8. CALCULATE HISTORICAL METRICS ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 8: Calculating historical metrics")
        logger.info("-" * 40)
        
        # Make predictions on historical data
        logger.info("Making predictions on historical data for metrics...")
        historical_forecast = model.predict(df_prophet[['ds']])
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(df_prophet, historical_forecast)
        
        # ========== 9. MAKE FUTURE PREDICTIONS ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 9: Making future predictions")
        logger.info("-" * 40)
        
        # Generate future dates for next 12 months
        last_month = df_monthly_complete['month'].iloc[-1]
        future_months = generate_future_months(last_month, periods=12)
        
        # Create future dataframe for Prophet
        future_df = pd.DataFrame({'ds': future_months})
        logger.info(f"Created future dataframe with {len(future_df)} months")
        
        # Make predictions
        logger.info("Making predictions with Prophet model...")
        forecast = model.predict(future_df)
        
        # Get predictions and confidence intervals
        predictions_log = forecast['yhat'].values
        lower_ci_log = forecast['yhat_lower'].values
        upper_ci_log = forecast['yhat_upper'].values
        
        # ========== 10. TRANSFORM PREDICTIONS BACK ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 10: Transforming predictions")
        logger.info("-" * 40)
        
        predictions = inverse_transform_predictions(predictions_log)
        lower_ci = inverse_transform_predictions(lower_ci_log)
        upper_ci = inverse_transform_predictions(upper_ci_log)
        
        logger.info(f"Prediction range: {predictions.min():,.2f} to {predictions.max():,.2f}")
        logger.info(f"Total predicted sales (12 months): {predictions.sum():,.2f}")
        
        # ========== 11. CREATE PREDICTION DATAFRAME ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 11: Creating prediction dataframe")
        logger.info("-" * 40)
        
        future_predictions = []
        for i, (month_date, pred_value, lower, upper) in enumerate(
            zip(future_months, predictions, lower_ci, upper_ci), 1):
            
            future_predictions.append({
                'date': month_date,
                'predicted_profit': float(pred_value),
                'lower_ci': float(lower),
                'upper_ci': float(upper)
            })
        
        df_future = pd.DataFrame(future_predictions)
        
        # ========== 12. SAVE PREDICTIONS ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 12: Saving predictions")
        logger.info("-" * 40)
        
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        df_future.to_parquet(prediction_path, index=False)
        
        csv_path = prediction_path.with_suffix('.csv')
        df_future.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to: {prediction_path}")
        
        # ========== 13. FINAL SUMMARY ==========
        logger.info("\n" + "-" * 40)
        logger.info("STEP 13: Pipeline summary")
        logger.info("-" * 40)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Complete months used: {len(df_monthly_complete)}")
        logger.info(f"Future months forecasted: {len(df_future)}")
        logger.info(f"Historical MAPE: {metrics['mape']:.2f}%")
        logger.info(f"Historical MAE: {metrics['mae']:,.2f}")
        logger.info(f"Historical RMSE: {metrics['rmse']:,.2f}")
        logger.info(f"Pipeline duration: {duration:.2f} seconds")
        logger.info("=" * 60)
        logger.info("MONTHLY FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Console output
        print(f"\n Monthly forecast complete!")
        print(f"   Used {len(df_monthly_complete)} complete months for training")
        print(f"   Forecast: {df_future['date'].min().strftime('%Y-%m')} to {df_future['date'].max().strftime('%Y-%m')}")
        print(f"   Total forecasted sales: ${df_future['predicted_profit'].sum():,.2f}")
        print(f"   Historical MAPE: {metrics['mape']:.2f}%")
        print(f"   Duration: {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Forecasting failed: {str(e)}", exc_info=True)
        print(f" Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()