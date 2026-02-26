import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import joblib
import logging
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

# Import utility functions
from utils_weekly import (
    resample_weekly,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    check_last_week_completeness,
    get_feature_columns,
    calculate_forecast_metrics
)

# Load environment variables locally
#load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Force load .env from the correct path
env_path = Path("/opt/airflow/sales_forecasting/.env")
print(f"DEBUG: Loading .env from: {env_path}")
print(f"DEBUG: .env exists: {env_path.exists()}")
#load_dotenv()
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"DEBUG: Loaded .env successfully")
    print(f"DEBUG: DW_DB_PASSWORD = {os.getenv('DW_DB_PASSWORD')}")
else:
    print(f"DEBUG: ERROR: .env file not found at {env_path}")

# Check ALL environment variables
print("\nDEBUG: All DB-related environment variables:")
for key, value in os.environ.items():
    if 'DB' in key or 'PASSWORD' in key:
        print(f"  {key}: {value}")

# ============================================
# PATHS & LOGGING
# ============================================
with open(Path(__file__).resolve().parents[2] / "configs" / "paths.yml", "r") as f:
    paths = yaml.safe_load(f)

# Paths
model_path = Path(__file__).resolve().parents[2] / paths["weekly_model_path"]
prediction_path = Path(__file__).resolve().parents[2] / paths["weekly_prediction"]
logs_dir = Path(__file__).resolve().parents[2] / paths["logs_dir_ml"]

# Create logs directory
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"weekly_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    logger.info("STARTING WEEKLY FORECASTING PIPELINE")
    
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
        
        # ========== 2. CHECK DATA COMPLETENESS ==========
        df_weekly = resample_weekly(df)
        logger.info(f"Weekly data shape before completeness check: {df_weekly.shape}")
        
        if check_last_week_completeness(df, df_weekly):
            df_weekly = df_weekly.iloc[:-1].copy()
            logger.info(f"Weekly data shape after removing incomplete week: {df_weekly.shape}")
        
        # ========== 3. FILTER DATA ==========
        df_weekly = df_weekly[df_weekly['week'] > '2023-01-02'].copy()
        logger.info(f"Filtered date range: {df_weekly['week'].min()} to {df_weekly['week'].max()}")
        
        # ========== 4. FEATURE ENGINEERING ==========   
        df_weekly = add_calendar_features(df_weekly)
        df_weekly = df_weekly.sort_values("week").reset_index(drop=True)
        logger.info(f"Weekly sales statistics - Mean: {df_weekly['weekly_sales'].mean():,.2f}, "
                   f"Std: {df_weekly['weekly_sales'].std():,.2f}")
        
        # ========== 5. PREPARE FEATURES FOR MODEL ==========
        df_features = df_weekly.copy()
        df_features = add_lag_features(df_features)
        df_features = add_rolling_features(df_features)
        
        df_features = df_features.dropna().copy()
        logger.info(f"Features shape after dropping NaN: {df_features.shape}")
        
        # ========== 6. LOAD MODEL ==========
        model = joblib.load(model_path)
        logger.info(f"Model loaded: {type(model).__name__}")
        
        # ========== 7. MAKE PREDICTIONS ==========
        feature_cols = get_feature_columns()        
        X = df_features[feature_cols]
        predictions = model.predict(X)
        
        logger.info(f"Prediction range: {predictions.min():,.2f} to {predictions.max():,.2f}")
        
        # ========== 8. CALCULATE METRICS ==========
        # Get actual values for the same periods as predictions
        y_true = df_features['weekly_sales'].values
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(y_true, predictions, logger)
        
        # ========== 9. GENERATE FUTURE FORECAST ==========
        last_date = df_weekly['week'].iloc[-1]
        logger.info(f"Last historical week: {last_date.date()}")
        
        future_weeks = []
        for i in range(1, 13):
            week_date = last_date + pd.Timedelta(weeks=i)
            if i == 1:
                forecast_value = float(predictions[-1])
            else:
                # Apply slight decay for longer horizons
                forecast_value = float(predictions[-1] * (0.98 ** (i-1)))
            
            future_weeks.append({
                'week': week_date,
                'predicted_profit': forecast_value
            })
            
            logger.debug(f"Week {i} ({week_date.date()}): {forecast_value:,.2f}")

        
        df_future = pd.DataFrame(future_weeks)
        df_future['predicted_profit'] = df_future['predicted_profit'].round(3)
        logger.info(f"Forecast period: {df_future['week'].min().date()} to {df_future['week'].max().date()}")        
        # ========== 10. SAVE PREDICTIONS ==========
        prediction_path.parent.mkdir(parents=True, exist_ok=True)        
        df_future.to_parquet(prediction_path, index=False)
        
        # Create a CSV version for easier inspection
        csv_path = prediction_path.with_suffix('.csv')
        df_future.to_csv(csv_path, index=False)
        
        # ========== 11. FINAL SUMMARY ==========
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Historical weeks processed: {len(df_weekly)}")
        logger.info(f"Pipeline duration: {duration:.2f} seconds")
        
        # Console output
        print(f"\n Weekly forecast complete!")
        print(f"   Forecast period: {df_future['week'].min().date()} to {df_future['week'].max().date()}")
        print(f"   Model MAPE: {metrics['mape']:.2f}%")
        print(f"   Model MAE: {metrics['mae']:,.2f}")
        print(f"   Model RMSE: {metrics['rmse']:,.2f}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}", exc_info=True)
        print(f" File not found: {str(e)}")
        sys.exit(1)
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}", exc_info=True)
        print(f" Database error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in forecasting pipeline: {str(e)}", exc_info=True)
        print(f" Unexpected error: {str(e)}")
        sys.exit(1)

# ============================================
# EXECUTION
# ============================================
if __name__ == "__main__":
    main()