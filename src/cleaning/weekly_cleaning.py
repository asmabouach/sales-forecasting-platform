import os
import sys
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Import the utility functions
from utils_clean import (
    clean_data,
    engineer_features,
    apply_business_rules,
    cap_outliers,
    create_temporal_client_features,
    evaluate_classification_model,
    perform_ml_inference
)

# ============================================
# PATHS & LOGGING
# ============================================
with open(Path(__file__).resolve().parents[2] / "configs" / "paths.yml", "r") as f:
    paths = yaml.safe_load(f)

raw_data_path = Path(__file__).resolve().parents[2] / paths["raw_data_path"]
interim_path = Path(__file__).resolve().parents[2] / paths["interim_path"]
processed_parquet = Path(__file__).resolve().parents[2] / paths["processed_parquet"]
processed_excel = Path(__file__).resolve().parents[2] / paths["processed_excel"]
processed_csv = Path(__file__).resolve().parents[2] / paths["processed_csv"]
model_path = Path(__file__).resolve().parents[2] / paths["client_category_model_path"]
encoder_path = Path(__file__).resolve().parents[2] / paths["client_category_encoder_path"]
logs_dir = Path(__file__).resolve().parents[2] / paths["logs_dir_cleaning"]

# === Logging setup ===
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================
#  Historical data integration
# =============================
"""
excel_files = [
    "/opt/airflow/data/raw/sales_2023.xlsx",
    "/opt/airflow/data/raw/sales_2024.xlsx",
    "/opt/airflow/data/raw/sales_2025.xlsx",  
    "/opt/airflow/data/raw/sales_2026.xlsx"
]
full_path = Path(__file__).parents[2]
files = [full_path / f for f in excel_files]

df_all = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
df_all["Date"] = pd.to_datetime(df_all["Date"]).dt.normalize()

logger.info(f"✅ Loaded ALL years: {len(df_all)} rows")
df_week = df_all.copy()
"""
# =============================

# ============================================
# LAST COMPLETED WEEK CHECK (HARD STOP)
# ============================================
# For historical data you have to uncomment this below

#"""
df_all = pd.read_excel(raw_data_path)
df_all["Date"] = pd.to_datetime(df_all["Date"]).dt.normalize()

today = datetime.today()
last_sunday = today - timedelta(days=today.weekday() + 1)
last_monday = last_sunday - timedelta(days=6)

df_week = df_all[
    (df_all["Date"] >= last_monday) &
    (df_all["Date"] <= last_sunday)
].copy()

logger.info(f"Last week data shape: {df_week.shape}")

if df_week.empty:
    logger.warning("No new data for last completed week. Pipeline stopped.")
else:
    # ============================================
    # CHECK IF LAST WEEK ALREADY EXISTS IN PROCESSED FILE
    # ============================================
    try:
        df_processed = pd.read_parquet(processed_parquet)
        df_processed["Date"] = pd.to_datetime(df_processed["Date"]).dt.normalize()
        logger.info(f"Loaded processed data: {len(df_processed)} rows")
       
        # Check if any data exists in processed file for the same week
        week_in_processed = df_processed[
            (df_processed["Date"] >= last_monday) &
            (df_processed["Date"] <= last_sunday)
        ]
       
        if not week_in_processed.empty:
            logger.warning(f"Last week's data already exists in processed file ({len(week_in_processed)} rows). Pipeline stopped.")
            
           
        logger.info("Last week's data is new. Proceeding with cleaning.")
       
    except FileNotFoundError:
        logger.info("No existing processed data found. Proceeding with cleaning.")
        df_processed = pd.DataFrame()
#"""
    # ============================================
    # DATA CLEANING
    # ============================================
    df_week = clean_data(df_week)

    # ============================================
    # FEATURE ENGINEERING
    # ============================================
    df_week = engineer_features(df_week)

    # ============================================
    # BUSINESS RULES – CLIENT CATEGORY
    # ============================================
    df_week = apply_business_rules(df_week)

    # ============================================
    # SAVE INTERIM DATA
    # ============================================
    df_week.to_parquet(interim_path, index=False)

    # ============================================
    # OUTLIER CAPPING
    # ============================================
    num_cols = [
        "Gross_Sales", "Discount", "Net_Sales",
        "CoG", "VAT", "Total_Amount", "Profit"
    ]
    df_week = cap_outliers(df_week, num_cols)

    # ============================================
    # TEMPORAL & CLIENT AGG FEATURES
    # ============================================
    df_week = create_temporal_client_features(df_week)

    # ============================================
    # ML INFERENCE FOR UNKNOWN CLIENTS + FULL MODEL EVALUATION
    # ============================================
    logger.info("Starting ML inference and comprehensive model evaluation...")
    # Load XGBoost model and label encoder
    xgb_model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    # Feature columns used by the model
    feature_cols = [
        "Net_Sales", "Stamp_Duty", "Profit",
        "Month", "Weekday", "CoG_to_Net",
        "Client_Total_Past_Sales",
        "Client_Avg_Past_Profit"
    ]
    # Prepare clean data for ML (drop rows with any NaN in features)
    df_for_ml = df_week[feature_cols + ['Client_Category']].dropna(subset=feature_cols)
    known_df = df_for_ml[df_for_ml['Client_Category'] != 'X']
    unknown_df = df_for_ml[df_for_ml['Client_Category'] == 'X']
    n_known = len(known_df)
    n_unknown = len(unknown_df)
    logger.info(f"Records for model evaluation (rule-based): {n_known}")
    logger.info(f"Records needing ML prediction (unknown 'X'): {n_unknown}")

    # ============================================
    # FULL MODEL EVALUATION ON RULE-BASED (KNOWN) DATA
    # ============================================
    if n_known > 0:
        X_known = known_df[feature_cols]
        y_true_str = known_df['Client_Category']
        y_true_encoded = le.transform(y_true_str)

        y_pred_encoded = xgb_model.predict(X_known)
        y_pred_str = le.inverse_transform(y_pred_encoded)
        y_proba = xgb_model.predict_proba(X_known)

        logger.info("Running comprehensive evaluation on rule-based data...")

        metrics = evaluate_classification_model(
            y_true=y_true_encoded,
            y_pred=y_pred_encoded,
            y_proba=y_proba,
            model_name="Client Category XGBoost"
        )

        # Extra: Log the full classification report as string
        from sklearn.metrics import classification_report
        class_report = classification_report(y_true_str, y_pred_str, digits=4)

        # Pretty print + log everything
        print("\n" + "="*80)
        print("CLIENT CATEGORY MODEL EVALUATION (on rule-based known data)")
        print("="*80)
        print(f" F1 Macro:         {metrics['f1_macro']:.4f}")
        print(f" F1 Weighted:      {metrics['f1_weighted']:.4f}")
        print(f" MCC:              {metrics['mcc']:.4f}")
        if 'roc_auc' in metrics:
            print(f" ROC-AUC (macro):  {metrics['roc_auc']:.4f}")
        print("\n DETAILED CLASSIFICATION REPORT:")
        print(class_report)

        # Log the same information
        logger.info("Model evaluation results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        logger.info(f"MCC: {metrics['mcc']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info("\nClassification Report:\n" + class_report)

    else:
        print("⚠️  No rule-based records available → skipping model evaluation.")
        logger.warning("No known records for model evaluation.")

    # ============================================
    # ML INFERENCE ON UNKNOWN ('X') CLIENTS
    # ============================================
    logger.info("Starting ML inference for unknown ('X') client categories...")

    try:
        df_week = perform_ml_inference(
            df_week=df_week,
            model_path=model_path,
            encoder_path=encoder_path
        )
        logger.info("ML inference for unknown clients completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        print("ERROR: Cannot find the trained model or encoder file!")
        # You can decide whether to continue or exit here
        # sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during ML inference: {str(e)}")
        print(f"ML inference failed: {str(e)}")
    # ============================================
    # FINAL DISTRIBUTION
    # ============================================
    print("\n" + "="*60)
    print("FINAL CLIENT CATEGORY DISTRIBUTION")
    print("="*60)
    dist = df_week['Client_Category'].value_counts().sort_values(ascending=False)
    print(dist)
    logger.info("Final client category distribution:\n" + dist.to_string())

    # ============================================
    # FINAL DATASET
    # ============================================
    df_final = df_week.drop(
        columns=[
            "Month", "Weekday", "CoG_to_Net",
            "Client_Avg_Past_Profit",
            "Client_Total_Past_Sales"
        ]
    )

    final_order = [
        "Client_ID", "Invoice_Number", "Date",
        "Gross_Sales", "Discount", "Net_Sales",
        "CoG", "Profit", "VAT", "Stamp_Duty",
        "Total_Amount", "Client_Category",
        "Payment_Method", "Is_Return", "Client_Type"
    ]

    df_final = df_final[final_order]
    logger.info(f"Total records: {len(df_final)}")
    logger.info(f"Columns: {len(df_final.columns)}")
    logger.info(f"Null values: {df_final.isnull().sum().sum()}")
    logger.info(f"Client categories distribution: {df_final['Client_Category'].value_counts()}")

    # ============================================
    # SAVE OUTPUTS
    # ============================================
    try:
        # Load historical cleaned data
        df_historical = pd.read_parquet(processed_parquet)
        logger.info(f"Loaded historical data: {len(df_historical)} rows")
        
        # Get unique invoices from historical
        existing_invoices = set(df_historical['Invoice_Number'].unique())
        
        # Filter only NEW rows based on Invoice_Number
        df_final_new = df_final[~df_final['Invoice_Number'].isin(existing_invoices)]
        
        if df_final_new.empty:
            msg = "No new cleaned rows to append to historical data."
            print(msg)
            logger.info(msg)
            # Nothing to append, just keep historical data
            df_combined = df_historical
        else:
            # Concatenate historical + new data
            df_combined = pd.concat([df_historical, df_final_new], ignore_index=True)
            
            msg = f"Cleaned dataset updated with {len(df_final_new)} new rows."
            print(msg)
            logger.info(msg)
            logger.info(f"Total rows after update: {len(df_combined)}")
            
    except FileNotFoundError:
        # If file doesn't exist, start fresh with current week
        logger.info("No historical data found. Starting fresh.")
        df_combined = df_final.copy()

    # ============================================
    # SAVE THE UPDATED DATASET
    # ============================================
    df_combined.to_parquet(processed_parquet, index=False)
    df_combined.to_excel(processed_excel, index=False)
    df_combined.to_csv(processed_csv, index=False)

    logger.info("Data saved successfully.")