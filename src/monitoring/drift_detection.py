import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from scipy.stats import ks_2samp
import logging
from datetime import datetime

# ------------------- Load Config Paths -------------------
with open(Path(__file__).resolve().parents[2] / "configs" / "paths.yml", "r") as f:
    paths = yaml.safe_load(f)

PROCESSED_PARQUET = Path(__file__).resolve().parents[2] / paths["processed_parquet_monitoring"]
log_dir = Path(__file__).resolve().parents[2] / paths["logs_dir_monitoring"]

# Log file with timestamp
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Update cutoff to be consistent with dashboard
REFERENCE_CUTOFF = "2025-09-30"  # End of Q3 2025 - stable reference period


def load_data():
    """Load the latest processed transactional data"""
    logger.info("Starting data load for drift detection")
    if not PROCESSED_PARQUET.exists():
        logger.error(f"Processed data file not found: {PROCESSED_PARQUET}")
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_PARQUET}")
    
    df = pd.read_parquet(PROCESSED_PARQUET)
    df["Date"] = pd.to_datetime(df["Date"])
    logger.info(f"Successfully loaded {len(df):,} rows from {PROCESSED_PARQUET}")
    return df


def split_ref_current(df):
    """Split into reference and current data"""
    ref = df[df["Date"] <= REFERENCE_CUTOFF].copy()
    curr = df[df["Date"] > REFERENCE_CUTOFF].copy()
    logger.info(f"Reference period (≤ {REFERENCE_CUTOFF}): {len(ref):,} rows")
    logger.info(f"Current period (> {REFERENCE_CUTOFF}): {len(curr):,} rows")
    return ref, curr


def ks_drift_test(ref_series, curr_series, feature_name, alpha=0.05):
    stat, p = ks_2samp(ref_series.dropna(), curr_series.dropna())
    drift = p < alpha
    logger.info(f"KS Test - {feature_name}: stat={round(stat,4)}, p={round(p,4)} → {'DRIFT' if drift else 'Stable'}")
    return {
        "feature": feature_name,
        "ks_statistic": round(stat, 4),
        "p_value": round(p, 4),
        "drift_detected": drift,
        "status": "DRIFT" if drift else "Stable"
    }


def tvd_drift(ref_series, curr_series, feature_name, threshold=0.10):
    ref_dist = ref_series.value_counts(normalize=True).reindex(curr_series.unique(), fill_value=0)
    curr_dist = curr_series.value_counts(normalize=True).reindex(ref_series.unique(), fill_value=0)
    all_categories = ref_dist.index.union(curr_dist.index)
    ref_dist = ref_dist.reindex(all_categories, fill_value=0)
    curr_dist = curr_dist.reindex(all_categories, fill_value=0)
    
    tvd = np.sum(np.abs(ref_dist - curr_dist)) / 2
    drift = tvd > threshold
    logger.info(f"TVD - {feature_name}: tvd={round(tvd,4)} → {'DRIFT' if drift else 'Stable'}")
    return {
        "feature": feature_name,
        "tvd": round(tvd, 4),
        "drift_detected": drift,
        "status": "DRIFT" if drift else "Stable"
    }


# The rest of the functions remain the same, but with added logging
def check_client_classifier_drift():
    logger.info("Starting drift check for Client Category Classifier")
    df = load_data()
    ref, curr = split_ref_current(df)
    
    if len(curr) == 0:
        logger.warning("No current data available for Client Classifier drift check")
        return {"model": "Client Category Classifier", "error": "No current data available"}

    results = []
    num_features = [
        "Net_Sales", "Stamp_Duty", "Profit",
        "CoG_to_Net", "Client_Total_Past_Sales", "Client_Avg_Past_Profit"
    ]
    
    for feat in num_features:
        if feat in df.columns:
            results.append(ks_drift_test(ref[feat], curr[feat], feat))
    
    if "Client_Category" in df.columns:
        results.append(tvd_drift(ref["Client_Category"], curr["Client_Category"], "Client_Category"))
    
    logger.info("Client Classifier drift check completed")
    return {"model": "Client Category Classifier", "results": results}


def check_weekly_forecast_drift():
    logger.info("Starting drift check for Weekly Forecast")
    df = load_data()
    df["week"] = df["Date"].dt.to_period("W-MON").apply(lambda x: x.start_time)
    weekly_sales = df.groupby("week")["Net_Sales"].sum().reset_index()
    
    ref = weekly_sales[weekly_sales["week"] <= REFERENCE_CUTOFF]
    curr = weekly_sales[weekly_sales["week"] > REFERENCE_CUTOFF]
    
    if len(curr) < 3:
        logger.warning("Not enough recent weeks for Weekly Forecast drift check")
        return {"model": "Weekly Forecast", "error": "Not enough recent weeks"}
    
    result = ks_drift_test(ref["Net_Sales"], curr["Net_Sales"], "Weekly Sales")
    logger.info("Weekly Forecast drift check completed")
    return {"model": "Weekly Forecast", "results": [result]}


def check_monthly_forecast_drift():
    logger.info("Starting drift check for Monthly Forecast")
    df = load_data()
    df["month"] = df["Date"].dt.to_period("M").apply(lambda x: x.start_time)
    monthly_sales = df.groupby("month")["Net_Sales"].sum().reset_index()
    
    ref = monthly_sales[monthly_sales["month"] <= REFERENCE_CUTOFF]
    curr = monthly_sales[monthly_sales["month"] > REFERENCE_CUTOFF]
    
    if len(curr) < 2:
        logger.warning("Not enough recent months for Monthly Forecast drift check")
        return {"model": "Monthly Forecast", "error": "Not enough recent months"}
    
    result = ks_drift_test(ref["Net_Sales"], curr["Net_Sales"], "Monthly Sales")
    logger.info("Monthly Forecast drift check completed")
    return {"model": "Monthly Forecast", "results": [result]}


def run_all_drift_checks():
    logger.info("=== Starting full drift detection run ===")
    checks = [
        check_client_classifier_drift(),
        check_weekly_forecast_drift(),
        check_monthly_forecast_drift()
    ]
    valid_checks = [c for c in checks if "error" not in c]
    logger.info(f"Drift detection completed. {len(valid_checks)}/{len(checks)} models checked successfully.")
    return valid_checks


if __name__ == "__main__":
    reports = run_all_drift_checks()
    for report in reports:
        print(f"\n=== {report['model']} ===")
        for r in report["results"]:
            print(f"{r['feature']}: {r['status']} (p={r.get('p_value', 'N/A')}, tvd={r.get('tvd', 'N/A')})")