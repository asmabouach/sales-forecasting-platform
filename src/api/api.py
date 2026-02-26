import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import yaml
from datetime import datetime
from src.monitoring.drift_detection import run_all_drift_checks
import numpy as np  
from fastapi.responses import JSONResponse

# ------------------- Load paths config -------------------
config_path = Path(__file__).parent.parent.parent / "configs" / "paths.yml"
with open(config_path) as f:
    paths_config = yaml.safe_load(f)

logs_dir = Path(paths_config["logs_dir_api"])
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ------------------- Setup Logging -------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------- FastAPI App -------------------
app = FastAPI(title="Sales Forecasting Drift API")

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    logger.info("API home endpoint accessed")
    return {"message": "Sales Forecasting Drift API is running!", "status": "healthy"}

def convert_to_python(obj):
    """Recursively convert numpy types to built-in Python types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Converts np.int64 -> int, np.float64 -> float
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python(item) for item in obj]
    return obj


@app.post("/run-drift-checks")
def run_drift_checks():
    logger.info("Drift checks requested via API")
    try:
        reports = run_all_drift_checks()
        # Convert any numpy types to standard Python types
        clean_reports = convert_to_python(reports)
        logger.info(f"Drift checks completed successfully. Found {len(clean_reports)} model reports.")
        return {"reports": clean_reports}
    
    except Exception as e:
        logger.error(f"Error during drift checks: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to run drift checks", "details": str(e)}
        )