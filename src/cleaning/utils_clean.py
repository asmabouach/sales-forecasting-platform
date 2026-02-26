import numpy as np
import pandas as pd
import joblib
import os

# ============================================
# DATA CLEANING
# ============================================
def clean_data(df_week):
    """Clean the weekly data"""
    # Missing values
    df_week["Disc"] = df_week["Disc"].fillna(0)
    df_week["CostOfGoods"] = df_week["CostOfGoods"].fillna(0)
    df_week["DutyStamp"] = df_week["DutyStamp"].fillna(0)
    df_week["VAT"] = df_week["VAT"].fillna(0)

    # Round floats
    float_cols = df_week.select_dtypes(include="float").columns
    df_week[float_cols] = df_week[float_cols].round(3)

    # Rename columns
    df_week.columns = [
        "Client_ID",
        "Invoice_Number",
        "Date",
        "Gross_Sales",
        "Discount",
        "Net_Sales",
        "CoG",
        "VAT",
        "Stamp_Duty",
        "Total_Amount",
        "Client_Type"
    ]
    
    return df_week


# ============================================
# FEATURE ENGINEERING
# ============================================
def engineer_features(df_week):
    """Create new features from the data"""
    df_week["Profit"] = df_week["Net_Sales"] - df_week["CoG"]

    df_week["Payment_Method"] = np.where(
        df_week["Client_Type"] == "CR", "Credit", "Cash"
    )

    df_week["Is_Return"] = np.where(
        df_week["Client_Type"] == "RE", "Yes", "No"
    )

    df_week["Total_Amount"] = (
        df_week["Net_Sales"] +
        df_week["VAT"] +
        df_week["Stamp_Duty"]
    )

    df_week.loc[df_week["Client_Type"] == "RE", "Profit"] = 0
    
    return df_week


# ============================================
# BUSINESS RULES – CLIENT CATEGORY
# ============================================
def categorize_client(row):
    """Categorize client based on business rules"""
    if row["Client_Type"] == "CL":
        return "Walk-In Client"
    elif row["Client_Type"] == "OR" and row["Stamp_Duty"] == 0:
        return "State Organization"
    elif row["Client_Type"] == "OR" and row["Stamp_Duty"] == 1:
        return "Private Organization"
    elif row["Client_Type"] == "IC":
        return "Insurance Agency"
    else:
        return "X"


def apply_business_rules(df_week):
    """Apply business rules for client categorization"""
    df_week["Client_Category"] = df_week.apply(categorize_client, axis=1)

    # Resolve X using client history
    client_categories = df_week.groupby("Client_ID")["Client_Category"].unique()

    def resolve_category(categories):
        priority = [
            "Walk-In Client",
            "State Organization",
            "Private Organization",
            "Insurance Agency"
        ]
        for p in priority:
            if p in categories:
                return p
        return "X"

    client_final_category = client_categories.apply(resolve_category)

    mask_X = df_week["Client_Category"] == "X"
    df_week.loc[mask_X, "Client_Category"] = (
        df_week.loc[mask_X, "Client_ID"]
        .map(client_final_category)
        .fillna("X")
    )
    
    return df_week


# ============================================
# OUTLIER CAPPING
# ============================================
def cap_outliers(df, cols, q_low=0.01, q_high=0.99):
    """Cap outliers in specified columns"""
    for col in cols:
        low, high = df[col].quantile([q_low, q_high])
        df[col] = df[col].clip(low, high)
    return df


# ============================================
# TEMPORAL & CLIENT AGG FEATURES
# ============================================
def create_temporal_client_features(df_week):
    """Create temporal and client aggregation features"""
    df_week["Month"] = df_week["Date"].dt.month
    df_week["Weekday"] = df_week["Date"].dt.weekday

    df_week["CoG_to_Net"] = (
        df_week["CoG"] / df_week["Net_Sales"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    df_week = df_week.sort_values(["Client_ID", "Date"])

    df_week["Client_Past_Transactions"] = (
        df_week.groupby("Client_ID").cumcount()
    )

    df_week["Client_Cum_Sales"] = (
        df_week.groupby("Client_ID")["Net_Sales"].cumsum()
        - df_week["Net_Sales"]
    )

    df_week["Client_Cum_Profit"] = (
        df_week.groupby("Client_ID")["Profit"].cumsum()
        - df_week["Profit"]
    )

    df_week["Client_Avg_Past_Profit"] = (
        df_week["Client_Cum_Profit"] /
        df_week["Client_Past_Transactions"]
    ).fillna(0)

    df_week["Client_Total_Past_Sales"] = df_week["Client_Cum_Sales"]

    df_week.drop(
        columns=[
            "Client_Cum_Sales",
            "Client_Cum_Profit",
            "Client_Past_Transactions"
        ],
        inplace=True
    )
    
    return df_week


def evaluate_classification_model(y_true, y_pred, y_proba=None, model_name="Classification Model"):
    """
    Evaluate classification model with comprehensive metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities (for ROC-AUC)
    model_name : str
        Name of the model for logging
    
    Returns:
    --------
    dict: Dictionary containing all metrics
    """
    from sklearn.metrics import (
        classification_report,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        confusion_matrix
    )
    import numpy as np
    
    print("=" * 60)
    print(f"CLASSIFICATION MODEL EVALUATION: {model_name}")
    print("=" * 60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    
    # Print basic metrics
    print(f"\n📊 BASIC METRICS:")
    print(f"   Accuracy:        {accuracy:.4f}")
    print(f"   F1 Macro:        {f1_macro:.4f}")
    print(f"   F1 Weighted:     {f1_weighted:.4f}")
    print(f"   MCC:             {mcc:.4f}")
    
    # Calculate ROC-AUC if probabilities are provided
    if y_proba is not None:
        try:
            # For multi-class ROC-AUC
            if len(np.unique(y_true)) > 2:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
            
            metrics['roc_auc'] = roc_auc
            print(f"   ROC-AUC:         {roc_auc:.4f}")
        except Exception as e:
            print(f"   ROC-AUC:         Could not calculate ({str(e)})")
    
    # Print classification report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred))
    
    # Print confusion matrix (simple format)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🎯 CONFUSION MATRIX (Shape: {cm.shape}):")
    print(cm)
    
    # Calculate per-class metrics from confusion matrix
    print(f"\n📈 PER-CLASS METRICS FROM CONFUSION MATRIX:")
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   Class {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return metrics

# ============================================
# ML INFERENCE FOR UNKNOWN CLIENTS
# ============================================
def perform_ml_inference(df_week, model_path, encoder_path):
    """Perform ML inference for unknown client categories"""
    feature_cols = [
        "Net_Sales", "Stamp_Duty", "Profit",
        "Month", "Weekday", "CoG_to_Net",
        "Client_Total_Past_Sales",
        "Client_Avg_Past_Profit"
    ]

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found")

    xgb_opt = joblib.load(model_path)
    le = joblib.load(encoder_path)

    df_unknown = df_week[df_week["Client_Category"] == "X"].dropna(subset=feature_cols)

    priority_weights = np.array([0.18, 0.15, 0.40, 0.15])

    if not df_unknown.empty:
        probs = xgb_opt.predict_proba(df_unknown[feature_cols])
        adjusted = probs / priority_weights
        preds = np.argmax(adjusted, axis=1)
        df_week.loc[df_unknown.index, "Client_Category"] = (
            le.inverse_transform(preds)
        )
    
    return df_week