import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import all functions from utils_clean
from src.cleaning.utils_clean import (
    clean_data,
    engineer_features,
    apply_business_rules,
    cap_outliers,
    create_temporal_client_features,
    perform_ml_inference,
    evaluate_classification_model
)


@pytest.fixture
def base_df():
    """Base dataframe with original column names (before cleaning)"""
    return pd.DataFrame({
        'Client_ID': ['C001', 'C002', 'C003'],
        'Invoice_Number': ['INV001', 'INV002', 'INV003'],
        'Date': pd.to_datetime(['2025-01-06', '2025-01-07', '2025-01-08']),
        'Gross_Sales': [1000.0, 2000.0, 1500.0],
        'Disc': [100.0, np.nan, 150.0],
        'Net_Sales': [900.0, 1800.0, 1350.0],
        'CostOfGoods': [500.0, np.nan, 750.0],
        'VAT': [162.0, 324.0, np.nan],
        'DutyStamp': [0.0, 1.0, 0.0],
        'Total_Amount': [1062.0, 2125.0, 1593.0],
        'Client_Type': ['CL', 'OR', 'IC']
    })


def test_clean_data(base_df):
    cleaned = clean_data(base_df.copy())
    # Check renaming
    assert 'Discount' in cleaned.columns
    assert 'CoG' in cleaned.columns
    assert 'Stamp_Duty' in cleaned.columns
    # Check missing values filled with 0
    assert cleaned[['Discount', 'CoG', 'Stamp_Duty', 'VAT']].isna().sum().sum() == 0
    assert (cleaned[['Discount', 'CoG', 'Stamp_Duty', 'VAT']] >= 0).all().all()


def test_engineer_features(base_df):
    # First clean to get correct column names
    df = clean_data(base_df.copy())
    engineered = engineer_features(df)

    assert 'Profit' in engineered.columns
    assert 'Payment_Method' in engineered.columns
    assert 'Is_Return' in engineered.columns

    # Specific checks
    assert engineered.loc[0, 'Profit'] == 400.0  # 900 - 500
    assert engineered.loc[0, 'Payment_Method'] == 'Cash'  # CL != CR
    assert engineered.loc[0, 'Is_Return'] == 'No'
    assert engineered.loc[0, 'Total_Amount'] == 900 + 162 + 0  # recalculated


def test_apply_business_rules(base_df):
    df = clean_data(base_df.copy())
    df = engineer_features(df)
    df = apply_business_rules(df)

    assert 'Client_Category' in df.columns
    expected = ['Walk-In Client', 'Private Organization', 'Insurance Agency']
    assert list(df['Client_Category']) == expected

    # Test X resolution with repeated client
    df2 = pd.concat([df, df.iloc[[0]]])  # duplicate C001
    df2.iloc[-1, df2.columns.get_loc('Client_Type')] = 'X'  # force X (FIXED: Use iloc for position-based access)
    resolved = apply_business_rules(df2)
    assert resolved['Client_Category'].iloc[-1] == 'Walk-In Client'  # resolved from history


def test_cap_outliers():
    df = pd.DataFrame({'values': [1, 2, 3, 1000, 10000]})
    capped = cap_outliers(df.copy(), ['values'])
    assert capped['values'].max() <= df['values'].quantile(0.99)
    assert capped['values'].min() >= df['values'].quantile(0.01)


def test_create_temporal_client_features(base_df):
    df = clean_data(base_df.copy())
    df = engineer_features(df)
    df = apply_business_rules(df)

    result = create_temporal_client_features(df)

    assert 'Month' in result.columns
    assert 'Weekday' in result.columns
    assert 'CoG_to_Net' in result.columns
    assert 'Client_Avg_Past_Profit' in result.columns
    assert 'Client_Total_Past_Sales' in result.columns

    # For single transactions per client → past = 0
    assert (result['Client_Avg_Past_Profit'] == 0).all()
    assert (result['Client_Total_Past_Sales'] == 0).all()


def test_evaluate_classification_model():
    X, y = make_classification(n_samples=100, n_classes=4, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = evaluate_classification_model(y_test, y_pred, y_proba, model_name="Test XGBoost")

    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'mcc' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1