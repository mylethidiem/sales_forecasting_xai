import json
import pickle

import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    """Load the trained sales forecast model"""
    try:
        with open("models/sales_forecast_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(
            "Model file not found. Please ensure 'models/sales_forecast_model.pkl' exists."
        )
        return None


@st.cache_resource
def load_feature_stats():
    """Load feature statistics used for normalization"""
    try:
        with open("models/feature_stats.json", "r") as file:
            feature_stats = json.load(file)
        return feature_stats
    except FileNotFoundError:
        st.error(
            "Feature stats file not found. Please ensure 'models/feature_stats.json' exists."
        )
        return {}


@st.cache_data
def load_data():
    """Load preprocessed sales data"""
    try:
        # Load the preprocessed data
        df = pd.read_csv("data/sales_data_preprocessed.csv")

        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df
    except FileNotFoundError:
        st.error(
            "Data file not found. Please ensure 'data/sales_data_preprocessed.csv' exists."
        )
        # Return empty DataFrame with expected columns as fallback
        return pd.DataFrame(columns=["date", "store", "sales"])


@st.cache_data
def load_feature_engineered_data():
    """Load feature engineered data with extended features for predictions"""
    try:
        import pyarrow.feather as feather

        feature_engineered_data = feather.read_feather(
            "data/feature_engineered_data_55_features.feather"
        )
        return feature_engineered_data
    except Exception as e:
        st.error(f"Error loading feature engineered data: {str(e)}")
        st.info(
            "Please ensure the file 'data/feature_engineered_data_55_features.feather' exists."
        )
        return pd.DataFrame()


def preprocess_data(df, feature_stats=None):
    """Preprocess data for prediction (simplified version)"""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Extract date features if date column exists
    if "date" in processed_df.columns:
        processed_df["day_of_week"] = processed_df["date"].dt.dayofweek
        processed_df["day_of_month"] = processed_df["date"].dt.day
        processed_df["month"] = processed_df["date"].dt.month
        processed_df["year"] = processed_df["date"].dt.year
        processed_df["is_weekend"] = processed_df["day_of_week"].apply(
            lambda x: 1 if x >= 5 else 0
        )

    # Normalize numerical features if stats are provided
    if feature_stats:
        for feature, stats in feature_stats.items():
            if feature in processed_df.columns and "mean" in stats and "std" in stats:
                processed_df[feature] = (processed_df[feature] - stats["mean"]) / stats[
                    "std"
                ]

    return processed_df
