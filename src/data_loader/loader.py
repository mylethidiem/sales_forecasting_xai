import json
import pickle
import pandas as pd
import gradio as gr
import pyarrow.feather as feather
from functools import lru_cache

# --- Data & Model Loading Logic ---

def load_model():
    """Load the trained sales forecast model"""
    try:
        with open("models/sales_forecast_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Using gr.Error for UI notification if called within an interaction
        # or standard print for startup logs
        print("Error: 'models/sales_forecast_model.pkl' not found.")
        return None

def load_feature_stats():
    """Load feature statistics used for normalization"""
    try:
        with open("models/feature_stats.json", "r") as file:
            feature_stats = json.load(file)
        return feature_stats
    except FileNotFoundError:
        print("Error: 'models/feature_stats.json' not found.")
        return {}

@lru_cache(maxsize=1)
def load_data():
    """Load preprocessed sales data (lru_cache replaces @st.cache_data)"""
    try:
        df = pd.read_csv("data/sales_data_preprocessed.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        print("Error: 'data/sales_data_preprocessed.csv' not found.")
        return pd.DataFrame(columns=["date", "store", "sales"])

def load_feature_engineered_data():
    """Load feature engineered data with extended features"""
    try:
        feature_engineered_data = feather.read_feather(
            "data/feature_engineered_data_55_features.feather"
        )
        return feature_engineered_data
    except Exception as e:
        print(f"Error loading feature engineered data: {str(e)}")
        return pd.DataFrame()

# --- Processing Logic ---

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

# --- Gradio UI Implementation ---

# Load resources once when the app starts
model = load_model()
stats = load_feature_stats()

def predict_sales_ui(store_id):
    """Example function to link the logic to a Gradio interface"""
    if model is None:
        raise gr.Error("Model not loaded. Check server logs.")

    data = load_data()
    # Apply your logic
    processed = preprocess_data(data, stats)

    # Filter for the specific store
    store_data = processed[processed['store'] == store_id]

    # Return results (placeholder for actual model.predict logic)
    return store_data.head()

# Simple Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Sales Forecast Prediction")
    store_input = gr.Number(label="Enter Store ID")
    output_table = gr.DataFrame(label="Preprocessed Data Preview")
    btn = gr.Button("Predict")

    btn.click(fn=predict_sales_ui, inputs=store_input, outputs=output_table)

if __name__ == "__main__":
    demo.launch()