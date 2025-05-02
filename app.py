import streamlit as st

from src.data_loader.loader import (
    load_data,
    load_feature_engineered_data,
    load_feature_stats,
    load_model,
)
from src.ui_builder.dashboard import historical_sales_view
from src.ui_predictor.prediction import sales_prediction_view

# Page configuration
st.set_page_config(page_title="Sales Forecasting App", page_icon="📈", layout="wide")


def main():
    # Sidebar
    st.sidebar.title("Sales Forecasting App")
    page = st.sidebar.selectbox(
        "Choose a page", ["Historical Sales Analysis", "Sales Prediction"]
    )

    # Load data and model
    data = load_data()
    model = load_model()
    feature_stats = load_feature_stats()

    # Display page based on selection
    if page == "Historical Sales Analysis":
        historical_sales_view(data)
    else:
        # Load feature engineered data for prediction
        feature_engineered_data = load_feature_engineered_data()
        sales_prediction_view(data, model, feature_stats, feature_engineered_data)


if __name__ == "__main__":
    main()
