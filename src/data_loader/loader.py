import json
import pickle
from functools import lru_cache
from typing import Dict, Any, Optional

import pandas as pd

# Note: Gradio doesn't have built-in caching like Streamlit's @st.cache_resource
# We'll use functools.lru_cache or implement a simple caching mechanism


class DataCache:
    """Simple cache for storing loaded data"""
    _model = None
    _feature_stats = None
    _data = None
    _feature_engineered_data = None


@lru_cache(maxsize=1)
def load_model():
    """Load the trained sales forecast model"""
    if DataCache._model is not None:
        return DataCache._model

    try:
        with open("models/sales_forecast_model.pkl", "rb") as file:
            model = pickle.load(file)
        DataCache._model = model
        return model
    except FileNotFoundError:
        print("ERROR: Model file not found. Please ensure 'models/sales_forecast_model.pkl' exists.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load model: {str(e)}")
        return None


@lru_cache(maxsize=1)
def load_feature_stats() -> Dict[str, Any]:
    """Load feature statistics used for normalization"""
    if DataCache._feature_stats is not None:
        return DataCache._feature_stats

    try:
        with open("models/feature_stats.json", "r") as file:
            feature_stats = json.load(file)
        DataCache._feature_stats = feature_stats
        return feature_stats
    except FileNotFoundError:
        print("ERROR: Feature stats file not found. Please ensure 'models/feature_stats.json' exists.")
        return {}
    except Exception as e:
        print(f"ERROR: Failed to load feature stats: {str(e)}")
        return {}


def load_data() -> pd.DataFrame:
    """Load preprocessed sales data"""
    if DataCache._data is not None:
        return DataCache._data.copy()

    try:
        # Load the preprocessed data
        df = pd.read_csv("data/sales_data_preprocessed.csv")

        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        DataCache._data = df
        return df
    except FileNotFoundError:
        print("ERROR: Data file not found. Please ensure 'data/sales_data_preprocessed.csv' exists.")
        # Return empty DataFrame with expected columns as fallback
        return pd.DataFrame(columns=["date", "store", "sales"])
    except Exception as e:
        print(f"ERROR: Failed to load data: {str(e)}")
        return pd.DataFrame(columns=["date", "store", "sales"])


def load_feature_engineered_data() -> pd.DataFrame:
    """Load feature engineered data with extended features for predictions"""
    if DataCache._feature_engineered_data is not None:
        return DataCache._feature_engineered_data.copy()

    try:
        import pyarrow.feather as feather

        feature_engineered_data = feather.read_feather(
            "data/feature_engineered_data_55_features.feather"
        )
        DataCache._feature_engineered_data = feature_engineered_data
        return feature_engineered_data
    except ImportError:
        print("ERROR: pyarrow not installed. Install with: pip install pyarrow")
        return pd.DataFrame()
    except FileNotFoundError:
        print("ERROR: Feature engineered data file not found.")
        print("Please ensure the file 'data/feature_engineered_data_55_features.feather' exists.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Error loading feature engineered data: {str(e)}")
        return pd.DataFrame()


def preprocess_data(df: pd.DataFrame, feature_stats: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Preprocess data for prediction (simplified version)

    Args:
        df: Input DataFrame
        feature_stats: Dictionary containing feature statistics for normalization

    Returns:
        Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Extract date features if date column exists
    if "date" in processed_df.columns:
        # Ensure date is datetime type
        if not pd.api.types.is_datetime64_any_dtype(processed_df["date"]):
            processed_df["date"] = pd.to_datetime(processed_df["date"])

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
                # Avoid division by zero
                std = stats["std"] if stats["std"] != 0 else 1
                processed_df[feature] = (processed_df[feature] - stats["mean"]) / std

    return processed_df


def clear_cache():
    """Clear all cached data (useful for reloading)"""
    DataCache._model = None
    DataCache._feature_stats = None
    DataCache._data = None
    DataCache._feature_engineered_data = None

    # Clear lru_cache
    load_model.cache_clear()
    load_feature_stats.cache_clear()

    print("Cache cleared successfully")


def get_data_info() -> Dict[str, Any]:
    """
    Get information about loaded data

    Returns:
        Dictionary with data information
    """
    info = {
        "model_loaded": DataCache._model is not None,
        "feature_stats_loaded": DataCache._feature_stats is not None,
        "data_loaded": DataCache._data is not None,
        "feature_engineered_data_loaded": DataCache._feature_engineered_data is not None,
    }

    if DataCache._data is not None:
        info["data_shape"] = DataCache._data.shape
        info["data_columns"] = list(DataCache._data.columns)

    if DataCache._feature_engineered_data is not None:
        info["feature_engineered_shape"] = DataCache._feature_engineered_data.shape

    return info


# Example usage with Gradio
if __name__ == "__main__":
    import gradio as gr

    def test_load_data():
        """Test function for Gradio interface"""
        model = load_model()
        stats = load_feature_stats()
        data = load_data()
        feature_data = load_feature_engineered_data()

        info = get_data_info()

        result = f"""
        **Data Loading Test Results:**

        - Model Loaded: {'✅' if info['model_loaded'] else '❌'}
        - Feature Stats Loaded: {'✅' if info['feature_stats_loaded'] else '❌'}
        - Data Loaded: {'✅' if info['data_loaded'] else '❌'}
        - Feature Engineered Data Loaded: {'✅' if info['feature_engineered_data_loaded'] else '❌'}

        **Data Info:**
        """

        if "data_shape" in info:
            result += f"\n- Data Shape: {info['data_shape']}"
            result += f"\n- Columns: {', '.join(info['data_columns'])}"

        if "feature_engineered_shape" in info:
            result += f"\n- Feature Engineered Shape: {info['feature_engineered_shape']}"

        # Show sample of data if available
        if not data.empty:
            result += f"\n\n**Sample Data:**\n{data.head().to_markdown()}"

        return result

    # Create simple Gradio interface for testing
    demo = gr.Interface(
        fn=test_load_data,
        inputs=[],
        outputs=gr.Markdown(),
        title="Data Loader Test",
        description="Test the data loading functions"
    )