from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import io
import base64

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Gradio
import numpy as np
import pandas as pd
import seaborn as sns
import gradio as gr


def create_sales_prediction_interface(data, model, feature_stats, feature_engineered_data):
    """Create Gradio interface for sales prediction"""

    if model is None:
        return gr.Markdown("⚠️ **Error:** Model not loaded. Please check if the model file exists.")

    if feature_engineered_data.empty:
        return gr.Markdown("⚠️ **Error:** Feature engineered data not loaded.")

    # Determine store and item column names
    store_col = "store_id" if "store_id" in feature_engineered_data.columns else "store"
    item_col = "item_id" if "item_id" in feature_engineered_data.columns else "item"

    # Check for store/item name columns
    has_store_names = "store_name" in feature_engineered_data.columns
    has_item_names = "item_name" in feature_engineered_data.columns

    # Create mapping dictionaries for names if available
    store_names, item_names = create_name_mappings(
        feature_engineered_data, store_col, item_col, has_store_names, has_item_names
    )

    # Get unique store and item lists
    stores = sorted(feature_engineered_data[store_col].unique())

    # Prepare store options
    if has_store_names:
        store_options = [f"{store_id} - {store_names[store_id]}" for store_id in stores]
    else:
        store_options = [str(store_id) for store_id in stores]

    def update_items(store_selection):
        """Update item dropdown based on selected store"""
        # Extract store_id from selection
        if has_store_names:
            store_id = int(store_selection.split(" - ")[0])
        else:
            store_id = int(store_selection)

        # Get items for selected store
        store_items = feature_engineered_data[feature_engineered_data[store_col] == store_id][item_col].unique()

        # Prepare item options
        if has_item_names:
            item_options = [
                f"{item_id} - {item_names[item_id]}"
                for item_id in sorted(store_items)
                if item_id in item_names
            ]
        else:
            item_options = [str(item_id) for item_id in sorted(store_items)]

        return gr.Dropdown(choices=item_options, value=item_options[0] if item_options else None)

    def make_prediction(
        store_selection,
        item_selection,
        prediction_date,
        is_holiday,
        special_event,
        special_event_impact,
        temperature,
        weather_condition,
        humidity,
        competition_level,
        supply_chain
    ):
        """Generate sales prediction"""
        try:
            # Extract IDs from selections
            if has_store_names:
                store_id = int(store_selection.split(" - ")[0])
            else:
                store_id = int(store_selection)

            if has_item_names:
                item_id = int(item_selection.split(" - ")[0])
            else:
                item_id = int(item_selection)

            # Parse date
            if isinstance(prediction_date, str):
                pred_date = datetime.strptime(prediction_date, "%Y-%m-%d").date()
            else:
                pred_date = prediction_date

            # Calculate derived parameters
            prediction_inputs = calculate_prediction_parameters(
                pred_date,
                is_holiday,
                special_event,
                special_event_impact,
                temperature,
                weather_condition,
                humidity,
                competition_level,
                supply_chain
            )

            # Generate prediction
            result_html, plot1, plot2, plot3 = generate_prediction(
                feature_engineered_data,
                model,
                store_id,
                item_id,
                store_col,
                item_col,
                prediction_inputs,
                has_store_names,
                has_item_names,
                store_names,
                item_names,
            )

            return result_html, plot1, plot2, plot3

        except Exception as e:
            error_html = f"""
            <div style='padding: 20px; background-color: #fee; border: 1px solid #c00; border-radius: 5px;'>
                <h3>❌ Error Making Prediction</h3>
                <p>{str(e)}</p>
                <p><small>Please ensure all inputs are valid and historical data exists for this store-item combination.</small></p>
            </div>
            """
            return error_html, None, None, None

    # Create Gradio interface with proper layout
    with gr.Blocks() as prediction_interface:
        gr.Markdown("# 🔮 Sales Prediction Tool")
        gr.Markdown("Generate sales forecasts based on historical data and market conditions")

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Product Selection")

                store_dropdown = gr.Dropdown(
                    choices=store_options,
                    value=store_options[0] if store_options else None,
                    label="Select Store",
                    interactive=True
                )

                # Get initial items for first store
                if has_store_names:
                    initial_store_id = int(store_options[0].split(" - ")[0])
                else:
                    initial_store_id = int(store_options[0])

                initial_items = feature_engineered_data[
                    feature_engineered_data[store_col] == initial_store_id
                ][item_col].unique()

                if has_item_names:
                    initial_item_options = [
                        f"{item_id} - {item_names[item_id]}"
                        for item_id in sorted(initial_items)
                        if item_id in item_names
                    ]
                else:
                    initial_item_options = [str(item_id) for item_id in sorted(initial_items)]

                item_dropdown = gr.Dropdown(
                    choices=initial_item_options,
                    value=initial_item_options[0] if initial_item_options else None,
                    label="Select Product",
                    interactive=True
                )

                # Update items when store changes
                store_dropdown.change(
                    fn=update_items,
                    inputs=[store_dropdown],
                    outputs=[item_dropdown]
                )

                gr.Markdown("### Date & Conditions")

                prediction_date = gr.Textbox(
                    label="Prediction Date",
                    value=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                    placeholder="YYYY-MM-DD"
                )

                is_holiday = gr.Checkbox(label="Holiday", value=False)

                gr.Markdown("### Special Events")

                special_event = gr.Dropdown(
                    choices=["None", "Sale/Promotion", "Local Event", "Inventory Clearance", "New Product Launch"],
                    value="None",
                    label="Special Event"
                )

                special_event_impact = gr.Slider(
                    minimum=-70,
                    maximum=200,
                    value=0,
                    step=5,
                    label="Event Impact (%)",
                    info="Adjust impact of special event on sales"
                )

                gr.Markdown("### Weather Conditions")

                temperature = gr.Slider(
                    minimum=-10.0,
                    maximum=40.0,
                    value=20.0,
                    step=0.5,
                    label="Temperature (°C)"
                )

                weather_condition = gr.Dropdown(
                    choices=["Clear", "Cloudy", "Rainy", "Snowy", "Stormy"],
                    value="Clear",
                    label="Weather Condition"
                )

                humidity = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Humidity (%)"
                )

                gr.Markdown("### Market Conditions")

                competition_level = gr.Radio(
                    choices=["Low", "Medium", "High"],
                    value="Medium",
                    label="Competition Level"
                )

                supply_chain = gr.Radio(
                    choices=["Constrained", "Normal", "Abundant"],
                    value="Normal",
                    label="Supply Chain Status"
                )

                predict_btn = gr.Button("🎯 Predict Sales", variant="primary", size="lg")

            # Right column - Results
            with gr.Column(scale=2):
                gr.Markdown("### Prediction Results")

                result_output = gr.HTML(label="Results")

                with gr.Row():
                    plot1_output = gr.Plot(label="Sales History")
                    plot2_output = gr.Plot(label="Weekly Pattern")

                plot3_output = gr.Plot(label="Feature Importance")

        # Connect prediction button
        predict_btn.click(
            fn=make_prediction,
            inputs=[
                store_dropdown,
                item_dropdown,
                prediction_date,
                is_holiday,
                special_event,
                special_event_impact,
                temperature,
                weather_condition,
                humidity,
                competition_level,
                supply_chain
            ],
            outputs=[result_output, plot1_output, plot2_output, plot3_output]
        )

    return prediction_interface


def create_name_mappings(df, store_col, item_col, has_store_names, has_item_names):
    """Create mapping dictionaries for store and item names"""
    store_names = {}
    item_names = {}

    if has_store_names:
        for _, row in df[[store_col, "store_name"]].drop_duplicates().iterrows():
            store_names[row[store_col]] = row["store_name"]

    if has_item_names:
        for _, row in df[[item_col, "item_name"]].drop_duplicates().iterrows():
            item_names[row[item_col]] = row["item_name"]

    return store_names, item_names


def calculate_prediction_parameters(
    pred_date,
    is_holiday,
    special_event,
    special_event_impact,
    temperature,
    weather_condition,
    humidity,
    competition_level,
    supply_chain
):
    """Calculate all prediction parameters"""

    # Temperature category
    if temperature < 15:
        temp_category = "Cool"
    elif temperature < 25:
        temp_category = "Warm"
    else:
        temp_category = "Hot"

    # Humidity level
    if humidity < 40:
        humidity_level = "Low"
    elif humidity < 70:
        humidity_level = "Medium"
    else:
        humidity_level = "High"

    # Season
    month = pred_date.month
    if month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    elif month in [9, 10, 11]:
        season = "fall"
    else:
        season = "winter"

    quarter = (pred_date.month - 1) // 3 + 1
    day_of_week = pred_date.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    # Calculate adjustment factors
    special_event_factor = 1.0 + (special_event_impact / 100) if special_event != "None" else 1.0

    weather_factor = {
        "Clear": 1.0,
        "Cloudy": 0.95,
        "Rainy": 0.9,
        "Snowy": 0.8,
        "Stormy": 0.7,
    }

    competition_factor = {"Low": 1.1, "Medium": 1.0, "High": 0.9}
    supply_factor = {"Constrained": 0.9, "Normal": 1.0, "Abundant": 1.05}
    weekend_factor = 1.15 if is_weekend else 1.0

    adjustment_factor = (
        special_event_factor
        * weather_factor.get(weather_condition, 1.0)
        * competition_factor.get(competition_level, 1.0)
        * supply_factor.get(supply_chain, 1.0)
        * weekend_factor
    )

    return {
        "date": pred_date,
        "is_holiday": is_holiday,
        "temperature": temperature,
        "temp_category": temp_category,
        "humidity": humidity,
        "humidity_level": humidity_level,
        "season": season,
        "quarter": quarter,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "special_event": special_event,
        "weather_condition": weather_condition,
        "competition_level": competition_level,
        "supply_chain": supply_chain,
        "adjustment_factor": adjustment_factor,
    }


def generate_prediction(
    feature_engineered_data,
    model,
    store_id,
    item_id,
    store_col,
    item_col,
    prediction_inputs,
    has_store_names,
    has_item_names,
    store_names,
    item_names,
):
    """Generate sales prediction and return formatted results"""

    # Find recent samples
    recent_samples = (
        feature_engineered_data[
            (feature_engineered_data[store_col] == store_id)
            & (feature_engineered_data[item_col] == item_id)
        ]
        .sort_values("date", ascending=False)
        .head(5)
    )

    if recent_samples.empty:
        error_html = """
        <div style='padding: 20px; background-color: #fee; border: 1px solid #c00; border-radius: 5px;'>
            <h3>❌ No Historical Data</h3>
            <p>No historical data found for this product-store combination.</p>
        </div>
        """
        return error_html, None, None, None

    # Prepare input
    input_row = prepare_prediction_input(recent_samples, prediction_inputs)
    input_df = pd.DataFrame([input_row])

    # Get model features
    if hasattr(model, "feature_name_"):
        model_features = model.feature_name_
    else:
        model_features = [
            col for col in input_df.columns
            if col not in ["sales", "date", "variation_factor", "adjustment_factor"]
        ]

    # Make prediction
    X_pred = input_df[model_features]
    base_prediction = model.predict(X_pred)[0]

    # Apply adjustments
    adjusted_prediction = base_prediction
    if "variation_factor" in input_row:
        adjusted_prediction *= input_row["variation_factor"]
    adjusted_prediction *= prediction_inputs["adjustment_factor"]

    # Get historical data
    historical = feature_engineered_data[
        (feature_engineered_data[store_col] == store_id)
        & (feature_engineered_data[item_col] == item_id)
    ].sort_values("date")

    # Generate HTML results
    result_html = generate_result_html(
        adjusted_prediction,
        base_prediction,
        store_id,
        item_id,
        prediction_inputs,
        historical,
        has_store_names,
        has_item_names,
        store_names,
        item_names
    )

    # Generate plots
    plot1 = create_history_plot(historical, prediction_inputs["date"], adjusted_prediction)
    plot2 = create_weekly_pattern_plot(historical, prediction_inputs["date"])
    plot3 = create_feature_importance_plot(model, model_features)

    return result_html, plot1, plot2, plot3


def prepare_prediction_input(recent_samples, prediction_inputs):
    """Prepare input row for prediction"""
    input_row = recent_samples.iloc[0].copy()

    # Update with user inputs
    input_row["date"] = pd.to_datetime(prediction_inputs["date"])
    input_row["day"] = prediction_inputs["date"].day
    input_row["month"] = prediction_inputs["date"].month
    input_row["year"] = prediction_inputs["date"].year
    input_row["quarter"] = prediction_inputs["quarter"]
    input_row["is_holiday"] = int(prediction_inputs["is_holiday"])
    input_row["day_of_week"] = input_row["date"].dayofweek
    input_row["day_of_month"] = input_row["date"].day
    input_row["is_weekend"] = 1 if input_row["day_of_week"] >= 5 else 0

    # Update weather
    if "temperature" in input_row:
        input_row["temperature"] = prediction_inputs["temperature"]
    if "humidity" in input_row:
        input_row["humidity"] = prediction_inputs["humidity"]

    # Update categories
    for category in ["Cool", "Warm", "Hot"]:
        if f"temp_category_{category}" in input_row:
            input_row[f"temp_category_{category}"] = (
                1 if category == prediction_inputs["temp_category"] else 0
            )

    for level in ["Low", "Medium", "High"]:
        if f"humidity_level_{level}" in input_row:
            input_row[f"humidity_level_{level}"] = (
                1 if level == prediction_inputs["humidity_level"] else 0
            )

    for s in ["spring", "summer", "fall", "winter", "wet"]:
        if f"season_{s}" in input_row:
            input_row[f"season_{s}"] = 1 if s == prediction_inputs["season"] else 0

    input_row["variation_factor"] = 1.0 + np.random.uniform(-0.02, 0.02)

    return input_row


def generate_result_html(
    prediction,
    base_prediction,
    store_id,
    item_id,
    prediction_inputs,
    historical,
    has_store_names,
    has_item_names,
    store_names,
    item_names
):
    """Generate HTML for prediction results"""

    # Calculate historical stats
    avg_sales = historical["sales"].mean() if "sales" in historical.columns else 0
    last_value = historical["sales"].iloc[-1] if len(historical) > 0 and "sales" in historical.columns else 0
    max_sales = historical["sales"].max() if "sales" in historical.columns else 0

    # Format store/item names
    store_name = store_names[store_id] if has_store_names else f"Store {store_id}"
    item_name = item_names[item_id] if has_item_names else f"Item {item_id}"

    html = f"""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='margin: 0 0 10px 0;'>💰 Predicted Sales: ${prediction:,.2f}</h2>
        <p style='margin: 5px 0; opacity: 0.9;'><strong>Store:</strong> {store_name}</p>
        <p style='margin: 5px 0; opacity: 0.9;'><strong>Product:</strong> {item_name}</p>
        <p style='margin: 5px 0; opacity: 0.9;'><strong>Date:</strong> {prediction_inputs['date'].strftime('%B %d, %Y')} ({prediction_inputs['season'].capitalize()})</p>
        {f"<p style='margin: 5px 0; opacity: 0.9;'><strong>Holiday:</strong> Yes 🎉</p>" if prediction_inputs['is_holiday'] else ""}
    </div>

    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;'>
        <div style='padding: 15px; background-color: #f0f7ff; border-radius: 8px; border-left: 4px solid #0F6CBD;'>
            <div style='font-size: 12px; color: #666; margin-bottom: 5px;'>Historical Average</div>
            <div style='font-size: 24px; font-weight: bold; color: #0F6CBD;'>${avg_sales:,.2f}</div>
        </div>
        <div style='padding: 15px; background-color: #f0fff4; border-radius: 8px; border-left: 4px solid #2E7D32;'>
            <div style='font-size: 12px; color: #666; margin-bottom: 5px;'>Last Recorded</div>
            <div style='font-size: 24px; font-weight: bold; color: #2E7D32;'>${last_value:,.2f}</div>
        </div>
        <div style='padding: 15px; background-color: #fff0f0; border-radius: 8px; border-left: 4px solid #C4314B;'>
            <div style='font-size: 12px; color: #666; margin-bottom: 5px;'>Historical Max</div>
            <div style='font-size: 24px; font-weight: bold; color: #C4314B;'>${max_sales:,.2f}</div>
        </div>
    </div>

    <details style='padding: 15px; background-color: #fafafa; border-radius: 8px; margin-bottom: 20px;'>
        <summary style='cursor: pointer; font-weight: bold; margin-bottom: 10px;'>📊 Adjustment Details</summary>
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px;'>
            <div><strong>Base Prediction:</strong> ${base_prediction:,.2f}</div>
            <div><strong>Final Prediction:</strong> ${prediction:,.2f}</div>
            <div><strong>Total Adjustment:</strong> {prediction_inputs['adjustment_factor']:.2f}x</div>
            <div><strong>Event:</strong> {prediction_inputs['special_event']}</div>
            <div><strong>Weather:</strong> {prediction_inputs['weather_condition']}</div>
            <div><strong>Competition:</strong> {prediction_inputs['competition_level']}</div>
            <div><strong>Supply Chain:</strong> {prediction_inputs['supply_chain']}</div>
            <div><strong>Weekend:</strong> {'Yes' if prediction_inputs['is_weekend'] else 'No'}</div>
        </div>
    </details>
    """

    return html


def create_history_plot(historical, prediction_date, prediction_value):
    """Create sales history plot"""
    if "sales" not in historical.columns or historical.empty:
        return None

    last_date = historical["date"].max()
    two_months_ago = last_date - pd.Timedelta(days=60)
    recent_history = historical[historical["date"] >= two_months_ago].copy()

    if recent_history.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(recent_history["date"], recent_history["sales"], 'b-', label="Sales", linewidth=2)
    ax.scatter(prediction_date, prediction_value, color="red", s=100, label="Prediction", zorder=5)

    if len(recent_history) > 7:
        recent_history["MA7"] = recent_history["sales"].rolling(window=7).mean()
        ax.plot(recent_history["date"], recent_history["MA7"], 'g--', label="7-Day Avg", alpha=0.7)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Sales ($)", fontsize=10)
    ax.set_title("Last 60 Days Sales History", fontsize=12, fontweight='bold')
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig


def create_weekly_pattern_plot(historical, prediction_date):
    """Create weekly pattern plot"""
    if len(historical) < 7 or "sales" not in historical.columns:
        return None

    recent_history = historical.copy()
    recent_history["day_of_week"] = recent_history["date"].dt.dayofweek
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    day_sales = recent_history.groupby("day_of_week")["sales"].mean()
    day_sales_df = pd.DataFrame({
        "day_name": [day_names[i] for i in range(7) if i in day_sales.index],
        "sales": [day_sales[i] for i in range(7) if i in day_sales.index],
    })

    fig, ax = plt.subplots(figsize=(10, 4))

    bars = ax.bar(day_sales_df["day_name"], day_sales_df["sales"])

    # Highlight prediction day
    prediction_day = prediction_date.weekday()
    for i, (bar, day_name) in enumerate(zip(bars, day_sales_df["day_name"])):
        if day_name == day_names[prediction_day]:
            bar.set_color('red')
            bar.set_alpha(0.8)

    ax.set_xlabel("Day of Week", fontsize=10)
    ax.set_ylabel("Average Sales ($)", fontsize=10)
    ax.set_title("Sales by Day of Week", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    fig.tight_layout()

    return fig


def create_feature_importance_plot(model, model_features):
    """Create feature importance plot"""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": model_features, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(8)
    )

    importance_df["Feature"] = importance_df["Feature"].apply(
        lambda x: x.replace("_", " ").title()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color='steelblue')
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.set_title("Top Factors Influencing Sales Prediction", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()

    return fig