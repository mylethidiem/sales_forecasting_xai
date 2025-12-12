from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gradio as gr


def sales_prediction_view(data, model, feature_stats, feature_engineered_data):
    """Display the sales prediction tool interface"""

    if model is None:
        return gr.Interface(
            fn=lambda: "Model not loaded. Please check if the model file exists.",
            inputs=[],
            outputs=gr.Textbox(label="Error"),
            title="Sales Prediction Tool"
        )

    if feature_engineered_data.empty:
        return gr.Interface(
            fn=lambda: "Feature engineered data not loaded.",
            inputs=[],
            outputs=gr.Textbox(label="Error"),
            title="Sales Prediction Tool"
        )

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

    # Create store options
    if has_store_names:
        store_options = [f"{store_id} - {store_names[store_id]}" for store_id in stores]
    else:
        store_options = stores

    def update_items(store_selection):
        """Update item dropdown based on selected store"""
        if has_store_names:
            store_id = int(store_selection.split(" - ")[0])
        else:
            store_id = store_selection

        store_items = feature_engineered_data[feature_engineered_data[store_col] == store_id][item_col].unique()

        if has_item_names:
            item_options = [
                f"{item_id} - {item_names[item_id]}"
                for item_id in store_items
                if item_id in item_names
            ]
        else:
            item_options = sorted(store_items)

        return gr.Dropdown(choices=item_options)

    def predict_sales(store_selection, item_selection, prediction_date, is_holiday,
                     special_event, promotion_impact, event_impact, clearance_impact,
                     launch_impact, temperature, weather_condition, humidity,
                     competition_level, supply_chain):
        """Wrapper function for prediction with all inputs"""

        # Parse store and item IDs
        if has_store_names:
            store_id = int(store_selection.split(" - ")[0])
        else:
            store_id = store_selection

        if has_item_names:
            item_id = int(item_selection.split(" - ")[0])
        else:
            item_id = item_selection

        # Collect prediction inputs
        prediction_inputs = collect_prediction_inputs_from_values(
            prediction_date, is_holiday, special_event, promotion_impact,
            event_impact, clearance_impact, launch_impact, temperature,
            weather_condition, humidity, competition_level, supply_chain
        )

        # Generate prediction and return results
        return generate_prediction(
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

    # Get initial items for first store
    initial_store = store_options[0] if store_options else None
    if initial_store:
        if has_store_names:
            initial_store_id = int(initial_store.split(" - ")[0])
        else:
            initial_store_id = initial_store

        initial_items = feature_engineered_data[feature_engineered_data[store_col] == initial_store_id][item_col].unique()

        if has_item_names:
            initial_item_options = [
                f"{item_id} - {item_names[item_id]}"
                for item_id in initial_items
                if item_id in item_names
            ]
        else:
            initial_item_options = sorted(initial_items)
    else:
        initial_item_options = []

    # Build Gradio interface
    with gr.Blocks(title="Sales Prediction Tool") as demo:
        gr.Markdown("# Sales Prediction Tool")

        with gr.Row():
            # Left column - Product Selection
            with gr.Column(scale=1):
                gr.Markdown("## Product Selection")
                store_dropdown = gr.Dropdown(
                    choices=store_options,
                    label="Select Store",
                    value=initial_store,
                    interactive=True,
                    allow_custom_value=False
                )
                item_dropdown = gr.Dropdown(
                    choices=initial_item_options,
                    label="Select Product",
                    value=initial_item_options[0] if initial_item_options else None,
                    interactive=True,
                    allow_custom_value=False
                )

                # Update items when store changes
                store_dropdown.change(
                    fn=update_items,
                    inputs=[store_dropdown],
                    outputs=[item_dropdown]
                )

            # Right column - Prediction Parameters
            with gr.Column(scale=2):
                gr.Markdown("## Prediction Parameters")

                with gr.Row():
                    with gr.Column():
                        prediction_date = gr.Textbox(
                            label="Prediction Date (YYYY-MM-DD)",
                            value=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                            interactive=True
                        )
                        is_holiday = gr.Checkbox(label="Holiday", value=False, interactive=True)
                        special_event = gr.Dropdown(
                            choices=["None", "Sale/Promotion", "Local Event",
                                   "Inventory Clearance", "New Product Launch"],
                            label="Special Event",
                            value="None",
                            interactive=True
                        )
                        promotion_impact = gr.Slider(-50, 100, value=20, label="Promotion Impact (%)", interactive=True)
                        event_impact = gr.Slider(-20, 50, value=10, label="Event Impact (%)", interactive=True)
                        clearance_impact = gr.Slider(-70, 30, value=-10, label="Clearance Impact (%)", interactive=True)
                        launch_impact = gr.Slider(0, 200, value=50, label="Launch Impact (%)", interactive=True)

                    with gr.Column():
                        temperature = gr.Slider(-10.0, 40.0, value=20.0, label="Temperature (°C)", interactive=True)
                        weather_condition = gr.Dropdown(
                            choices=["Clear", "Cloudy", "Rainy", "Snowy", "Stormy"],
                            label="Weather Condition",
                            value="Clear",
                            interactive=True
                        )
                        gr.Markdown("*Note: Weather impacts vary by product category*")

                    with gr.Column():
                        humidity = gr.Slider(0, 100, value=50, label="Humidity (%)", interactive=True)
                        competition_level = gr.Radio(
                            choices=["Low", "Medium", "High"],
                            label="Competition Level",
                            value="Medium",
                            interactive=True
                        )
                        supply_chain = gr.Radio(
                            choices=["Constrained", "Normal", "Abundant"],
                            label="Supply Chain Status",
                            value="Normal",
                            interactive=True
                        )

                predict_btn = gr.Button("Predict Sales", variant="primary")

        # Output section
        gr.Markdown("## Prediction Results")
        with gr.Row():
            result_text = gr.Textbox(label="Results", lines=10)
            result_plot1 = gr.Plot(label="Sales History")

        with gr.Row():
            result_plot2 = gr.Plot(label="Weekly Pattern")
            result_plot3 = gr.Plot(label="Feature Importance")

        # Connect button to prediction function
        predict_btn.click(
            fn=predict_sales,
            inputs=[
                store_dropdown, item_dropdown, prediction_date, is_holiday,
                special_event, promotion_impact, event_impact, clearance_impact,
                launch_impact, temperature, weather_condition, humidity,
                competition_level, supply_chain
            ],
            outputs=[result_text, result_plot1, result_plot2, result_plot3]
        )

    return demo


def create_name_mappings(df, store_col, item_col, has_store_names, has_item_names):
    """Create mapping dictionaries for store and item names"""

    store_names = {}
    item_names = {}

    if has_store_names:
        # Create store ID to name mapping
        for _, row in df[[store_col, "store_name"]].drop_duplicates().iterrows():
            store_names[row[store_col]] = row["store_name"]

    if has_item_names:
        # Create item ID to name mapping
        for _, row in df[[item_col, "item_name"]].drop_duplicates().iterrows():
            item_names[row[item_col]] = row["item_name"]

    return store_names, item_names


def create_product_selection_sidebar(
    df,
    stores,
    store_col,
    item_col,
    has_store_names,
    has_item_names,
    store_names,
    item_names,
):
    """Create sidebar for store and product selection"""
    # This function is kept for compatibility but not used in Gradio version
    # The logic is integrated into sales_prediction_view
    pass


def collect_prediction_inputs():
    """Collect all prediction inputs from the user"""
    # This function is kept for compatibility but adapted for Gradio
    # See collect_prediction_inputs_from_values instead
    pass


def collect_prediction_inputs_from_values(
    prediction_date_str, is_holiday, special_event, promotion_impact,
    event_impact, clearance_impact, launch_impact, temperature,
    weather_condition, humidity, competition_level, supply_chain
):
    """Collect all prediction inputs from provided values"""

    # Parse date
    prediction_date = datetime.strptime(prediction_date_str, "%Y-%m-%d").date()

    # Calculate special event factor
    special_event_factor = 1.0
    if special_event == "Sale/Promotion":
        special_event_factor = promotion_impact / 100 + 1.0
    elif special_event == "Local Event":
        special_event_factor = event_impact / 100 + 1.0
    elif special_event == "Inventory Clearance":
        special_event_factor = clearance_impact / 100 + 1.0
    elif special_event == "New Product Launch":
        special_event_factor = launch_impact / 100 + 1.0

    # Determine temperature category
    if temperature < 15:
        temp_category = "Cool"
    elif temperature < 25:
        temp_category = "Warm"
    else:
        temp_category = "Hot"

    # Determine humidity level
    if humidity < 40:
        humidity_level = "Low"
    elif humidity < 70:
        humidity_level = "Medium"
    else:
        humidity_level = "High"

    # Calculate derived parameters
    month = prediction_date.month
    if month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    elif month in [9, 10, 11]:
        season = "fall"
    else:
        season = "winter"

    quarter = (prediction_date.month - 1) // 3 + 1
    day_of_week = prediction_date.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    # Calculate factors
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

    # Combined adjustment factor
    adjustment_factor = (
        special_event_factor
        * weather_factor.get(weather_condition, 1.0)
        * competition_factor.get(competition_level, 1.0)
        * supply_factor.get(supply_chain, 1.0)
        * weekend_factor
    )

    return {
        "date": prediction_date,
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
    """Generate sales prediction and display results"""

    try:
        # Find recent samples for the same store-item combination
        recent_samples = (
            feature_engineered_data[
                (feature_engineered_data[store_col] == store_id)
                & (feature_engineered_data[item_col] == item_id)
            ]
            .sort_values("date", ascending=False)
            .head(5)
        )

        if recent_samples.empty:
            return "No historical data found for this product-store combination.", None, None, None

        # Create input based on most recent sample
        input_row = prepare_prediction_input(recent_samples, prediction_inputs)

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_row])

        # Get the features that the model expects
        if hasattr(model, "feature_name_"):
            model_features = model.feature_name_
        else:
            model_features = [
                col
                for col in input_df.columns
                if col
                not in ["sales", "date", "variation_factor", "adjustment_factor"]
            ]

        # Select only the features used by the model
        X_pred = input_df[model_features]

        # Make prediction
        base_prediction = model.predict(X_pred)[0]

        # Apply adjustment factors
        adjusted_prediction = base_prediction

        # Apply the variation factor if it exists
        if "variation_factor" in input_row:
            adjusted_prediction *= input_row["variation_factor"]

        # Apply adjustment factor from user inputs
        if "adjustment_factor" in prediction_inputs:
            adjusted_prediction *= prediction_inputs["adjustment_factor"]

        # Display results
        result_text, plot1, plot2, plot3 = display_prediction_results(
            adjusted_prediction,
            base_prediction,
            store_id,
            item_id,
            prediction_inputs,
            feature_engineered_data,
            store_col,
            item_col,
            has_store_names,
            has_item_names,
            store_names,
            item_names,
            model,
            model_features,
        )

        return result_text, plot1, plot2, plot3

    except Exception as e:
        import traceback
        error_msg = f"Error making prediction: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, None, None


def prepare_prediction_input(recent_samples, prediction_inputs):
    """Prepare input row for prediction based on recent sample and user inputs"""

    # Create input row based on most recent sample
    input_row = recent_samples.iloc[0].copy()

    # Update with user inputs
    input_row["date"] = pd.to_datetime(prediction_inputs["date"])
    input_row["day"] = prediction_inputs["date"].day
    input_row["month"] = prediction_inputs["date"].month
    input_row["year"] = prediction_inputs["date"].year
    input_row["quarter"] = prediction_inputs["quarter"]
    input_row["is_holiday"] = int(prediction_inputs["is_holiday"])

    # Add day of week information
    input_row["day_of_week"] = input_row["date"].dayofweek
    input_row["day_of_month"] = input_row["date"].day
    input_row["is_weekend"] = 1 if input_row["day_of_week"] >= 5 else 0

    # Update actual temperature and humidity values if they exist in the dataframe
    if "temperature" in input_row:
        input_row["temperature"] = prediction_inputs["temperature"]

    if "humidity" in input_row:
        input_row["humidity"] = prediction_inputs["humidity"]

    # Update temperature and humidity categories
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

    # Update season
    for s in ["spring", "summer", "fall", "winter", "wet"]:
        if f"season_{s}" in input_row:
            input_row[f"season_{s}"] = 1 if s == prediction_inputs["season"] else 0

    # Set a random variation factor
    variation_factor = 1.0 + np.random.uniform(-0.02, 0.02)
    input_row["variation_factor"] = variation_factor

    return input_row


def display_prediction_results(
    prediction_value,
    base_prediction,
    store_id,
    item_id,
    prediction_inputs,
    historical_data,
    store_col,
    item_col,
    has_store_names,
    has_item_names,
    store_names,
    item_names,
    model,
    model_features,
):
    """Display prediction results with visualizations"""

    # Build result text
    result_lines = []
    result_lines.append("=" * 50)
    result_lines.append("PREDICTION RESULTS")
    result_lines.append("=" * 50)
    result_lines.append(f"\nPredicted Sales: ${prediction_value:,.2f}")

    if has_store_names:
        result_lines.append(f"Store: {store_names[store_id]}")
    else:
        result_lines.append(f"Store ID: {store_id}")

    if has_item_names:
        result_lines.append(f"Product: {item_names[item_id]}")
    else:
        result_lines.append(f"Product ID: {item_id}")

    result_lines.append(f"Date: {prediction_inputs['date'].strftime('%B %d, %Y')}")
    result_lines.append(f"Season: {prediction_inputs['season'].capitalize()}")
    if prediction_inputs["is_holiday"]:
        result_lines.append("Holiday: Yes")

    # Adjustment details
    result_lines.append(f"\n{'='*50}")
    result_lines.append("ADJUSTMENT DETAILS")
    result_lines.append("="*50)
    result_lines.append(f"Base prediction: ${base_prediction:.2f}")
    result_lines.append(f"Final prediction: ${prediction_value:.2f}")
    result_lines.append(f"Total adjustment: {prediction_inputs['adjustment_factor']:.2f}x")
    result_lines.append(f"\nEvent: {prediction_inputs['special_event']}")
    result_lines.append(f"Weather: {prediction_inputs['weather_condition']}")
    result_lines.append(f"Competition: {prediction_inputs['competition_level']}")
    result_lines.append(f"Supply: {prediction_inputs['supply_chain']}")
    result_lines.append(f"Weekend: {'Yes' if prediction_inputs['is_weekend'] else 'No'}")
    result_lines.append(f"Holiday: {'Yes' if prediction_inputs['is_holiday'] else 'No'}")

    # Get historical context
    historical = historical_data[
        (historical_data[store_col] == store_id)
        & (historical_data[item_col] == item_id)
    ].sort_values("date")

    if "sales" in historical.columns and len(historical) > 0:
        last_value = historical["sales"].iloc[-1]
        last_date = historical["date"].iloc[-1]
        avg_sales = historical["sales"].mean()
        max_sales = historical["sales"].max()
        max_date = historical.loc[historical["sales"].idxmax(), "date"]

        result_lines.append(f"\n{'='*50}")
        result_lines.append("HISTORICAL CONTEXT")
        result_lines.append("="*50)
        result_lines.append(f"Historical Average: ${avg_sales:,.2f}")
        result_lines.append(f"Period: {historical['date'].min().strftime('%b %d, %Y')} to {historical['date'].max().strftime('%b %d, %Y')}")
        result_lines.append(f"\nLast Recorded Sales: ${last_value:,.2f}")
        result_lines.append(f"Date: {last_date.strftime('%b %d, %Y')}")
        result_lines.append(f"\nHistorical Maximum: ${max_sales:,.2f}")
        result_lines.append(f"Date: {max_date.strftime('%b %d, %Y')}")

    result_text = "\n".join(result_lines)

    # Create visualizations
    plot1 = display_historical_context(historical, prediction_inputs["date"], prediction_value)
    plot2 = display_weekly_pattern(historical, prediction_inputs["date"])
    plot3 = display_feature_importance(model, model_features)

    return result_text, plot1, plot2, plot3


def display_historical_context(historical_data, prediction_date, prediction_value):
    """Display historical context visualizations"""

    if "sales" not in historical_data.columns or historical_data.empty:
        return None

    # Limit to last 2 months
    last_date = historical_data["date"].max()
    two_months_ago = last_date - pd.Timedelta(days=60)
    recent_history = historical_data[historical_data["date"] >= two_months_ago].copy()

    if recent_history.empty:
        return None

    # Plot recent sales history
    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Plot historical sales
    ax.plot(
        recent_history["date"],
        recent_history["sales"],
        "b-",
        label="Sales",
    )

    # Add the prediction point
    ax.scatter(
        prediction_date,
        prediction_value,
        color="red",
        s=60,
        label="Prediction",
    )

    # Add moving average
    if len(recent_history) > 7:
        recent_history["MA7"] = recent_history["sales"].rolling(window=7).mean()
        ax.plot(
            recent_history["date"],
            recent_history["MA7"],
            "g--",
            label="7-Day Avg",
        )

    ax.set_xlabel("")
    ax.set_ylabel("Sales ($)")
    ax.set_title("Last 60 Days Sales History")
    ax.legend(loc="upper left", fontsize="x-small")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig


def display_weekly_pattern(recent_history, prediction_date):
    """Display weekly sales pattern visualization"""

    if len(recent_history) < 7:
        return None

    # Add day of week
    recent_history = recent_history.copy()
    recent_history["day_of_week"] = recent_history["date"].dt.dayofweek
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Group by day of week
    day_sales = recent_history.groupby("day_of_week")["sales"].mean()
    day_sales_df = pd.DataFrame(
        {
            "day_name": [day_names[i] for i in range(7) if i in day_sales.index],
            "sales": [day_sales[i] for i in range(7) if i in day_sales.index],
        }
    )

    # Plot
    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Plot day of week pattern
    sns.barplot(x="day_name", y="sales", data=day_sales_df, ax=ax)

    # Highlight the day of the prediction
    prediction_day = prediction_date.weekday()
    for i, patch in enumerate(ax.patches):
        if day_sales_df.iloc[i]["day_name"] == day_names[prediction_day]:
            patch.set_facecolor("red")

    ax.set_xlabel("")
    ax.set_ylabel("Avg Sales ($)")
    ax.set_title("Sales by Day of Week")
    plt.xticks(rotation=45, fontsize=8)
    fig.tight_layout()

    return fig


def display_feature_importance(model, model_features):
    """Display feature importance visualization"""

    if not hasattr(model, "feature_importances_"):
        return None

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame with feature importances
    importance_df = (
        pd.DataFrame({"Feature": model_features, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(8)
    )

    # Clean feature names for display
    importance_df["Feature"] = importance_df["Feature"].apply(
        lambda x: x.replace("_", " ").title()
    )

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(6, 2.5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    ax.set_title("Top Factors Influencing Sales Prediction")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()

    return fig