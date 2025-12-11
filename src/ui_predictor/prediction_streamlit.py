from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def sales_prediction_view(data, model, feature_stats, feature_engineered_data):
    """Display the sales prediction tool interface"""

    st.title("Sales Prediction Tool")

    if model is None:
        st.error("Model not loaded. Please check if the model file exists.")
        return

    if feature_engineered_data.empty:
        st.error("Feature engineered data not loaded.")
        return

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

    # Create sidebar for selections
    store_id, item_id = create_product_selection_sidebar(
        feature_engineered_data,
        stores,
        store_col,
        item_col,
        has_store_names,
        has_item_names,
        store_names,
        item_names,
    )

    # Main form content
    st.subheader("Prediction Parameters")

    prediction_inputs = collect_prediction_inputs()

    # Make prediction button
    if st.button("Predict Sales"):
        generate_prediction(
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

    with st.sidebar:
        st.header("Product Selection")

        # Store selection with names if available
        if has_store_names:
            store_options = [
                f"{store_id} - {store_names[store_id]}" for store_id in stores
            ]
            selected_store_option = st.selectbox("Select Store", options=store_options)
            store_id = int(selected_store_option.split(" - ")[0])
        else:
            store_id = st.selectbox("Select Store ID", options=stores)

        # Get items for the selected store
        store_items = df[df[store_col] == store_id][item_col].unique()

        # Item selection with names if available
        if has_item_names:
            item_options = [
                f"{item_id} - {item_names[item_id]}"
                for item_id in store_items
                if item_id in item_names
            ]
            selected_item_option = st.selectbox("Select Product", options=item_options)
            item_id = int(selected_item_option.split(" - ")[0])
        else:
            item_id = st.selectbox("Select Product ID", options=sorted(store_items))

    return store_id, item_id


def collect_prediction_inputs():
    """Collect all prediction inputs from the user"""

    col1, col2, col3 = st.columns(3)

    with col1:
        # Date selection
        prediction_date = st.date_input(
            "Prediction Date", datetime.now().date() + timedelta(days=1)
        )

        # Holiday checkbox
        is_holiday = st.checkbox("Holiday", value=False)

        # Special events that might affect sales
        special_event = st.selectbox(
            "Special Event",
            [
                "None",
                "Sale/Promotion",
                "Local Event",
                "Inventory Clearance",
                "New Product Launch",
            ],
        )
        special_event_factor = 1.0
        if special_event == "Sale/Promotion":
            special_event_factor = (
                st.slider("Promotion Impact (%)", -50, 100, 20) / 100 + 1.0
            )
        elif special_event == "Local Event":
            special_event_factor = (
                st.slider("Event Impact (%)", -20, 50, 10) / 100 + 1.0
            )
        elif special_event == "Inventory Clearance":
            special_event_factor = (
                st.slider("Clearance Impact (%)", -70, 30, -10) / 100 + 1.0
            )
        elif special_event == "New Product Launch":
            special_event_factor = (
                st.slider("Launch Impact (%)", 0, 200, 50) / 100 + 1.0
            )

    with col2:
        # Temperature slider
        temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0)

        # Determine temperature category based on temperature value
        if temperature < 15:
            temp_category = "Cool"
        elif temperature < 25:
            temp_category = "Warm"
        else:
            temp_category = "Hot"

        st.write(f"Temperature Category: {temp_category}")

        # Precipitation/Weather
        weather_condition = st.selectbox(
            "Weather Condition", ["Clear", "Cloudy", "Rainy", "Snowy", "Stormy"]
        )

        # Different product categories are affected differently by weather
        st.write("Note: Weather impacts vary by product category")

    with col3:
        # Humidity slider
        humidity = st.slider("Humidity (%)", 0, 100, 50)

        # Determine humidity level
        if humidity < 40:
            humidity_level = "Low"
        elif humidity < 70:
            humidity_level = "Medium"
        else:
            humidity_level = "High"

        st.write(f"Humidity Level: {humidity_level}")

        # Competition intensity
        competition_level = st.select_slider(
            "Competition Level", options=["Low", "Medium", "High"], value="Medium"
        )

        # Supply chain status
        supply_chain = st.select_slider(
            "Supply Chain Status",
            options=["Constrained", "Normal", "Abundant"],
            value="Normal",
        )

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

    # Calculate a combined factor based on all special conditions
    weather_factor = {
        "Clear": 1.0,
        "Cloudy": 0.95,
        "Rainy": 0.9,
        "Snowy": 0.8,
        "Stormy": 0.7,
    }

    competition_factor = {"Low": 1.1, "Medium": 1.0, "High": 0.9}

    supply_factor = {"Constrained": 0.9, "Normal": 1.0, "Abundant": 1.05}

    # Weekend factor (weekends might have different sales patterns)
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

    with st.spinner("Generating prediction..."):
        try:
            # Find recent samples for the same store-item combination
            recent_samples = (
                feature_engineered_data[
                    (feature_engineered_data[store_col] == store_id)
                    & (feature_engineered_data[item_col] == item_id)
                ]
                .sort_values("date", ascending=False)
                .head(5)  # Get more samples for better prediction context
            )

            if recent_samples.empty:
                st.error("No historical data found for this product-store combination.")
                return

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

            # Apply adjustment factor from user inputs (special events, weather, etc.)
            if "adjustment_factor" in prediction_inputs:
                adjusted_prediction *= prediction_inputs["adjustment_factor"]

                # Log the adjustment in a more compact format
                with st.expander("Adjustment Details"):
                    adj_col1, adj_col2, adj_col3 = st.columns(3)

                    with adj_col1:
                        st.write(f"Base prediction: ${base_prediction:.2f}")
                        st.write(f"Final prediction: ${adjusted_prediction:.2f}")
                        st.write(
                            f"Total adjustment: {prediction_inputs['adjustment_factor']:.2f}x"
                        )

                    with adj_col2:
                        st.write(f"Event: {prediction_inputs['special_event']}")
                        st.write(f"Weather: {prediction_inputs['weather_condition']}")
                        st.write(
                            f"Competition: {prediction_inputs['competition_level']}"
                        )

                    with adj_col3:
                        st.write(f"Supply: {prediction_inputs['supply_chain']}")
                        st.write(
                            f"Weekend: {'Yes' if prediction_inputs['is_weekend'] else 'No'}"
                        )
                        st.write(
                            f"Holiday: {'Yes' if prediction_inputs['is_holiday'] else 'No'}"
                        )

            # Display results
            display_prediction_results(
                adjusted_prediction,
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

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info(
                "Please ensure all required features are available in the input data."
            )
            # Print more detailed error for debugging
            import traceback

            st.error(traceback.format_exc())


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

    # Set a random variation factor to ensure predictions aren't identical
    # This simulates real-world variability even when inputs are similar
    # Adjust the scale (0.02 = ±2%) based on how much variation you want
    variation_factor = 1.0 + np.random.uniform(-0.02, 0.02)

    # Store this factor for logging and debugging purposes
    input_row["variation_factor"] = variation_factor

    return input_row


def display_prediction_results(
    prediction_value,
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

    st.header("Prediction Results")

    # Create results in columns
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        # Display prediction with context
        st.metric(label="Predicted Sales", value=f"${prediction_value:,.2f}")

        # Display store and item info
        if has_store_names:
            st.write(f"**Store:** {store_names[store_id]}")
        else:
            st.write(f"**Store ID:** {store_id}")

        if has_item_names:
            st.write(f"**Product:** {item_names[item_id]}")
        else:
            st.write(f"**Product ID:** {item_id}")

        st.write(f"**Date:** {prediction_inputs['date'].strftime('%B %d, %Y')}")
        st.write(f"**Season:** {prediction_inputs['season'].capitalize()}")
        if prediction_inputs["is_holiday"]:
            st.write("**Holiday:** Yes")

    with res_col2:
        # Get historical context
        historical = historical_data[
            (historical_data[store_col] == store_id)
            & (historical_data[item_col] == item_id)
        ].sort_values("date")

        if "sales" in historical.columns:
            # Calculate key statistics
            last_value = historical["sales"].iloc[-1] if len(historical) > 0 else 0
            last_date = historical["date"].iloc[-1] if len(historical) > 0 else None

            avg_sales = historical["sales"].mean()

            max_sales = historical["sales"].max()
            max_date = (
                historical.loc[historical["sales"].idxmax(), "date"]
                if len(historical) > 0
                else None
            )

            # Display average and trend with dates
            st.metric(
                label="Historical Average",
                value=f"${avg_sales:,.2f}",
            )
            st.write(
                f"**Period:** {historical['date'].min().strftime('%b %d, %Y')} to {historical['date'].max().strftime('%b %d, %Y')}"
            )

            st.metric(
                label="Last Recorded Sales",
                value=f"${last_value:,.2f}",
            )
            if last_date is not None:
                st.write(f"**Date:** {last_date.strftime('%b %d, %Y')}")

            st.metric(label="Historical Maximum", value=f"${max_sales:,.2f}")
            if max_date is not None:
                st.write(f"**Date:** {max_date.strftime('%b %d, %Y')}")

    # Historical context
    display_historical_context(historical, prediction_inputs["date"], prediction_value)

    # Feature importance
    display_feature_importance(model, model_features)


def display_historical_context(historical_data, prediction_date, prediction_value):
    """Display historical context visualizations"""

    st.subheader("Recent Sales History")

    if "sales" not in historical_data.columns or historical_data.empty:
        st.info(
            "No historical sales data available for this product-store combination."
        )
        return

    # Limit to last 2 months
    last_date = historical_data["date"].max()
    two_months_ago = last_date - pd.Timedelta(days=60)
    recent_history = historical_data[historical_data["date"] >= two_months_ago]

    if recent_history.empty:
        st.info("No recent sales data available for the last 60 days.")
        return

    # Plot recent sales history - SMALLER SIZE
    fig, ax = plt.subplots(figsize=(6, 2.5))  # Reduced size

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
        s=60,  # Smaller point
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
    ax.legend(loc="upper left", fontsize="x-small")  # Smaller font
    fig.autofmt_xdate(rotation=45)  # Adjust date format
    fig.tight_layout()

    st.pyplot(fig)

    # Weekly pattern visualization
    display_weekly_pattern(recent_history, prediction_date)


def display_weekly_pattern(recent_history, prediction_date):
    """Display weekly sales pattern visualization"""

    if len(recent_history) >= 7:
        st.subheader("Weekly Sales Pattern")

        # Add day of week
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

        # Plot - SMALLER SIZE
        fig, ax = plt.subplots(figsize=(6, 2.5))  # Reduced size

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
        plt.xticks(rotation=45, fontsize=8)  # Smaller font
        fig.tight_layout()

        st.pyplot(fig)


def display_feature_importance(model, model_features):
    """Display feature importance visualization"""

    if hasattr(model, "feature_importances_"):
        st.subheader("Key Factors Influencing This Prediction")

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

        # Plot feature importances - SMALLER SIZE
        fig, ax = plt.subplots(figsize=(6, 2.5))  # Reduced size
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        ax.set_title("Top Factors Influencing Sales Prediction")
        plt.xticks(fontsize=8)  # Smaller font
        plt.yticks(fontsize=8)  # Smaller font
        fig.tight_layout()

        st.pyplot(fig)
