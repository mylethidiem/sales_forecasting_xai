import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter

# Set up plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 12


def visualize_predictions_by_store_item(test_results, output_dir="visualizations"):
    """
    Create visualizations of actual vs predicted values for each store-item combination.

    Args:
        test_results: DataFrame containing test results with columns:
                     'date', 'store_name', 'item_name', 'sales', 'prediction'
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a time series plot for each store-item combination
    store_items = test_results.groupby(["store_name", "item_name"])

    # Get total number of combinations for progress tracking
    total_combinations = len(store_items)
    print(
        f"Creating visualizations for {total_combinations} store-item combinations..."
    )

    # Counter for progress tracking
    counter = 0

    # For each store-item combination, create a plot
    for (store, item), group in store_items:
        # Sort by date to ensure proper time series order
        group = group.sort_values("date")

        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(group["date"]):
            group["date"] = pd.to_datetime(group["date"])

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot actual and predicted values
        ax.plot(
            group["date"], group["sales"], "o-", label="Actual", alpha=0.7, linewidth=2
        )
        ax.plot(
            group["date"],
            group["prediction"],
            "s--",
            label="Predicted",
            alpha=0.7,
            linewidth=2,
        )

        # Calculate error metrics for this store-item
        mae = np.mean(np.abs(group["sales"] - group["prediction"]))
        mape = (
            np.mean(np.abs((group["sales"] - group["prediction"]) / group["sales"]))
            * 100
        )

        # Add title and labels
        ax.set_title(f"Store: {store}, Item: {item}\nMAE: {mae:.2f}, MAPE: {mape:.2f}%")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")

        # Format x-axis dates
        date_formatter = DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_formatter)
        # Rotate date labels for better readability
        plt.xticks(rotation=45)

        # Add grid for easier reading
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend
        ax.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        safe_store = store.replace(" ", "_").replace("/", "_")
        safe_item = item.replace(" ", "_").replace("/", "_")
        filename = f"{safe_store}_{safe_item}.png"
        plt.savefig(os.path.join(output_dir, filename))

        # Close the figure to free memory
        plt.close(fig)

        # Update progress
        counter += 1
        if counter % 10 == 0:
            print(f"Processed {counter}/{total_combinations} combinations")

    print(f"All visualizations saved to {output_dir}/")


def visualize_aggregated_predictions(test_results, output_dir="visualizations"):
    """
    Create aggregated visualizations of actual vs predicted values by store, item, and date.

    Args:
        test_results: DataFrame containing test results
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(test_results["date"]):
        test_results["date"] = pd.to_datetime(test_results["date"])

    # 1. Aggregate by date
    daily_results = (
        test_results.groupby("date")
        .agg({"sales": "sum", "prediction": "sum"})
        .reset_index()
    )

    # Plot daily aggregated results
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        daily_results["date"],
        daily_results["sales"],
        "o-",
        label="Actual",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        daily_results["date"],
        daily_results["prediction"],
        "s--",
        label="Predicted",
        alpha=0.7,
        linewidth=2,
    )

    # Add title and labels
    ax.set_title("Total Daily Sales: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")

    # Format x-axis dates
    date_formatter = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_daily_sales.png"))
    plt.close(fig)

    # 2. Aggregate by store
    store_results = (
        test_results.groupby(["store_name", "date"])
        .agg({"sales": "sum", "prediction": "sum"})
        .reset_index()
    )

    # Plot for each store
    stores = store_results["store_name"].unique()
    for store in stores:
        store_data = store_results[store_results["store_name"] == store]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            store_data["date"],
            store_data["sales"],
            "o-",
            label="Actual",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(
            store_data["date"],
            store_data["prediction"],
            "s--",
            label="Predicted",
            alpha=0.7,
            linewidth=2,
        )

        # Add title and labels
        ax.set_title(f"Store: {store} - Total Daily Sales")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")

        # Format x-axis dates
        ax.xaxis.set_major_formatter(date_formatter)
        plt.xticks(rotation=45)

        # Add grid and legend
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        # Adjust layout and save
        plt.tight_layout()
        safe_store = store.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"store_{safe_store}_total.png"))
        plt.close(fig)

    # 3. Aggregate by item
    item_results = (
        test_results.groupby(["item_name", "date"])
        .agg({"sales": "sum", "prediction": "sum"})
        .reset_index()
    )

    # Plot for each item
    items = item_results["item_name"].unique()
    for item in items:
        item_data = item_results[item_results["item_name"] == item]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            item_data["date"],
            item_data["sales"],
            "o-",
            label="Actual",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(
            item_data["date"],
            item_data["prediction"],
            "s--",
            label="Predicted",
            alpha=0.7,
            linewidth=2,
        )

        # Add title and labels
        ax.set_title(f"Item: {item} - Total Daily Sales")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")

        # Format x-axis dates
        ax.xaxis.set_major_formatter(date_formatter)
        plt.xticks(rotation=45)

        # Add grid and legend
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        # Adjust layout and save
        plt.tight_layout()
        safe_item = item.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"item_{safe_item}_total.png"))
        plt.close(fig)

    print(f"Aggregated visualizations saved to {output_dir}/")


def create_interactive_dashboard(test_results, output_dir="visualizations"):
    """
    Create an interactive HTML dashboard with plots for all store-item combinations.
    Requires Plotly and Dash libraries.

    Args:
        test_results: DataFrame containing test results
        output_dir: Directory to save the dashboard
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        print("Creating interactive dashboard...")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Ensure date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(test_results["date"]):
            test_results["date"] = pd.to_datetime(test_results["date"])

        # Create overall performance figure
        daily_results = (
            test_results.groupby("date")
            .agg({"sales": "sum", "prediction": "sum"})
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_results["date"],
                y=daily_results["sales"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily_results["date"],
                y=daily_results["prediction"],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            title="Overall Sales Performance: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Total Sales",
            legend_title="Series",
            height=600,
        )

        # Save the overall chart as HTML
        fig.write_html(os.path.join(output_dir, "overall_performance.html"))

        # Create an error heatmap
        store_item_error = (
            test_results.groupby(["store_name", "item_name"])
            .apply(
                lambda x: np.mean(np.abs((x["sales"] - x["prediction"]) / x["sales"]))
                * 100
            )
            .reset_index()
        )
        store_item_error.columns = ["store_name", "item_name", "mape"]

        # Pivot the data for the heatmap
        heatmap_data = store_item_error.pivot(
            index="store_name", columns="item_name", values="mape"
        )

        # Create heatmap figure
        heatmap_fig = px.imshow(
            heatmap_data,
            labels=dict(x="Item", y="Store", color="MAPE (%)"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="RdBu_r",
            title="Mean Absolute Percentage Error by Store and Item",
        )

        heatmap_fig.update_layout(height=800, width=1200)

        # Save the heatmap as HTML
        heatmap_fig.write_html(os.path.join(output_dir, "error_heatmap.html"))

        print(f"Interactive dashboard elements saved to {output_dir}/")

    except ImportError:
        print("Could not create interactive dashboard. Plotly library is required.")
        print("Install it with: pip install plotly dash")


def visualize_error_distribution(test_results, output_dir="visualizations"):
    """
    Visualize the distribution and patterns of prediction errors.

    Args:
        test_results: DataFrame containing test results
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate errors
    test_results["error"] = test_results["sales"] - test_results["prediction"]
    test_results["abs_error"] = np.abs(test_results["error"])
    test_results["pct_error"] = (test_results["error"] / test_results["sales"]) * 100

    # 1. Error distribution histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(test_results["error"], kde=True, bins=50)
    plt.axvline(x=0, color="red", linestyle="--")
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()

    # 2. Error vs Actual Sales
    plt.figure(figsize=(12, 6))
    plt.scatter(test_results["sales"], test_results["error"], alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("Prediction Error vs Actual Sales")
    plt.xlabel("Actual Sales")
    plt.ylabel("Error (Actual - Predicted)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_sales.png"))
    plt.close()

    # 3. Error over time
    plt.figure(figsize=(14, 6))
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(test_results["date"]):
        test_results["date"] = pd.to_datetime(test_results["date"])

    # Group by date to see overall error trend
    daily_error = test_results.groupby("date")["error"].mean().reset_index()
    plt.plot(daily_error["date"], daily_error["error"], "o-")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("Mean Prediction Error Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mean Error")
    date_formatter = DateFormatter("%Y-%m-%d")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_over_time.png"))
    plt.close()

    # 4. Error by day of week
    test_results["day_of_week"] = test_results["date"].dt.dayofweek
    test_results["day_name"] = test_results["date"].dt.day_name()

    plt.figure(figsize=(12, 6))
    day_error = (
        test_results.groupby("day_name")["pct_error"]
        .mean()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
    )
    sns.barplot(x=day_error.index, y=day_error.values)
    plt.title("Mean Percentage Error by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Mean Percentage Error (%)")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_day_of_week.png"))
    plt.close()

    # 5. Error by category - only if 'category' column exists
    if "category" in test_results.columns:
        plt.figure(figsize=(12, 6))
        cat_error = test_results.groupby("category")["pct_error"].mean().sort_values()
        sns.barplot(x=cat_error.index, y=cat_error.values)
        plt.title("Mean Percentage Error by Category")
        plt.xlabel("Category")
        plt.ylabel("Mean Percentage Error (%)")
        plt.axhline(y=0, color="red", linestyle="--")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_by_category.png"))
        plt.close()

    print(f"Error analysis visualizations saved to {output_dir}/")


def create_forecast_dashboard(
    model, X_test, y_test, test_results, data, output_dir="visualizations"
):
    """
    Create a comprehensive dashboard of forecast visualizations.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        test_results: DataFrame with test results
        data: Original data with date, store, item info
        output_dir: Directory to save visualizations
    """
    # Create all visualizations
    print("Creating forecast visualizations...")

    # 1. Individual store-item visualizations (limited to avoid too many plots)
    # Get the top 20 store-item combinations by sales volume
    store_item_sales = (
        test_results.groupby(["store_name", "item_name"])["sales"].sum().reset_index()
    )
    top_combinations = store_item_sales.sort_values("sales", ascending=False).head(20)

    # Filter test_results to include only these top combinations
    top_results = pd.merge(
        test_results,
        top_combinations[["store_name", "item_name"]],
        on=["store_name", "item_name"],
    )

    # Create visualizations for top combinations
    visualize_predictions_by_store_item(top_results, output_dir)

    # 2. Aggregated visualizations
    visualize_aggregated_predictions(test_results, output_dir)

    # 3. Error distribution and patterns
    visualize_error_distribution(test_results, output_dir)

    # 4. Try to create interactive dashboard if plotly is available
    create_interactive_dashboard(test_results, output_dir)

    print("Forecast visualization dashboard created successfully!")
