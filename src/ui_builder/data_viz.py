import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sales_forecast(
    historical_data, prediction_date, prediction_value, store_id=None
):
    """
    Plot historical sales with prediction point
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter for specific store if provided
    if store_id is not None and "store" in historical_data.columns:
        plot_data = historical_data[historical_data["store"] == store_id].copy()
    else:
        plot_data = historical_data.copy()

    # Group by date if multiple records per date
    if len(plot_data) > len(plot_data["date"].unique()):
        plot_data = plot_data.groupby("date")["sales"].sum().reset_index()

    # Sort by date
    plot_data = plot_data.sort_values("date")

    # Plot historical data
    ax.plot(plot_data["date"], plot_data["sales"], label="Historical Sales")

    # Add prediction point
    ax.scatter(
        prediction_date, prediction_value, color="red", s=100, label="Prediction"
    )

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    if store_id is not None:
        ax.set_title(f"Sales Forecast for Store {store_id}")
    else:
        ax.set_title("Sales Forecast")
    ax.legend()
    fig.autofmt_xdate()

    return fig


def plot_sales_time_series(
    filtered_data, selected_store=None, selected_store_name=None
):
    """Generate time series plot of sales with moving average"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot data based on store selection
    if selected_store_name == "All Stores" and selected_store == "All Stores":
        # Group by date for the trend line
        sales_by_date = filtered_data.groupby("date")["sales"].sum()
        ax.plot(sales_by_date.index, sales_by_date.values, "b-")

        # Add moving average
        if len(sales_by_date) > 7:
            sales_by_date_df = sales_by_date.reset_index()
            sales_by_date_df["MA7"] = sales_by_date_df["sales"].rolling(window=7).mean()
            ax.plot(
                sales_by_date_df["date"],
                sales_by_date_df["MA7"],
                "r--",
                label="7-Day Moving Avg",
            )
            ax.legend()
    else:
        # Single store - show daily sales and trend
        sales_by_date = filtered_data.groupby("date")["sales"].sum()
        ax.plot(sales_by_date.index, sales_by_date.values, "b-")

        # Add moving average if enough data
        if len(sales_by_date) > 7:
            sales_by_date_df = sales_by_date.reset_index()
            sales_by_date_df["MA7"] = sales_by_date_df["sales"].rolling(window=7).mean()
            ax.plot(
                sales_by_date_df["date"],
                sales_by_date_df["MA7"],
                "r--",
                label="7-Day Moving Avg",
            )
            ax.legend()

    ax.set_xlabel("")
    ax.set_ylabel("Sales ($)")

    if "store_name" in filtered_data.columns and selected_store_name != "All Stores":
        ax.set_title(f"Daily Sales - {selected_store_name}")
    elif "store" in filtered_data.columns and selected_store != "All Stores":
        ax.set_title(f"Daily Sales - Store {selected_store}")
    else:
        ax.set_title("Daily Sales - All Stores")

    fig.autofmt_xdate()
    return fig


def plot_day_of_week_pattern(filtered_data):
    """Generate bar chart showing sales by day of week"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Add day of week name
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    filtered_data["day_name"] = filtered_data["date"].dt.dayofweek.apply(
        lambda x: day_names[x]
    )

    # Group by day of week
    day_sales = filtered_data.groupby("day_name")["sales"].mean().reindex(day_names)

    # Calculate average line
    avg_daily = day_sales.mean()

    # Create bar chart with average line
    bars = ax.bar(day_sales.index, day_sales.values, color="skyblue")
    ax.axhline(y=avg_daily, color="red", linestyle="--", label="Daily Average")

    # Highlight best and worst days
    best_day = day_sales.idxmax()
    worst_day = day_sales.idxmin()

    for i, (day, sales) in enumerate(day_sales.items()):
        if day == best_day:
            bars[i].set_color("green")
        elif day == worst_day:
            bars[i].set_color("orange")

    ax.set_xlabel("")
    ax.set_ylabel("Average Sales ($)")
    ax.set_title("Sales by Day of Week")
    plt.xticks(rotation=45)
    ax.legend()

    return fig


def plot_category_distribution(filtered_data):
    """Generate pie chart of sales by category"""
    fig, ax = plt.subplots(figsize=(6, 6))

    category_sales = (
        filtered_data.groupby("category")["sales"].sum().sort_values(ascending=False)
    )

    top_categories = category_sales.head(5)
    others = category_sales.iloc[5:].sum() if len(category_sales) > 5 else 0

    if others > 0:
        plot_data = pd.concat([top_categories, pd.Series([others], index=["Others"])])
    else:
        plot_data = top_categories

    plt.pie(
        plot_data,
        labels=plot_data.index,
        autopct="%1.1f%%",
        startangle=90,
        shadow=False,
    )
    plt.axis("equal")
    plt.title("Sales by Category")

    return fig


def plot_store_comparison(filtered_data, store_identifier="store"):
    """Generate horizontal bar chart for top stores by sales"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Group by store
    store_sales = (
        filtered_data.groupby(store_identifier)["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    # Take top 10 stores
    top_stores = store_sales.head(10)

    # Plot horizontal bar chart
    y_pos = np.arange(len(top_stores))
    ax.barh(y_pos, top_stores.values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_stores.index)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Sales ($)")
    ax.set_title("Top 10 Stores by Sales")

    return fig


def plot_sales_distribution(filtered_data):
    """Generate histogram with KDE and summary statistics"""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create histogram with KDE
    sns.histplot(filtered_data["sales"], bins=30, kde=True, ax=ax)

    # Add vertical lines for key statistics
    median_sales = filtered_data["sales"].median()
    mean_sales = filtered_data["sales"].mean()

    ax.axvline(
        x=median_sales, color="r", linestyle="--", label=f"Median: ${median_sales:.2f}"
    )
    ax.axvline(
        x=mean_sales, color="g", linestyle="--", label=f"Mean: ${mean_sales:.2f}"
    )

    ax.set_xlabel("Sales ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sales Distribution")
    ax.legend()

    return fig
