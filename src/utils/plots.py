import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sales(df, store_id=1, item_id=1):
    """Plot sales and visualize missing values"""

    df_2plot = df.query("(store_id==@store_id)&(item_id==@item_id)")
    store_name = df_2plot["store_name"].iloc[-1]
    item_name = df_2plot["item_name"].iloc[-1]

    fig, ax = plt.subplots(figsize=(6, 3))
    df_2plot[["date", "sales"]].plot(x="date", y="sales", ax=ax, legend=False)

    # Replace NaN values with the mean of surrounding two points
    nan_indices = df_2plot[df_2plot["sales"].isna()].index

    if len(nan_indices) >= 1:
        df_2plot = df_2plot.assign(sales=lambda df: df["sales"].fillna(method="ffill"))
        # Draw arrows for NaN values
        nan_dates = df_2plot.loc[nan_indices, "date"]
        nan_sales = df_2plot.loc[nan_indices, "sales"]
        for date, sales in zip(nan_dates, nan_sales):
            ax.annotate(
                "-",
                xy=(date, sales),
                color="red",  # Set text color to red
                size=20,
            )

    # Set plot labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title(f"Store: {store_name} - Item: {item_name}")
    ax.legend()
    plt.show()


def plot_forecast_single(flat_df, store_item):
    """
    Plot actual vs predicted sales for one store-item combo from flattened predictions for Prophet.
    """
    df = flat_df[flat_df["store_item"] == store_item].copy()

    if df.empty:
        print(f"No data found for: {store_item}")
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="ds", y="y", label="Actual", color="black")
    sns.lineplot(data=df, x="ds", y="yhat", label="Forecast", color="blue")
    plt.fill_between(
        df["ds"],
        df["yhat_lower"],
        df["yhat_upper"],
        color="blue",
        alpha=0.2,
        label="Confidence Interval",
    )
    plt.title(f"Forecast vs Actual for {store_item}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_sales_predictions(
    df_prediction, store_id=1, nrows=6, ncols=5, figsize=(20, 20)
):
    """
    Plots actual vs predicted sales for items in a given store.

    Parameters:
        df_prediction (DataFrame): Must include ['store_id', 'item_id', 'date', 'sales', 'prediction']
        store_id (int): Store to filter on
        nrows (int): Rows of subplots
        ncols (int): Columns of subplots
        figsize (tuple): Size of the full figure
    """
    df_sample = df_prediction[df_prediction["store_id"] == store_id]
    store_name = df_sample["store_name"].iloc[-1]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    item_ids = sorted(df_sample["item_id"].unique())

    for i, ax in enumerate(axes):
        if i >= len(item_ids):
            ax.axis("off")  # Hide unused subplots
            continue

        item_id = item_ids[i]
        df2plot = df_sample[df_sample["item_id"] == item_id]
        item_name = df2plot["item_name"].iloc[-1]

        if df2plot.empty:
            ax.axis("off")
            continue

        # Plot actual and predicted sales
        ax.plot(df2plot["date"], df2plot["sales"], label="Actual", color="blue")
        ax.plot(
            df2plot["date"],
            df2plot["prediction"],
            label="Forecast",
            color="red",
            linestyle="--",
            marker=".",
        )

        ax.set_title(f"Item: {item_name}")
        ax.set_xlabel("")
        ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)

    # Only add legend to the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the legend
    fig.suptitle(
        f"Sales Forecast vs Actual - Store {store_name}", fontsize=16, fontweight="bold"
    )
    plt.show()
