import pandas as pd
import gradio as gr
import datetime

# Strict import as requested
from src.ui_builder.data_viz import (
    plot_category_distribution,
    plot_day_of_week_pattern,
    plot_sales_distribution,
    plot_sales_time_series,
    plot_store_comparison,
)

def filter_data(data, start_date_str, end_date_str, selected_store_val, selected_categories):
    """Helper to filter data based on inputs"""
    df = data.copy()

    # Convert string dates from Gradio to datetime.date
    # Gradio DateTime/Textbox usually returns string YYYY-MM-DD or full timestamp
    if isinstance(start_date_str, str):
        start_date = pd.to_datetime(start_date_str).date()
    else:
        start_date = start_date_str.date() if hasattr(start_date_str, 'date') else start_date_str

    if isinstance(end_date_str, str):
        end_date = pd.to_datetime(end_date_str).date()
    else:
        end_date = end_date_str.date() if hasattr(end_date_str, 'date') else end_date_str

    # Date Filter
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)

    # Store Filter logic
    selected_store_name = "All Stores"
    selected_store_id = "All Stores"

    if selected_store_val != "All Stores":
        if "store_name" in df.columns:
            mask &= df["store_name"] == selected_store_val
            selected_store_name = selected_store_val
        elif "store" in df.columns:
            mask &= df["store"].astype(str) == str(selected_store_val) # Ensure type match
            selected_store_id = selected_store_val

    # Category Filter
    if selected_categories and "category" in df.columns:
        mask &= df["category"].isin(selected_categories)

    return df[mask], selected_store_id, selected_store_name

def generate_kpi_html(filtered_data, original_data, start_date, end_date):
    """Generates HTML string for KPIs to mimic st.metric"""
    total_sales = filtered_data["sales"].sum()
    avg_daily_sales = filtered_data.groupby("date")["sales"].sum().mean()
    if pd.isna(avg_daily_sales): avg_daily_sales = 0

    # Calculate transactions
    if "transactions" in filtered_data.columns:
        total_transactions = filtered_data["transactions"].sum()
    else:
        total_transactions = filtered_data.shape[0]

    avg_txn_value = (total_sales / total_transactions) if total_transactions > 0 else 0

    # Calculate Delta (Approximate logic from original)
    sales_change_pct = 0
    unique_dates = filtered_data["date"].unique()

    # Use the session dates passed in for mid-point calculation
    s_date = pd.to_datetime(start_date).date() if isinstance(start_date, str) else start_date
    e_date = pd.to_datetime(end_date).date() if isinstance(end_date, str) else end_date

    if len(unique_dates) >= 2:
        mid_date = s_date + (e_date - s_date) / 2
        period1 = filtered_data[filtered_data["date"].dt.date <= mid_date]["sales"].sum()
        period2 = filtered_data[filtered_data["date"].dt.date > mid_date]["sales"].sum()

        if period1 > 0:
            sales_change_pct = ((period2 - period1) / period1 * 100)

    # HTML styling for KPIs
    delta_color = "green" if sales_change_pct >= 0 else "red"
    delta_arrow = "↑" if sales_change_pct >= 0 else "↓"

    html = f"""
    <div style="display: flex; gap: 20px; text-align: center; justify-content: space-around; background: #f9fafb; padding: 20px; border-radius: 10px;">
        <div>
            <p style="color: gray; font-size: 0.9em; margin-bottom: 5px;">Total Sales</p>
            <h2 style="margin: 0;">${total_sales:,.2f}</h2>
            <small style="color: {delta_color}">{delta_arrow} {sales_change_pct:.1f}%</small>
        </div>
        <div>
            <p style="color: gray; font-size: 0.9em; margin-bottom: 5px;">Avg Daily Sales</p>
            <h2 style="margin: 0;">${avg_daily_sales:,.2f}</h2>
        </div>
        <div>
            <p style="color: gray; font-size: 0.9em; margin-bottom: 5px;">Total Transactions</p>
            <h2 style="margin: 0;">{total_transactions:,}</h2>
        </div>
        <div>
            <p style="color: gray; font-size: 0.9em; margin-bottom: 5px;">Avg Txn Value</p>
            <h2 style="margin: 0;">${avg_txn_value:,.2f}</h2>
        </div>
    </div>
    """
    return html

def prepare_category_table(filtered_data):
    if "category" not in filtered_data.columns or filtered_data.empty:
        return pd.DataFrame()

    cat_sales = filtered_data.groupby("category")["sales"].sum().sort_values(ascending=False)
    cat_pct = (cat_sales / cat_sales.sum() * 100).round(1)

    df = pd.DataFrame({"Sales": cat_sales, "Percentage": cat_pct}).reset_index()
    # Note: Gradio DataFrame handles formatting better if we keep them as numbers,
    # but to match st script exact formatting, we convert to string
    df["Sales"] = df["Sales"].apply(lambda x: f"${x:,.2f}")
    df["Percentage"] = df["Percentage"].apply(lambda x: f"{x}%")
    return df

def prepare_store_table(filtered_data, selected_store_name):
    if selected_store_name != "All Stores":
        return pd.DataFrame() # Don't show store comparison if specific store selected

    store_col = "store_name" if "store_name" in filtered_data.columns else "store"
    if store_col not in filtered_data.columns or filtered_data.empty:
        return pd.DataFrame()

    store_sales = filtered_data.groupby(store_col)["sales"].sum().sort_values(ascending=False).head(10)
    df = pd.DataFrame({"Store": store_sales.index, "Sales": store_sales.values})
    df["Sales"] = df["Sales"].apply(lambda x: f"${x:,.2f}")
    return df

def historical_sales_view(data):
    """
    Main entry point to launch the Gradio Dashboard.
    Pass your dataframe 'data' here.
    """

    if data.empty:
        print("No sales data available.")
        return

    # Pre-calculate filter options
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()

    store_options = ["All Stores"]
    if "store_name" in data.columns:
        store_options += sorted(list(data["store_name"].unique()))
    elif "store" in data.columns:
        store_options += sorted(list(data["store"].unique()))

    cat_options = []
    if "category" in data.columns:
        cat_options = sorted(list(data["category"].unique()))

    # --- THE UPDATE FUNCTION (The Engine) ---
    def update_dashboard(start_date, end_date, store_selection, cat_selection):
        # 1. Filter Data
        filtered, sel_store_id, sel_store_name = filter_data(
            data, start_date, end_date, store_selection, cat_selection
        )

        if filtered.empty:
            # Return empty states/placeholders
            return (
                "<h3>No data available for selected filters</h3>", # KPI
                None, None, # Trends
                pd.DataFrame(), None, # Category
                pd.DataFrame(), None, # Store
                None, # Distribution
                pd.DataFrame() # Raw Data
            )

        # 2. KPIs
        kpi_html = generate_kpi_html(filtered, data, start_date, end_date)

        # 3. Trends Plots
        # Note: We pass the store identifier logic to match original script expectations
        ts_fig = plot_sales_time_series(filtered, sel_store_id, sel_store_name)

        dow_fig = None
        if len(filtered["date"].unique()) >= 7:
            dow_fig = plot_day_of_week_pattern(filtered)

        # 4. Performance Breakdown
        # Category
        cat_df = prepare_category_table(filtered)
        cat_fig = plot_category_distribution(filtered)

        # Store Comparison (Only if All Stores selected)
        store_df = prepare_store_table(filtered, sel_store_name)
        store_fig = None
        if sel_store_name == "All Stores":
            store_col = "store_name" if "store_name" in data.columns else "store"
            store_fig = plot_store_comparison(filtered, store_col)

        # 5. Distribution
        dist_fig = plot_sales_distribution(filtered)

        # 6. Raw Data
        raw_df = filtered.sort_values("date", ascending=False)

        return (
            kpi_html,
            ts_fig, dow_fig,
            cat_df, cat_fig,
            store_df, store_fig,
            dist_fig,
            raw_df
        )

    # --- GRADIO UI LAYOUT ---
    with gr.Blocks(title="Store Sales Dashboard") as demo:
        gr.Markdown("# Store Sales Dashboard")

        with gr.Row():
            # Filters Sidebar (using Column to mimic sidebar)
            with gr.Column(scale=1, min_width=250, variant="panel"):
                gr.Markdown("### Filters")
                # Gradio doesn't have a specific DateRange picker, so we use two inputs
                date_start = gr.DateTime(label="From", value=min_date, type="datetime")
                date_end = gr.DateTime(label="To", value=max_date, type="datetime")

                dd_store = gr.Dropdown(choices=store_options, value="All Stores", label="Select Store")
                dd_cat = gr.Dropdown(choices=cat_options, value=cat_options, multiselect=True, label="Select Categories")

                btn_refresh = gr.Button("Update Dashboard", variant="primary")

            # Main Content Area
            with gr.Column(scale=4):
                # KPI Section
                out_kpi = gr.HTML()

                gr.Markdown("---")
                gr.Markdown("## Sales Trends")
                with gr.Row():
                    out_ts_plot = gr.Plot(label="Time Series")
                    out_dow_plot = gr.Plot(label="Weekly Pattern")

                gr.Markdown("## Performance Breakdown")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Category Performance")
                        out_cat_table = gr.DataFrame(interactive=False)
                        out_cat_plot = gr.Plot()
                    with gr.Column():
                        gr.Markdown("### Store Comparison")
                        out_store_table = gr.DataFrame(interactive=False)
                        out_store_plot = gr.Plot()

                gr.Markdown("## Sales Distribution")
                out_dist_plot = gr.Plot()

                with gr.Accordion("View Detailed Sales Data", open=False):
                    out_raw_data = gr.DataFrame()

    # --- EVENT WIRING ---
    # Trigger update on load and on button click
    inputs = [date_start, date_end, dd_store, dd_cat]
    outputs = [
        out_kpi,
        out_ts_plot, out_dow_plot,
        out_cat_table, out_cat_plot,
        out_store_table, out_store_plot,
        out_dist_plot,
        out_raw_data
    ]

    # Update when button clicked
    btn_refresh.click(fn=update_dashboard, inputs=inputs, outputs=outputs)

    # Optional: Update immediately when filters change (mimics Streamlit's auto-rerun)
    # If dataset is large, you might want to remove these lines and rely only on the button
    dd_store.change(fn=update_dashboard, inputs=inputs, outputs=outputs)
    dd_cat.change(fn=update_dashboard, inputs=inputs, outputs=outputs)

    # Initialize view on load
    demo.load(fn=update_dashboard, inputs=inputs, outputs=outputs)

    return demo

# Example usage:
# if __name__ == "__main__":
#     df = pd.read_csv("your_data.csv")
#     df['date'] = pd.to_datetime(df['date'])
#     demo = historical_sales_view(df)
#     demo.launch()