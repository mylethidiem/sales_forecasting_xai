import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime

from src.ui_builder.data_viz import (
    plot_category_distribution,
    plot_day_of_week_pattern,
    plot_sales_distribution,
    plot_sales_time_series,
    plot_store_comparison,
)

# Mocking st.session_state for Gradio logic compatibility
class SessionState(dict):
    def __getattr__(self, item): return self.get(item)
    def __setattr__(self, key, value): self[key] = value

session_state = SessionState()

def configure_filters(data, start_date, end_date, selected_store_input, selected_categories):
    """Logic-only version of configure_filters (removing st.sidebar calls)"""

    # Resolve store selection logic from the original code
    selected_store = "All Stores"
    selected_store_name = "All Stores"

    if "store_name" in data.columns:
        selected_store_name = selected_store_input
    elif "store" in data.columns:
        selected_store = selected_store_input

    # Filter data based on selection
    filtered_data = data.copy()

    # Gradio strings to datetime.date
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    mask = (filtered_data["date"].dt.date >= start_dt) & (
        filtered_data["date"].dt.date <= end_dt
    )

    # Apply store filter
    if "store_name" in data.columns and selected_store_name != "All Stores":
        mask &= filtered_data["store_name"] == selected_store_name
    elif "store" in data.columns and selected_store != "All Stores":
        mask &= filtered_data["store"] == selected_store

    # Apply category filter
    if selected_categories:
        mask &= filtered_data["category"].isin(selected_categories)

    # Update state for other functions
    session_state.selected_store = selected_store
    session_state.selected_store_name = selected_store_name
    session_state.start_date = start_dt
    session_state.end_date = end_dt

    return filtered_data[mask]

def display_kpis(filtered_data):
    """Logic-only version of display_kpis (returns strings for UI)"""
    total_sales = filtered_data["sales"].sum()
    avg_daily_sales = filtered_data.groupby("date")["sales"].sum().mean()

    if len(filtered_data["date"].unique()) >= 2:
        mid_date = session_state.start_date + (session_state.end_date - session_state.start_date) / 2
        period1_data = filtered_data[filtered_data["date"].dt.date <= mid_date]
        period2_data = filtered_data[filtered_data["date"].dt.date > mid_date]
        period1_sales = period1_data["sales"].sum() if not period1_data.empty else 0
        period2_sales = period2_data["sales"].sum() if not period2_data.empty else 0
        sales_change_pct = (((period2_sales - period1_sales) / period1_sales * 100) if period1_sales > 0 else 0)
    else:
        sales_change_pct = 0

    if "transactions" in filtered_data.columns:
        total_transactions = filtered_data["transactions"].sum()
    else:
        total_transactions = filtered_data.shape[0]

    avg_transaction_value = (total_sales / total_transactions if total_transactions > 0 else 0)

    # Return formatted strings for Gradio Label/Textbox components
    return (
        total_sales,
        sales_change_pct,
        avg_daily_sales,
        total_transactions,
        avg_transaction_value
    )

def display_sales_trends(filtered_data):
    """Logic-only version of display_sales_trends (returns figures)"""
    fig1 = plot_sales_time_series(
        filtered_data,
        session_state.selected_store,
        session_state.selected_store_name,
    )

    fig2 = None
    if len(filtered_data["date"].unique()) >= 7:
        fig2 = plot_day_of_week_pattern(filtered_data)

    return fig1, fig2

def display_performance_breakdown(filtered_data):
    """Logic-only version of display_performance_breakdown (returns DF and Fig)"""
    category_df = pd.DataFrame()
    fig_cat = None
    store_df = pd.DataFrame()
    fig_store = None

    if "category" in filtered_data.columns and len(filtered_data["category"].unique()) > 1:
        category_sales = filtered_data.groupby("category")["sales"].sum().sort_values(ascending=False)
        category_sales_pct = (category_sales / category_sales.sum() * 100).round(1)
        category_df = pd.DataFrame({"Sales": category_sales, "Percentage": category_sales_pct}).reset_index()
        category_df["Sales"] = category_df["Sales"].apply(lambda x: f"${x:,.2f}")
        category_df["Percentage"] = category_df["Percentage"].apply(lambda x: f"{x}%")
        fig_cat = plot_category_distribution(filtered_data)

    if (session_state.selected_store_name == "All Stores" and session_state.selected_store == "All Stores") and \
       ("store_name" in filtered_data.columns or "store" in filtered_data.columns):
        store_identifier = "store_name" if "store_name" in filtered_data.columns else "store"
        store_sales = filtered_data.groupby(store_identifier)["sales"].sum().sort_values(ascending=False)
        top_stores = store_sales.head(10)
        store_df = pd.DataFrame({"Store": top_stores.index, "Sales": top_stores.values})
        store_df["Sales"] = store_df["Sales"].apply(lambda x: f"${x:,.2f}")
        fig_store = plot_store_comparison(filtered_data, store_identifier)

    return category_df, fig_cat, store_df, fig_store

def format_kpi_html(label, value_str, delta_pct=None):
    """Create HTML for metric"""

    # Process Delta
    delta_html = ""
    if delta_pct is not None and delta_pct != 0:
        if delta_pct > 0:
            color = "color: #38a169;"  # Greenn
            arrow = "▲"
        else:
            color = "color: #e53e3e;"  # Red
            arrow = "▼"

        # Format delta: Ví dụ: "▲ 4.2%"
        delta_str = f"{arrow} {abs(delta_pct):.1f}%"
        delta_html = f'<div style="{color} font-size: 14px; font-weight: 500; margin-top: 5px; line-height: 1;">{delta_str}</div>'

    html_output = f"""
    <div style="font-family: Arial, sans-serif; padding: 10px;">
        <div style="font-size: 14px; color: #555; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 30px; font-weight: 600; color: #1a1a1a; line-height: 1;">{value_str}</div>
        {delta_html}
    </div>
    """
    return html_output

def update_kpis_html(total_sales, sales_change_pct, avg_daily_sales, total_transactions, avg_transaction_value):
    """wrapper function update KPI HTML"""

    html1 = format_kpi_html(
        "💰 Total Sales",
        f"${total_sales:,.2f}",
        sales_change_pct
    )

    html2 = format_kpi_html(
        "📊 Avg Daily Sales",
        f"${avg_daily_sales:,.2f}"
    )

    html3 = format_kpi_html(
        "🛒 Total Transactions",
        f"{total_transactions:,}"
    )

    html4 = format_kpi_html(
        "💵 Avg Transaction Value",
        f"${avg_transaction_value:,.2f}"
    )

    return html1, html2, html3, html4

def historical_sales_view(data):
    """Main Gradio Interface Builder"""

    def run_dashboard_update(start_date, end_date, store_selection, categories):
        # 1. Logic: Filter
        filtered_data = configure_filters(data, start_date, end_date, store_selection, categories)

        if filtered_data.empty:
            empty_msg = "⚠️ No data available for the selected filters. Please adjust your selections."
            return [empty_msg] * 4 + [None] * 5 + [pd.DataFrame()]

        # 2. Logic: KPIs
        kpi_metrics = display_kpis(filtered_data)
        html1, html2, html3, html4 = update_kpis_html(
            *kpi_metrics
        )

        # 3. Logic: Trends
        fig_ts, fig_dow = display_sales_trends(filtered_data)

        # 4. Logic: Breakdown
        cat_df, fig_cat, store_df, fig_store = display_performance_breakdown(filtered_data)

        # 5. Logic: Distribution
        fig_dist = plot_sales_distribution(filtered_data)

        # 6. Logic: Table
        detailed_table = filtered_data.sort_values("date", ascending=False)

        return (
            html1, html2, html3, html4,
            fig_ts, fig_dow,
            cat_df, fig_cat,
            store_df, fig_store,
            fig_dist,
            detailed_table
        )

    # Define the App Layout (Compatible with older Gradio versions)
    with gr.Blocks(title="Store Sales Dashboard") as demo:
        # Header
        gr.Markdown(
            """
            # 📊 Store Sales Dashboard
            ### Comprehensive sales analytics and performance insights
            """
        )

        # Left Sidebar - Filters (Fixed)
        with gr.Sidebar(position="right"):
                gr.Markdown("## 🔍 Dashboard Filters")
                gr.Markdown("---")

                # Date Filters
                gr.Markdown("### 📅 Date Range")
                min_date = data["date"].min().date()
                max_date = data["date"].max().date()

                start_in = gr.DateTime(
                    label="From",
                    value=str(min_date),
                    type="string",
                    interactive=True
                )
                end_in = gr.DateTime(
                    label="To",
                    value=str(max_date),
                    type="string",
                    interactive=True
                )

                gr.Markdown("---")

                # Store Filter
                gr.Markdown("### 🏬 Store Selection")
                if "store_name" in data.columns:
                    opts = ["All Stores"] + sorted(data["store_name"].unique().tolist())
                elif "store" in data.columns:
                    opts = ["All Stores"] + sorted(data["store"].unique().tolist())
                else:
                    opts = ["All Stores"]

                store_in = gr.Dropdown(
                    choices=opts,
                    value="All Stores",
                    label="Select Store",
                    interactive=True
                )

                # Category Filter
                cat_in = None
                if "category" in data.columns:
                    gr.Markdown("---")
                    gr.Markdown("### 📦 Product Categories")
                    cats = sorted(data["category"].unique().tolist())
                    cat_in = gr.CheckboxGroup(
                        choices=cats,
                        value=cats,
                        label="Select Categories",
                        interactive=True
                    )

                gr.Markdown("---")
                btn = gr.Button("🔄 Update Dashboard", variant="primary", size="lg")

                gr.Markdown(
                    """
                    <br>
                    💡 **Tip:** Adjust filters and click Update to refresh
                    """
                )

        # Right Column - Main Dashboard
        # with gr.Column():
        # KPI Section
        gr.Markdown("## 📈 Key Performance Indicators")
        with gr.Row():
            m1 = gr.HTML(label=None, scale=1, container=True)
            m2 = gr.HTML(label=None, scale=1, container=True)
            m3 = gr.HTML(label=None, scale=1, container=True)
            m4 = gr.HTML(label=None, scale=1, container=True)

        gr.Markdown("---")

        # Sales Trends Section
        gr.Markdown("## 📉 Sales Trends Analysis")
        with gr.Row():
            p_ts = gr.Plot(label="📈 Sales Time Series", container=True, scale=1)
            p_dow = gr.Plot(label="📅 Weekly Patterns", container=True, scale=1)

        gr.Markdown("---")

        # Performance Breakdown Section
        gr.Markdown("## 🎯 Performance Breakdown")

        # Category Performance Section
        gr.Markdown("### 📦 Category Performance")
        with gr.Row():
            with gr.Column(scale=1):
                df_cat = gr.DataFrame(label="Category Sales Data", max_height=300)
            with gr.Column(scale=1):
                p_cat = gr.Plot(label="Sales by Category", container=True)

        gr.Markdown("---")

        # Store Comparison Section
        gr.Markdown("### 🏪 Store Comparison (Top 10)")
        with gr.Row():
            with gr.Column(scale=1):
                df_store = gr.DataFrame(label="Top Performing Stores", max_height=300)
            with gr.Column(scale=2):
                p_store = gr.Plot(label="Top 10 Stores by Sales", container=True)

        gr.Markdown("---")

        # Sales Distribution Section
        gr.Markdown("## 📊 Sales Distribution")
        p_dist = gr.Plot(label="Distribution Analysis", container=True)

        gr.Markdown("---")

        # Detailed Data Section
        with gr.Accordion("📋 View Detailed Sales Data", open=True):
            gr.Markdown("*Complete transaction history for the selected period*")
            df_detailed = gr.DataFrame(max_height=400)

        # Footer
        gr.Markdown(
            """
            ---
            <div style='text-align: center; color: #666; font-size: 0.9em;'>
            📊 Store Sales Dashboard | Powered by Gradio
            </div>
            """
        )

        # Link event - Update button
        btn.click(
            run_dashboard_update,
            inputs=[start_in, end_in, store_in, cat_in],
            outputs=[m1, m2, m3, m4, p_ts, p_dow, df_cat, p_cat, df_store, p_store, p_dist, df_detailed]
        )

        # Auto-load initial data on page load
        demo.load(
            run_dashboard_update,
            inputs=[start_in, end_in, store_in, cat_in],
            outputs=[m1, m2, m3, m4, p_ts, p_dow, df_cat, p_cat, df_store, p_store, p_dist, df_detailed]
        )

    return demo

# Usage:
# if __name__ == "__main__":
#     df = pd.read_csv("your_data.csv", parse_dates=['date'])
#     app = historical_sales_view(df)
#     app.launch()