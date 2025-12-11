import pandas as pd
import streamlit as st

from src.ui_builder.data_viz import (
    plot_category_distribution,
    plot_day_of_week_pattern,
    plot_sales_distribution,
    plot_sales_time_series,
    plot_store_comparison,
)


def historical_sales_view(data):
    """Display the historical sales analysis dashboard"""

    st.title("Store Sales Dashboard")

    if data.empty:
        st.warning("No sales data available. Please check the data file.")
        return

    # Dashboard Filters Section
    filtered_data = configure_filters(data)

    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return

    # Display the KPIs section
    display_kpis(filtered_data)

    # Display the sales trends section
    display_sales_trends(filtered_data)

    # Display the performance breakdown section
    display_performance_breakdown(filtered_data)

    # Display the sales distribution section
    st.header("Sales Distribution")
    fig = plot_sales_distribution(filtered_data)
    st.pyplot(fig)

    # Data Table (Expandable)
    with st.expander("View Detailed Sales Data"):
        st.dataframe(
            filtered_data.sort_values("date", ascending=False), use_container_width=True
        )


def configure_filters(data):
    """Configure and apply dashboard filters"""

    with st.sidebar:
        st.header("Dashboard Filters")

        # Date range selector
        st.subheader("Date Range")
        min_date = data["date"].min().date()
        max_date = data["date"].max().date()

        start_date = st.date_input(
            "From", min_date, min_value=min_date, max_value=max_date
        )
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)

        # Store selector dropdown
        st.subheader("Store Selection")
        if "store_name" in data.columns:
            store_names = sorted(data["store_name"].unique())
            selected_store_name = st.selectbox(
                "Select Store", options=["All Stores"] + list(store_names)
            )
            selected_store = "All Stores"
        elif "store" in data.columns:
            stores = sorted(data["store"].unique())
            selected_store = st.selectbox(
                "Select Store", options=["All Stores"] + list(stores)
            )
            selected_store_name = "All Stores"
        else:
            selected_store = "All Stores"
            selected_store_name = "All Stores"

        # Category filter if category column exists
        if "category" in data.columns:
            st.subheader("Product Categories")
            categories = sorted(data["category"].unique())
            selected_categories = st.multiselect(
                "Select Categories", categories, default=categories
            )
        else:
            selected_categories = None

    # Filter data based on selection
    filtered_data = data.copy()
    mask = (filtered_data["date"].dt.date >= start_date) & (
        filtered_data["date"].dt.date <= end_date
    )

    # Apply store filter
    if "store_name" in data.columns and selected_store_name != "All Stores":
        mask &= filtered_data["store_name"] == selected_store_name
    elif "store" in data.columns and selected_store != "All Stores":
        mask &= filtered_data["store"] == selected_store

    # Apply category filter
    if selected_categories:
        mask &= filtered_data["category"].isin(selected_categories)

    # Store filter selections in session state for other functions to access
    st.session_state.selected_store = selected_store
    st.session_state.selected_store_name = selected_store_name
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    return filtered_data[mask]


def display_kpis(filtered_data):
    """Display KPI metrics in the dashboard"""

    st.header("Key Performance Indicators")

    # Calculate KPIs
    total_sales = filtered_data["sales"].sum()
    avg_daily_sales = filtered_data.groupby("date")["sales"].sum().mean()

    # Calculate period comparison if enough data
    if len(filtered_data["date"].unique()) >= 2:
        # Split the date range in half for comparison
        mid_date = (
            st.session_state.start_date
            + (st.session_state.end_date - st.session_state.start_date) / 2
        )

        period1_data = filtered_data[filtered_data["date"].dt.date <= mid_date]
        period2_data = filtered_data[filtered_data["date"].dt.date > mid_date]

        period1_sales = period1_data["sales"].sum() if not period1_data.empty else 0
        period2_sales = period2_data["sales"].sum() if not period2_data.empty else 0

        sales_change_pct = (
            ((period2_sales - period1_sales) / period1_sales * 100)
            if period1_sales > 0
            else 0
        )
    else:
        sales_change_pct = 0

    # Transaction count if available
    if "transactions" in filtered_data.columns:
        total_transactions = filtered_data["transactions"].sum()
        avg_transaction_value = (
            total_sales / total_transactions if total_transactions > 0 else 0
        )
    else:
        total_transactions = filtered_data.shape[0]  # Use row count as proxy
        avg_transaction_value = (
            total_sales / total_transactions if total_transactions > 0 else 0
        )

    # Display KPIs in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Sales",
            value=f"${total_sales:,.2f}",
            delta=f"{sales_change_pct:.1f}%" if sales_change_pct != 0 else None,
        )

    with col2:
        st.metric(label="Avg Daily Sales", value=f"${avg_daily_sales:,.2f}")

    with col3:
        st.metric(label="Total Transactions", value=f"{total_transactions:,}")

    with col4:
        st.metric(label="Avg Transaction Value", value=f"${avg_transaction_value:,.2f}")


def display_sales_trends(filtered_data):
    """Display sales trends section with time series and day of week patterns"""

    st.header("Sales Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Time series plot of sales
        fig = plot_sales_time_series(
            filtered_data,
            st.session_state.selected_store,
            st.session_state.selected_store_name,
        )
        st.pyplot(fig)

    with col2:
        # Weekly patterns
        if len(filtered_data["date"].unique()) >= 7:
            fig = plot_day_of_week_pattern(filtered_data)
            st.pyplot(fig)


def display_performance_breakdown(filtered_data):
    """Display performance breakdown section with category and store comparisons"""

    st.header("Performance Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        # Category performance if available
        if (
            "category" in filtered_data.columns
            and len(filtered_data["category"].unique()) > 1
        ):
            st.subheader("Category Performance")

            # Group by category
            category_sales = (
                filtered_data.groupby("category")["sales"]
                .sum()
                .sort_values(ascending=False)
            )

            # Calculate percentage of total
            category_sales_pct = (category_sales / category_sales.sum() * 100).round(1)

            # Create DataFrame for display
            category_df = pd.DataFrame(
                {"Sales": category_sales, "Percentage": category_sales_pct}
            ).reset_index()

            # Format for display
            category_df["Sales"] = category_df["Sales"].apply(lambda x: f"${x:,.2f}")
            category_df["Percentage"] = category_df["Percentage"].apply(
                lambda x: f"{x}%"
            )

            st.dataframe(category_df, use_container_width=True)

            # Create pie chart
            fig = plot_category_distribution(filtered_data)
            st.pyplot(fig)

    with col2:
        # Store comparison if all stores selected
        if (
            st.session_state.selected_store_name == "All Stores"
            and st.session_state.selected_store == "All Stores"
        ) and (
            "store_name" in filtered_data.columns or "store" in filtered_data.columns
        ):
            st.subheader("Store Comparison")

            # Determine store identifier
            if "store_name" in filtered_data.columns:
                store_identifier = "store_name"
            else:
                store_identifier = "store"

            # Group by store and create DataFrame
            store_sales = (
                filtered_data.groupby(store_identifier)["sales"]
                .sum()
                .sort_values(ascending=False)
            )

            # Take top 10 stores
            top_stores = store_sales.head(10)

            # Create DataFrame for display
            store_df = pd.DataFrame(
                {
                    "Store": top_stores.index,
                    "Sales": top_stores.values,
                }
            )

            # Format for display
            store_df["Sales"] = store_df["Sales"].apply(lambda x: f"${x:,.2f}")

            st.dataframe(store_df, use_container_width=True)

            # Create bar chart
            fig = plot_store_comparison(filtered_data, store_identifier)
            st.pyplot(fig)
