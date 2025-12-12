import gradio as gr
import ui_template as ui
# CAUTION: Ensure these loaders return standard data (DataFrames), not Streamlit widgets.
from src.data_loader.loader import (
    load_data,
    load_feature_engineered_data,
    load_feature_stats,
    load_model,
)

# CAUTION: These files must contain functions that generate Gradio components,
# NOT 'st.write' or 'st.sidebar' calls.
from src.ui_builder.dashboard import historical_sales_view
from src.ui_predictor.prediction import sales_prediction_view

def main():
    # 1. Load heavy data once (Global scope within main, acts like cache)
    data = load_data()
    model = load_model()
    feature_stats = load_feature_stats()

    # 2. Define the Gradio Interface
    with gr.Blocks(title="Sales Forecasting App") as demo:

        # --- Sidebar ---
        with gr.Sidebar():
            gr.Markdown("## Sales Forecasting App 📈")

            # Mimics st.sidebar.selectbox
            page_selector = gr.Dropdown(
                choices=["Historical Sales Analysis", "Sales Prediction"],
                value="Historical Sales Analysis",
                label="Choose a page",
                interactive=True
            )
        # --- Main Content Area ---
        # The @gr.render decorator mimics the 'if/else' flow of Streamlit.
        # It watches 'page_selector' and re-runs this function when it changes.
        @gr.render(inputs=page_selector)
        def render_content(page):
            if page == "Historical Sales Analysis":
                # This function must create Gradio components (e.g., gr.Plot, gr.Dataframe)
                historical_sales_view(data)

            else:
                # Lazy load this data only when this tab is selected (preserving your logic)
                print("Lazy load this data only when this tab is selected (preserving your logic)")
                feature_engineered_data = load_feature_engineered_data()

                # Pass data to the prediction view
                print("sales_prediction_view")
                sales_prediction_view(
                    data,
                    model,
                    feature_stats,
                    feature_engineered_data
                )

        ui.create_footer(
            logo_path="static/intelligent_retail.png",
            creator_name="Thi-Diem-My Le",
            creator_link="https://beacons.ai/elizabethmyn",
            org_name="AI VIET NAM",
            org_link="https://aivietnam.edu.vn/"
        )



    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch()