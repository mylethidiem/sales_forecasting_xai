import gradio as gr
from pathlib import Path

from src.data_loader.loader import (
    load_data,
    load_feature_engineered_data,
    load_feature_stats,
    load_model,
)
from src.ui_builder.dashboard import historical_sales_view
from src.ui_predictor.prediction import create_sales_prediction_interface
import ui_template as ui


def main():
    # Configure theme
    ui.configure(
        project_name="Sales Forecasting App",
        year="2025",
        about="Analyze historical sales and predict future trends",
        description="An interactive dashboard for sales analysis and forecasting using machine learning.",
        colors={
            "primary": "#0F6CBD",
            "accent": "#C4314B",
            "success": "#2E7D32",
            "bg1": "#F0F7FF",
            "bg2": "#E8F0FA",
            "bg3": "#DDE7F8"
        },
        meta_items=[
            ("Model", "Time Series Forecasting"),
            ("Features", "Historical Analysis & Predictions"),
        ]
    )

    # Load data and model
    data = load_data()
    model = load_model()
    feature_stats = load_feature_stats()
    feature_engineered_data = load_feature_engineered_data()

    # Create Gradio interface
    demo = gr.Blocks(title="Sales Forecasting App")

    with demo:
        # Header
        ui.create_header(logo_path="static/intelligent_retail2-removebg-preview.png")

        # Info card
        gr.HTML(ui.render_info_card(
            icon="📈",
            title="About this Application"
        ))

        # Disclaimer
        gr.HTML(ui.render_disclaimer(
            text="This application is for educational and analytical purposes only. "
                 "Predictions should be validated with domain expertise before business decisions.",
            icon="⚠️",
            title="Important Notice"
        ))

        # Main content with tabs
        with gr.Tabs():
            # Tab 1: Historical Sales Analysis
            with gr.Tab("📊 Historical Sales Analysis"):
                gr.Markdown("### Explore and visualize historical sales data")

                with gr.Row():
                    analyze_btn = gr.Button(
                        "Load Historical Analysis",
                        variant="primary",
                        size="lg"
                    )

                historical_output = gr.HTML(label="Analysis Results")

                def load_historical():
                    try:
                        return historical_sales_view(data)
                    except Exception as e:
                        return f"<div style='color: red; padding: 20px;'>Error loading analysis: {str(e)}</div>"

                analyze_btn.click(
                    fn=load_historical,
                    outputs=historical_output
                )

            # Tab 2: Sales Prediction
            with gr.Tab("🔮 Sales Prediction"):
                gr.Markdown("### Generate sales forecasts using machine learning")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input Parameters")

                        # Prediction inputs
                        date_input = gr.Textbox(
                            label="Prediction Date",
                            placeholder="YYYY-MM-DD",
                            info="Enter the date for prediction",
                            value=""
                        )

                        # Additional parameters can be added here
                        forecast_horizon = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=7,
                            step=1,
                            label="Forecast Horizon (days)",
                            info="Number of days to forecast"
                        )

                        predict_btn = gr.Button(
                            "Generate Prediction",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### Prediction Results")
                        prediction_output = gr.HTML(label="Forecast")

                def make_prediction(date_str, horizon):
                    try:
                        return create_sales_prediction_interface(
                            data,
                            model,
                            feature_stats,
                            feature_engineered_data,
                            date_str,
                            horizon
                        )
                    except Exception as e:
                        return f"<div style='color: red; padding: 20px;'>Error generating prediction: {str(e)}</div>"

                predict_btn.click(
                    fn=make_prediction,
                    inputs=[date_input, forecast_horizon],
                    outputs=prediction_output
                )

        # Footer
        ui.create_footer(
            logo_path="static/heart_sentinel.png",
            creator_name="Your Name",
            creator_link="https://your-profile-link.com",
            org_name="Your Organization",
            org_link="https://your-org-link.com"
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(
        css=ui.get_custom_css()
    )