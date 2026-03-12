"""
Package initializer for modules.
This file makes the `modules` directory a proper Python package so imports
like `from modules.weather_service import fetch_live_weather` work in all
execution environments (Streamlit, tests, deployments).
"""

__all__ = [
    "weather_service",
    "preprocessing",
    "data_loader",
    "visualization",
    "train_models",
    "predictor",
    "model_selector",
    "evaluate_models",
    "deep_learning_models",
    "create_new model",
]
