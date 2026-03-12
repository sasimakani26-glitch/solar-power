import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

def plot_forecast(prediction):

    times = pd.date_range(datetime.now(), periods=4, freq="15min")
    forecast = [prediction * (1 + i*0.02) for i in range(4)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=forecast,
        mode='lines+markers'
    ))

    fig.update_layout(
        title="Next 1 Hour Forecast (15-min Interval)",
        xaxis_title="Time",
        yaxis_title="Power (kW)"
    )

    return fig