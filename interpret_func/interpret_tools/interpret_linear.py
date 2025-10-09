"""Linear model interpretation utilities.

This module implements a minimal set of interpretation tools for gaining insights into how a linear model makes predictions.

Input format
------------
interpret_linear(
    model,
    X,
    y=None,
    feature_names=None,
)

Requirements for model: exposes predict(X), coef_ (shape [n_features]) and
intercept_ (scalar). Works with scikit-learn style linear regressors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

def get_coefficients(
    model: Any,
    X: Any,
    y: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a compact interpretation suite for linear models.
    """

    # Get the coefficients and intercept
    coefs = model.coef_
    intercept = model.intercept_

    # Get the feature names
    feature_names = feature_names or model.feature_names_in_

    # Get the unstandardized coefficients
    global_section = {
        feature_names[i]: coefs[i] for i in range(len(feature_names)) if coefs[i] != 0.0
    }
    global_section["intercept"] = intercept

    # Get the standardized coefficients
    global_section["standardized"] = {
        feature_names[i]: coefs[i] / np.std(X[:, i]) for i in range(len(feature_names)) if coefs[i] != 0.0
    }

    return {
        "global": global_section,
    }

def interactive_coefficients(
    pipeline: Any,
    X: Any,
    y: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    An interactive dashboard to see the impact of each feature on the model's prediction.
    By dragging the sliders of each individual feature, the user can see how the model's 
    prediction changes in 2D space. This treats the model as a black box and allows
    exploration of the prediction landscape.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    from dash import Dash, html, dcc, Input, Output, callback
    import numpy as np
    
    # Convert inputs to numpy arrays
    X_np = np.asarray(X)
    if y is not None:
        y_np = np.asarray(y)
    else:
        y_np = None
    
    # Get feature names
    if feature_names is None:
        if hasattr(pipeline, 'feature_names_in_'):
            feature_names = list(pipeline.feature_names_in_)
        else:
            feature_names = [f"Feature_{i}" for i in range(X_np.shape[1])]
    
    # Get feature statistics for sliders
    feature_stats = {
        'min': np.percentile(X_np, 5, axis=0),
        'max': np.percentile(X_np, 95, axis=0),
        'mean': np.mean(X_np, axis=0),
        'std': np.std(X_np, axis=0)
    }
    
    # Create base prediction with mean values
    base_values = feature_stats['mean'].copy()
    base_prediction = pipeline.predict(base_values.reshape(1, -1))[0]
    
    # Initialize Dash app
    app = Dash(__name__)
    
    # Create layout
    app.layout = html.Div([
        html.H1("Interactive Model Interpretation Dashboard", 
                style={'textAlign': 'center', 'marginBottom': '30px'}),
        
        # Feature sliders
        html.Div([
            html.H3("Feature Controls", style={'marginBottom': '20px'}),
            *[html.Div([
                html.Label(f"{feature_names[i]}:", style={'fontWeight': 'bold', 'width': '200px', 'display': 'inline-block'}),
                dcc.Slider(
                    id=f'slider-{i}',
                    min=float(feature_stats['min'][i]),
                    max=float(feature_stats['max'][i]),
                    value=float(feature_stats['mean'][i]),
                    step=round((feature_stats['max'][i] - feature_stats['min'][i]) / 100, 2),
                    marks={
                        float(feature_stats['min'][i]): f"{feature_stats['min'][i]:.2f}",
                        float(feature_stats['mean'][i]): f"{feature_stats['mean'][i]:.2f}",
                        float(feature_stats['max'][i]): f"{feature_stats['max'][i]:.2f}"
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'  # Update during drag, not just on release
                )
            ], style={'marginBottom': '15px'}) for i in range(len(feature_names))]
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),
        
        # Prediction display and plots
        html.Div([
            html.H3("Model Predictions", style={'marginBottom': '20px'}),
            html.Div(id='prediction-display', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '20px'}),
            
            # Prediction history
            dcc.Graph(id='prediction-history')
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Hidden div to store prediction history
        dcc.Store(id='prediction-history-store', data=[])
    ])
    
    @app.callback(
        [Output('prediction-display', 'children'),
         Output('prediction-history', 'figure'),
         Output('prediction-history-store', 'data')],
        [Input(f'slider-{i}', 'value') for i in range(len(feature_names))] +
        [Input('prediction-history-store', 'data')]
    )
    def update_predictions(*args):
        # Extract slider values
        slider_values = args[:len(feature_names)]
        history_data = args[-1] if args[-1] is not None else []
        
        # Create input vector
        input_vector = np.array(slider_values).reshape(1, -1)
        
        # Get prediction
        prediction = pipeline.predict(input_vector)[0]
        
        # Update prediction display
        pred_text = f"Current Prediction: {prediction:.4f}"
        
        # Add current prediction to trace
        history_data.append({
            'prediction': float(prediction),
            'features': list(slider_values),
            'timestamp': len(history_data)
        })
        
        # Keep only last 100 points for smooth trace
        if len(history_data) > 100:
            history_data = history_data[-100:]
        
        # Create prediction trace plot
        if len(history_data) > 1:
            # Create a smooth trace showing prediction changes
            x_values = list(range(len(history_data)))
            y_values = [h['prediction'] for h in history_data]
            
            history_fig = go.Figure(data=[
                go.Scatter(
                    x=x_values, 
                    y=y_values,
                    mode='lines',
                    name='Prediction Trace',
                    line=dict(
                        color='rgba(0, 100, 200, 0.6)',  # Semi-transparent blue
                        width=2, 
                        dash='dash'  # Dashed line
                    ),
                    hovertemplate='<b>Prediction: %{y:.4f}</b><extra></extra>',
                    showlegend=False
                )
            ])
            
            # Add current prediction as a highlighted point
            history_fig.add_trace(go.Scatter(
                x=[len(history_data)-1], 
                y=[history_data[-1]['prediction']],
                mode='markers',
                name='Current',
                marker=dict(size=12, color='red', symbol='circle', line=dict(width=2, color='white')),
                hovertemplate='<b>Current Prediction: %{y:.4f}</b><extra></extra>',
                showlegend=False
            ))
            
            # Add prediction range indicators
            if len(history_data) > 5:
                min_pred = min(y_values)
                max_pred = max(y_values)
                history_fig.add_hline(y=min_pred, line_dash="dot", line_color="green", 
                                    annotation_text=f"Min: {min_pred:.4f}", 
                                    annotation_position="bottom right",
                                    line_width=1)
                history_fig.add_hline(y=max_pred, line_dash="dot", line_color="orange", 
                                    annotation_text=f"Max: {max_pred:.4f}", 
                                    annotation_position="top right",
                                    line_width=1)
        else:
            # Single point case
            history_fig = go.Figure(data=[
                go.Scatter(
                    x=[0], 
                    y=[history_data[0]['prediction']],
                    mode='markers',
                    name='Current',
                    marker=dict(size=12, color='red', symbol='circle', line=dict(width=2, color='white')),
                    hovertemplate='<b>Current Prediction: %{y:.4f}</b><extra></extra>',
                    showlegend=False
                )
            ])
        
        history_fig.update_layout(
            title="Prediction Trace",
            xaxis_title="",
            yaxis_title="Prediction Value",
            hovermode='y unified',
            showlegend=False,
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return pred_text, history_fig, history_data
    
    # Run the app
    app.run(debug=True, port=8050)
    
    return {
        "message": "Interactive dashboard launched at http://localhost:8050",
            "feature_names": feature_names,
        "feature_stats": {k: v.tolist() for k, v in feature_stats.items()},
        "base_prediction": float(base_prediction)
    }


__all__ = ["get_coefficients", "interactive_coefficients"]


