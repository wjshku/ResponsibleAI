# Interface for interpret_mcp

from interpret_tools.interpret_linear import get_coefficients, interactive_coefficients
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import json
import pandas as pd
import pyperclip
from sklearn.pipeline import Pipeline

def build_model(model='linear'):
    # Load California Housing dataset
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    feature_names = california.feature_names
    feature_description = {
        "MedInc": "Median income in block group",
        "HouseAge": "Median house age in block group",
        "AveRooms": "Average number of rooms per household",
        "AveBedrms": "Average number of bedrooms per household",
        "Population": "Block group population",
        "AveOccup": "Average number of occupants per household",
        "Latitude": "Latitude",
        "Longitude": "Longitude"
    }
    feature_names = [feature_description[name] for name in feature_names]
    X = california.data
    Y = california.target
    # Preprocess the data, Add cross-product features, 
    # with sklearn PolynomialFeatures, and add names to the features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_transformed = poly.fit_transform(X)
    full_feature_names = poly.get_feature_names_out(feature_names).tolist()
    if model == 'linear':
        model = LinearRegression()
    elif model == 'ridge':
        model = Ridge()
    elif model == 'lasso':
        model = Lasso()
    elif model == 'tree':
        model = DecisionTreeRegressor()
    model.fit(X_transformed, Y)
    pipeline = Pipeline([('poly', poly), ('model', model)])
    return model, X, Y, X_transformed, feature_names, full_feature_names, pipeline

def save_analysis():
    # Build a linear model
    _, X, Y, X_transformed, feature_names, full_feature_names, pipeline = build_model('tree')
    model = pipeline.named_steps['model']
    model_interpretation:dict = get_coefficients(model, X_transformed, Y, 
                        feature_names=full_feature_names)
    # Save the model interpretation to a file, formatted with indent and easy to read
    with open("model_interpretation.json", "w") as f:
        json.dump(model_interpretation, f, indent=4)

def interact():
    # Build a linear model
    _, X, Y, _, feature_names, _, pipeline = build_model('tree')

    interactive_coefficients(pipeline, X, Y, feature_names=feature_names)

def read_analysis():
    # Read the model interpretation from a file, 小数点 改为 科学计数法
    with open("model_interpretation.json", "r") as f:
        model_interpretation = json.load(f)
    
    # Convert decimal numbers to scientific notation for better readability
    def convert_to_scientific(obj):
        if isinstance(obj, dict):
            return {k: convert_to_scientific(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_scientific(item) for item in obj]
        elif isinstance(obj, float):
            return f"{obj:.2e}" if abs(obj) < 0.01 or abs(obj) > 1000 else f"{obj:.2f}"
        else:
            return obj
    
    # Apply scientific notation conversion
    formatted_interpretation = convert_to_scientific(model_interpretation)

    # Write prompt for ChatGPT to interpret the model
    prompt = f"""
    You are a helpful scientist that can explain the model interpretation for non-technical audience.
    The model interpretation is as follows:
    {formatted_interpretation}
    Avoid using technical terms and jargon. Explain in natural language. Be concise and to the point.
    Write in markdown format.
    """
    # Save the formatted interpretation to clipboard
    pyperclip.copy(prompt)

    print(formatted_interpretation)

    return formatted_interpretation

if __name__ == "__main__":
    # save_analysis() # Save the model interpretation to a file
    interact() # Interact with the model interpretation
    # read_analysis() # Read the model interpretation from a file