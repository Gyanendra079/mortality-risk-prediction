# dash_app.py
import os
import pickle
import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output, State

# -----------------------------------
# Load Model and Preprocessor
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'models', 'preprocessing.pkl'), 'rb') as f:
    preprocessor = pickle.load(f)

with open(os.path.join(BASE_DIR, 'models', 'classifier.pkl'), 'rb') as f:
    model = pickle.load(f)

# -----------------------------------
# Define feature names
# -----------------------------------
feature_names = [
    'Year', 'Sex_adm', 'Tobacco Price\nIndex', 'Retail Prices\nIndex',
    'Tobacco Price Index Relative to Retail Price Index', 'Real Households\' Disposable Income',
    'Affordability of Tobacco Index', 'Household Expenditure on Tobacco',
    'Household Expenditure Total', 'Expenditure on Tobacco as a Percentage of Expenditure',
    'smoking_prevalence', 'total_prescriptions', 'nrt_prescription_ratio',
    'bupropion_prescription_ratio', 'varenicline_prescription_ratio', 'total_prescription_cost',
    'nrt_cost_ratio', 'bupropion_cost_ratio', 'varenicline_cost_ratio', 'tobacco_expenditure_ratio',
    'tobacco_price_relative_index', '16 and Over', '16-24', '25-34', '35-49', '50-59', '60 and Over'
]

# -----------------------------------
# Initialize Dash App
# -----------------------------------
app = dash.Dash(__name__)
server = app.server  # for deployment (Flask-compatible)

# -----------------------------------
# Layout
# -----------------------------------
app.layout = html.Div(
    style={
        'maxWidth': '900px', 'margin': 'auto', 'padding': '40px',
        'fontFamily': 'Segoe UI', 'backgroundColor': '#f4f7fb', 'borderRadius': '10px'
    },
    children=[
        html.H1("🩺 Mortality Risk Prediction Dashboard", style={'textAlign': 'center', 'color': '#0d47a1'}),

        html.Hr(),

        html.Div([
            html.Label("Select Year:"),
            dcc.Slider(2000, 2025, 1, value=2020, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id='year'),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("Sex:"),
            dcc.Dropdown(
                options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
                value='Male', id='sex'
            ),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("Tobacco Price Index:"),
            dcc.Input(id='tobacco_index', type='number', value=100, step=0.1, style={'width': '100%'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("Retail Prices Index:"),
            dcc.Input(id='retail_index', type='number', value=100, step=0.1, style={'width': '100%'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("Smoking Prevalence (%):"),
            dcc.Slider(0, 100, 1, value=25, tooltip={"placement": "bottom", "always_visible": True}, id='smoking_prev'),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("Total Prescriptions:"),
            dcc.Input(id='total_prescriptions', type='number', value=1000, step=1, style={'width': '100%'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("NRT Prescription Ratio:"),
            dcc.Slider(0, 1, 0.01, value=0.5, tooltip={"placement": "bottom", "always_visible": True}, id='nrt_ratio'),
        ], style={'marginBottom': '20px'}),

        html.Button('Predict', id='predict-btn', n_clicks=0,
                    style={'backgroundColor': '#0d6efd', 'color': 'white',
                           'border': 'none', 'padding': '10px 25px',
                           'borderRadius': '8px', 'fontSize': '16px'}),

        html.Div(id='prediction-output', style={'marginTop': '30px', 'textAlign': 'center', 'fontSize': '20px'})
    ]
)

# -----------------------------------
# Callback Function
# -----------------------------------
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('year', 'value'),
    State('sex', 'value'),
    State('tobacco_index', 'value'),
    State('retail_index', 'value'),
    State('smoking_prev', 'value'),
    State('total_prescriptions', 'value'),
    State('nrt_ratio', 'value')
)
def predict(n_clicks, year, sex, tob_idx, ret_idx, smoking_prev, total_rx, nrt_ratio):
    if not n_clicks:
        return ""

    try:
        # Create a DataFrame (fill only available features, rest as NaN)
        data = {
            'Year': [year],
            'Sex_adm': [sex],
            'Tobacco Price\nIndex': [tob_idx],
            'Retail Prices\nIndex': [ret_idx],
            'smoking_prevalence': [smoking_prev],
            'total_prescriptions': [total_rx],
            'nrt_prescription_ratio': [nrt_ratio]
        }

        df_input = pd.DataFrame(data)

        # Ensure all required columns exist (fill missing ones with NaN)
        for f in feature_names:
            if f not in df_input.columns:
                df_input[f] = np.nan

        processed = preprocessor.transform(df_input)
        prediction = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        if prediction == 1:
            return html.Span(
                f"⚠️ High Mortality Risk (Probability: {prob:.3f})",
                style={'color': 'red', 'fontWeight': 'bold'}
            )
        else:
            return html.Span(
                f"✅ Low Mortality Risk (Probability: {prob:.3f})",
                style={'color': 'green', 'fontWeight': 'bold'}
            )
    except Exception as e:
        return f"❌ Error: {e}"

# -----------------------------------
# Run the app
# -----------------------------------
if __name__ == '__main__':
    app.run(debug=True)

