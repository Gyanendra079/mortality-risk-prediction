from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load your models and preprocessor
with open('models/preprocessing.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

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

@app.route('/')
def home():
    return render_template('home.html')


# -----------------------------
# ✅ Route 1: JSON API response
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict_json():
    try:
        data = request.json['features']
        df_input = pd.DataFrame([data], columns=feature_names)

        for col in df_input.columns:
            if col != 'Sex_adm':
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

        processed = preprocessor.transform(df_input)
        prediction = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "mortality_risk_probability": round(float(prob), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# --------------------------------------
# ✅ Route 2: Web Form (HTML interface)
# --------------------------------------
@app.route('/predict_page', methods=['POST'])
def predict_page():
    try:
        data = request.form['features']
        data = [x.strip() for x in data.split(',')]

        df_input = pd.DataFrame([data], columns=feature_names)
        for col in df_input.columns:
            if col != 'Sex_adm':
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

        processed = preprocessor.transform(df_input)
        prediction = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        if prediction == 1:
            prediction_text = f"⚠️ High Mortality Risk (Probability: {prob:.3f})"
        else:
            prediction_text = f"✅ Low Mortality Risk (Probability: {prob:.3f})"

        return render_template('home.html', prediction_text=prediction_text)

    except Exception as e:
        return render_template('home.html', prediction_text=f"❌ Error: {e}")


if __name__ == '__main__':
    app.run(debug=True)
