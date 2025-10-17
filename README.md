# 🩺 Mortality Risk Prediction Web App

This project uses Flask and Dash to predict patient mortality risk based on health and socioeconomic data.

## Features
- Flask-based backend model
- Interactive Dash dashboard for predictions
- Model trained and saved in Google Colab

## Files
- `app.py`: Flask API version
- `dash_app.py`: Dash web UI
- `models/`: Pickle files for model and preprocessor
- `Mortality_Risk_Model.ipynb`: Notebook used for model training

### Flask App
<img width="930" height="440" alt="Image" src="https://github.com/user-attachments/assets/26351b30-636d-46bf-9b27-7d5e2de868c0" />

### Dash App
<img width="811" height="415" alt="Image" src="https://github.com/user-attachments/assets/0ee94d28-3cdf-4136-9443-26c120b662d1" />

Structure of Folder
project/
│
├── model/
│   ├── classifier.pkl          ← your trained model
│   └── preprocessing.pkl       ← (optional) your scaler/encoder
│
├── app.py                      ← Flask API file
├── requirements.txt
└── templates/
    └── index.html              ← (optional) simple UI
