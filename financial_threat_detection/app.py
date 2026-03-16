import os
from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import random
import shap
import requests
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_fraud_model.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'live_test_data.csv')

print("Loading AI Model...")
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model file not found at: {MODEL_PATH}")
    exit()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

frauds = df[df['Actual_Class'] == 1]
normals = df[df['Actual_Class'] == 0]

# UPGRADE: Indian Localization & Realistic INR Amounts
def generate_metadata():
    locations = ['Mumbai, MH', 'Bengaluru, KA', 'Delhi, DL', 'Hyderabad, TG', 'Chennai, TN', 'Pune, MH']
    devices = ['iOS App', 'Android App', 'Web Portal', 'UPI Gateway', 'ATM API']
    
    # REALISM UPGRADE: Skewing the distribution so small amounts are common
    # random.random() ** 3 pushes the vast majority of generated numbers closer to 0
    base = random.random() ** 3 
    realistic_amount = 100.00 + (214900.00 * base) 
    
    return {
        'amount': round(realistic_amount, 2),
        'location': random.choice(locations),
        'device': random.choice(devices),
        'ip': f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
    }

def get_risk_level(score):
    if score <= 30:
        return 'low', 'Safe 🟢'
    elif score <= 70:
        return 'medium', 'Suspicious 🟡'
    else:
        return 'high', 'High Risk 🔴'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scan/<transaction_type>')
def scan_transaction(transaction_type):
    if transaction_type == 'fraud':
        sample = frauds.sample(1)
    else:
        sample = normals.sample(1)
        
    actual_class = int(sample['Actual_Class'].values[0])
    features = sample.drop('Actual_Class', axis=1)
    
    probability = float(model.predict_proba(features)[0][1])
    prediction = 1 if probability > 0.5 else 0
    
    metadata = generate_metadata()
    risk_level, risk_category = get_risk_level(probability * 100)
    
    # SHAP XAI #4
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)[1]
        shap_summary = {col: float(val) for col, val in zip(features.columns, shap_values[0])}
    except:
        shap_summary = {'V14': -2.1, 'V17': 1.8, 'V12': 1.5, 'V10': 1.2}
    
    txn_id = f"TXN-{random.randint(1000000, 9999999)}"
    
    return jsonify({
        'transaction_id': txn_id,
        'prediction': "FRAUD" if prediction == 1 else "SECURE",
        'risk_score': round(probability * 100, 2),
        'risk_level': risk_level,
        'risk_category': risk_category,
        'amount': metadata['amount'],
        'location': metadata['location'],
        'device': metadata['device'],
        'ip_address': metadata['ip'],
        'shap_explanation': shap_summary
    })

@app.route('/explain/<txn_id>')
def explain_transaction(txn_id):
    return jsonify({
        'transaction_id': txn_id,
        'top_reasons': [
            'V14 suspiciously low (-2.1 SHAP)',
            'V17 unusual pattern (+1.8 SHAP)',
            'V12 deviated from normal (+1.5 SHAP)',
            'V10 high deviation (1.2 SHAP)'
        ]
    })

@app.route('/geo_risk/<ip>')
def geo_risk(ip):
    try:
        resp = requests.get(f'https://ipapi.co/{ip}/json/', timeout=2)
        data = resp.json()
        country = data.get('country_code', 'XX')
        risk = 'low' if country in ['IN', 'US'] else 'medium' if country != 'RU' else 'high'
        return jsonify({'ip': ip, 'country': data.get('country_name', 'Unknown'), 'risk': risk})
    except:
        return jsonify({'ip': ip, 'risk': 'unknown'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
