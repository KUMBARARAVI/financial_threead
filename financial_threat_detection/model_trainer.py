import os
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("[INFO] Initializing Next-Gen XGBoost Model...")
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
    
    model = xgb.XGBClassifier(
        scale_pos_weight=ratio, 
        random_state=42, 
        eval_metric='logloss'
    )
    
    print("[INFO] Training model (this may take a minute or two)...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # --- EVALUATION ---
    print("\n" + "="*50)
    print("THREAT DETECTION REPORT")
    print("="*50)
    print(classification_report(y_test, predictions, target_names=['Normal (0)', 'Fraud (1)']))
    
    # --- VISUALIZATION ---
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Financial Threat Detection - Confusion Matrix')
    
    # --- BULLETPROOF PATHS FOR SAVING ---
    # This finds the exact folder where model_trainer.py lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    png_path = os.path.join(base_dir, 'threat_detection_matrix.png')
    model_path = os.path.join(base_dir, 'xgboost_fraud_model.joblib')
    test_data_path = os.path.join(base_dir, 'dataset', 'live_test_data.csv')
    
    # Save the PNG
    plt.savefig(png_path)
    print(f"[INFO] Saved matrix to: {png_path}")
    
    # Save the Model
    print("[INFO] Saving model and test data for the backend server...")
    joblib.dump(model, model_path)
    print(f"[INFO] Model successfully saved to: {model_path}")
    
    # Save the Simulation Data
    test_data = X_test.copy()
    test_data['Actual_Class'] = y_test
    test_data.to_csv(test_data_path, index=False)
    print(f"[INFO] Simulation data successfully saved to: {test_data_path}")
    
    print("\n[SUCCESS] Training Pipeline Complete. Ready to launch web server!")