import os
from data_preprocessing import load_and_preprocess_data
from model_trainer import train_and_evaluate

def main():
    print("Starting Next Generation Financial Threat Detection System...")
    
    # This automatically finds exactly where main.py is located on your MacBook
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # It then safely builds the path to your dataset folder
    data_path = os.path.join(script_dir, 'dataset', 'creditcard.csv')
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Could not find {data_path}. Did you put the CSV in the dataset folder?")
        return

    # Run the pipeline
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()