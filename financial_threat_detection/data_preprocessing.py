import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Scale 'Time' and 'Amount' features so large transactions don't skew the model
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop old columns and rearrange
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Define features (X) and target/labels (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("[INFO] Splitting data into training and testing sets...")
    # 80% for training, 20% for testing. Stratify ensures the fraud ratio remains the same in both sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test