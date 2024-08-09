import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    # Assuming target column is 'target'
    X = data.drop('target', axis=1)
    y = data['target']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
