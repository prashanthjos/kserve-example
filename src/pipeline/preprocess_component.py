from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def preprocess_data(
    data_path: str,
    x_train: Output[Dataset],
    x_test: Output[Dataset], 
    y_train: Output[Dataset],
    y_test: Output[Dataset],
    scaler: Output[Artifact],
    feature_names: Output[Dataset]
):
    """Preprocess the data and split it into train and test sets."""
    import pandas as pd
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    import json
    
    # Load data
    print(f"Reading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler_obj = StandardScaler()
    X_train_scaled = scaler_obj.fit_transform(X_train)
    X_test_scaled = scaler_obj.transform(X_test)
    
    # Save preprocessed data
    np.save(x_train.path, X_train_scaled)
    np.save(x_test.path, X_test_scaled)
    np.save(y_train.path, y_train.values)
    np.save(y_test.path, y_test.values)
    
    # Save scaler
    joblib.dump(scaler_obj, scaler.path)
    
    # Save feature names
    with open(feature_names.path, 'w') as f:
        json.dump(X.columns.tolist(), f)