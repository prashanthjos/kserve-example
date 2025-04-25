from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def preprocess_data(
    data: Input[Dataset],
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
    print(f"Reading data from: {data.path}")
    df = pd.read_csv(data.path)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data - use different variable names to avoid conflict
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler_obj = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler_obj.fit_transform(X_train_df))
    X_test_scaled = pd.DataFrame(scaler_obj.transform(X_test_df))


    with open(x_train.path, 'w') as f:
        X_train_scaled.to_csv(f)

    with open(x_test.path, 'w') as f:
        X_test_scaled.to_csv(f)

    with open(y_train.path, 'w') as f:
        y_train_df.to_csv(f)

    with open(y_test.path, 'w') as f:
        y_test_df.to_csv(f)
    
    # Save scaler
    joblib.dump(scaler_obj, scaler.path)
    
    # Save feature names
    with open(feature_names.path, 'w') as f:
        json.dump(X.columns.tolist(), f)