from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def preprocess_data(
    data: Input[Dataset],
    x_train: Output[Dataset],
    x_test: Output[Dataset], 
    y_train: Output[Dataset],
    y_test: Output[Dataset],
    scaler: Output[Artifact],
    feature_names: Output[Dataset],
    feature_stats: Output[Dataset]
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
    df = pd.read_csv(data.path, index_col=0)  # Properly handle index column
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Original X shape: {X.shape}")
    print(f"Original feature names: {X.columns.tolist()}")
    
    # Split data - use different variable names to avoid conflict
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features - PRESERVE COLUMN NAMES
    scaler_obj = StandardScaler()
    
   
    X_train_scaled = pd.DataFrame(
        scaler_obj.fit_transform(X_train_df), 
        columns=X_train_df.columns,  # Preserve original column names
        index=X_train_df.index       # Preserve original index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler_obj.transform(X_test_df), 
        columns=X_test_df.columns,   # Preserve original column names
        index=X_test_df.index        # Preserve original index
    )
    
    print(f"Scaled X_train shape: {X_train_scaled.shape}")
    print(f"Scaled X_test shape: {X_test_scaled.shape}")
    print(f"Feature names after scaling: {X_train_scaled.columns.tolist()}")

    # ✅ FIX: Save CSVs without index to avoid extra columns
    with open(x_train.path, 'w') as f:
        X_train_scaled.to_csv(f, index=False)  # Don't save index

    with open(x_test.path, 'w') as f:
        X_test_scaled.to_csv(f, index=False)   # Don't save index

    with open(y_train.path, 'w') as f:
        y_train_df.to_csv(f, index=False)      # Don't save index

    with open(y_test.path, 'w') as f:
        y_test_df.to_csv(f, index=False)       # Don't save index
    
    # Save scaler
    joblib.dump(scaler_obj, scaler.path)
    
    with open(feature_names.path, 'w') as f:
        json.dump(X_train_scaled.columns.tolist(), f)
    
    print(f"Saved {len(X_train_scaled.columns)} feature names: {X_train_scaled.columns.tolist()}")
    
    # Calculate and save feature statistics for transformer validation
    feature_stats_dict = {}
    for col in X_train_scaled.columns:
        feature_stats_dict[col] = {
            'mean': float(X_train_scaled[col].mean()),
            'std': float(X_train_scaled[col].std()),
            'min': float(X_train_scaled[col].min()),
            'max': float(X_train_scaled[col].max()),
            'median': float(X_train_scaled[col].median())
        }
    
    with open(feature_stats.path, 'w') as f:
        json.dump(feature_stats_dict, f, indent=2)
    
    print(f"Saved feature statistics for {len(feature_stats_dict)} features")