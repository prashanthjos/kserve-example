from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["numpy", "scikit-learn", "joblib", "pandas"]
)
def train_model(
    x_train: Input[Dataset],
    y_train: Input[Dataset],
    feature_names: Input[Dataset],
    model: Output[Model],
    model_config: Output[Artifact]
):
    """Train a fraud detection model."""
    import numpy as np
    import joblib
    import json
    import os
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    print("Reading training data..", end="\n")

    with open(x_train.path) as f:
        X_train_df = pd.read_csv(f)

    print("Reading target data..", end="\n")

    with open(y_train.path) as f:
        Y_train_df = pd.read_csv(f)
    
    print("Reading training data complete..", end="\n")
    
    # Load feature names
    with open(feature_names.path, 'r') as f:
        feature_names_list = json.load(f)

    print("Started training data..", end="\n")
    
    # Train model
    model_obj = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model_obj.fit(X_train_df, Y_train_df)

    print("Finished training data..", end="\n")
    
    # Save model
    joblib.dump(model_obj, model.path)

    print("Finished Saving Model..", end="\n")
    
    # Save model config
    model_config_dict = {
        'feature_names': feature_names_list,
        'model_type': 'RandomForestClassifier',
        'threshold': 0.5,
        'positive_class': 1
    }
    
    with open(model_config.path, 'w') as f:
        json.dump(model_config_dict, f)