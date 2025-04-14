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
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    X_train = np.load(x_train.path)
    y_train = np.load(y_train.path)
    
    # Load feature names
    with open(feature_names.path, 'r') as f:
        feature_names_list = json.load(f)
    
    # Train model
    model_obj = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model_obj.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model_obj, model.path)
    
    # Save model config
    model_config_dict = {
        'feature_names': feature_names_list,
        'model_type': 'RandomForestClassifier',
        'threshold': 0.5,
        'positive_class': 1
    }
    
    with open(model_config.path, 'w') as f:
        json.dump(model_config_dict, f)