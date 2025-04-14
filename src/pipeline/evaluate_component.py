from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact, Metrics
from typing import Dict

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["numpy", "scikit-learn", "joblib", "pandas", "matplotlib", "shap"]
)
def evaluate_model(
    model: Input[Model],
    x_test: Input[Dataset],
    y_test: Input[Dataset],
    feature_names: Input[Dataset],
    metrics_output: Output[Metrics],
    confusion_matrix: Output[Dataset],
    shap_values: Output[Dataset]
) -> Dict[str, float]:
    """Evaluate the trained model and generate SHAP explanations."""
    import numpy as np
    import pandas as pd
    import joblib
    import json
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix as cm_func, classification_report
    )
    import shap
    
    # Load data and model
    X_test = np.load(x_test.path)
    y_test = np.load(y_test.path)
    model_obj = joblib.load(model.path)
    
    with open(feature_names.path, 'r') as f:
        feature_names_list = json.load(f)
    
    # Make predictions
    y_pred = model_obj.predict(X_test)
    y_prob = model_obj.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob))
    }
    
    # Log metrics to the metrics_output artifact
    for metric_name, metric_value in metrics.items():
        metrics_output.log_metric(metric_name, metric_value)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_output.log_metrics({f"class_{k}_{metric}": v for k, v_dict in report.items() 
                           if isinstance(v_dict, dict) for metric, v in v_dict.items()})
    
    # Generate confusion matrix
    cm = cm_func(y_test, y_pred)
    pd.DataFrame(cm).to_csv(confusion_matrix.path, index=False)
    
    # Generate SHAP values (for model explainability)
    explainer = shap.TreeExplainer(model_obj)
    shap_values_output = explainer.shap_values(X_test[:100])  # Sample for speed
    np.save(shap_values.path, shap_values_output)
    
    # Return metrics
    return metrics