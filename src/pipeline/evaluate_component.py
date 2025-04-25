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
    
    # Read the X test data
    with open(x_test.path) as f:
        X_test_df = pd.read_csv(f)

    # Read the Y test data - be more careful with this
    with open(y_test.path) as f:
        Y_test_df = pd.read_csv(f)
    
    # Print debug info about the dataframes    
    print(f"X_test_df shape: {X_test_df.shape}")
    print(f"Y_test_df shape: {Y_test_df.shape}")
    print(f"Y_test_df columns: {Y_test_df.columns.tolist()}")
    print(f"First few rows of Y_test_df:\n{Y_test_df.head()}")
        
    # Make sure Y_test_df has the same number of rows as X_test_df
    if len(Y_test_df) != len(X_test_df):
        print(f"WARNING: Length mismatch between Y_test_df ({len(Y_test_df)}) and X_test_df ({len(X_test_df)})")
        # If Y_test_df has an index column that matches X_test_df, we can try to align them
        if 'index' in Y_test_df.columns:
            print("Attempting to align Y_test_df with X_test_df using index column")
            # Reset index to ensure proper alignment
            X_test_df = X_test_df.reset_index(drop=True)
            Y_test_df = Y_test_df.reset_index(drop=True)
        else:
            # Use the first len(X_test_df) rows from Y_test_df
            print(f"Taking first {len(X_test_df)} rows from Y_test_df")
            Y_test_df = Y_test_df.iloc[:len(X_test_df)]

    # Extract the target values from the DataFrame - make sure it's a 1D array
    # If it's binary classification, there might be only one target column
    if len(Y_test_df.columns) == 1:
        y_true = Y_test_df.iloc[:, 0].values
    else:
        # Look for the target column (often named 'target', 'label', or 'class')
        potential_target_cols = ['target', 'label', 'class', 'fraud', 'is_fraud']
        target_col = None
        for col in potential_target_cols:
            if col in Y_test_df.columns:
                target_col = col
                break
        
        if target_col is not None:
            y_true = Y_test_df[target_col].values
        else:
            # If no known target column is found, use the last column
            y_true = Y_test_df.iloc[:, -1].values
    
    print(f"Final y_true shape: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}")
    print(f"Sample of y_true: {y_true[:5]}")

    # Load the model
    model_obj = joblib.load(model.path)
    
    # Load feature names
    with open(feature_names.path, 'r') as f:
        feature_names_list = json.load(f)

    print("Predicting started")
    
    # Make predictions - ensure we get a 1D array for classification
    y_pred_raw = model_obj.predict(X_test_df)
    
    # Check the shape of predictions and handle accordingly
    if hasattr(y_pred_raw, 'shape') and len(y_pred_raw.shape) > 1 and y_pred_raw.shape[1] > 1:
        print(f"Model returned multi-dimensional predictions with shape {y_pred_raw.shape}")
        # This could be one-hot encoded or probability predictions
        # For classification metrics, we need class labels (not probabilities)
        y_pred = np.argmax(y_pred_raw, axis=1)
    else:
        y_pred = y_pred_raw
    
    print(f"Final y_pred shape: {y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred)}")
    print(f"Sample of y_pred: {y_pred[:5]}")
    
    # Get probabilities for ROC AUC - shape should be (n_samples, n_classes)
    try:
        y_prob_raw = model_obj.predict_proba(X_test_df)
        print(f"y_prob_raw shape: {y_prob_raw.shape}")
        
        # For binary classification with ROC AUC, we need probabilities of the positive class (usually class 1)
        if y_prob_raw.shape[1] == 2:  # Binary classification
            # Take probability of positive class (index 1)
            y_prob = y_prob_raw[:, 1]
        else:  # Multi-class case
            # For multi-class, we'll need to use OneVsRest strategy - not handling that here
            y_prob = y_prob_raw
            print("Multi-class ROC AUC not supported in this component")
    except Exception as e:
        print(f"Error getting prediction probabilities: {e}")
        # Fall back to using the predictions themselves
        y_prob = y_pred
    
    print("Predicting finished")
    
    # Ensure same length for metrics calculation
    min_len = min(len(y_true), len(y_pred))
    if min_len < len(y_true) or min_len < len(y_pred):
        print(f"Truncating y_true and y_pred to length {min_len}")
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        if len(y_prob) > min_len:
            y_prob = y_prob[:min_len]
    
    # Calculate metrics with error handling
    metrics = {}
    
    try:
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        metrics['accuracy'] = 0.0
        
    try:
        metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
    except Exception as e:
        print(f"Error calculating precision: {e}")
        metrics['precision'] = 0.0
        
    try:
        metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
    except Exception as e:
        print(f"Error calculating recall: {e}")
        metrics['recall'] = 0.0
        
    try:
        metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted'))
    except Exception as e:
        print(f"Error calculating F1: {e}")
        metrics['f1'] = 0.0
    
    # ROC-AUC calculation
    try:
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        else:
            # Multi-class AUC requires specific handling
            print("Skipping ROC AUC for multi-class case")
            metrics['roc_auc'] = 0.0
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")
        metrics['roc_auc'] = 0.0
    
    # Log metrics to the metrics_output artifact
    for metric_name, metric_value in metrics.items():
        metrics_output.log_metric(metric_name, metric_value)
    
    # Generate classification report
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics_output.log_metrics({f"class_{k}_{metric}": v for k, v_dict in report.items() 
                           if isinstance(v_dict, dict) for metric, v in v_dict.items()})
    except Exception as e:
        print(f"Could not generate classification report: {e}")
    
    # Generate confusion matrix
    try:
        cm = cm_func(y_true, y_pred)
        pd.DataFrame(cm).to_csv(confusion_matrix.path, index=False)
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")
        # Save empty confusion matrix
        pd.DataFrame([[0]]).to_csv(confusion_matrix.path, index=False)
    
    # Generate SHAP values (for model explainability)
    try:
        # Only use SHAP for certain model types that support it
        if hasattr(model_obj, "feature_importances_") or hasattr(model_obj, "coef_"):
            explainer = shap.TreeExplainer(model_obj)
            # Use a smaller sample size if the dataset is large
            sample_size = min(100, X_test_df.shape[0])
            shap_values_output = explainer.shap_values(X_test_df[:sample_size])
            np.save(shap_values.path, shap_values_output)
        else:
            print("Model type doesn't support SHAP TreeExplainer, skipping SHAP values")
            np.save(shap_values.path, np.array([]))
    except Exception as e:
        print(f"Could not generate SHAP values: {e}")
        # Save an empty array as a placeholder
        np.save(shap_values.path, np.array([]))
    
    # Return metrics
    return metrics