
from kfp.dsl import Input, Output, Dataset, Artifact, component

# Generate synthetic data for credit card fraud detection
@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn", "pandas", "numpy"]
)
def generate_synthetic_data(data_set: Output[Dataset]):

    from sklearn.datasets import make_classification
    import pandas as pd
    import os
    import numpy as np

    output_path="data/credit_card_data.csv"

    n_samples = 100

    # Create a synthetic dataset with imbalanced classes (fraud is rare)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,  # Common features in credit card data
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.97, 0.03],  # 3% fraud rate (imbalanced)
        random_state=42
    )
    
    # Create feature names similar to credit card transaction data
    feature_names = []
    # Transaction amount and time
    feature_names.append('Amount')
    feature_names.append('Time')
    # Add PCA-like features (V1-V28) as often seen in fraud datasets
    for i in range(1, 29):
        feature_names.append(f'V{i}')
    
    # Create dataframe
    data = pd.DataFrame(X, columns=feature_names)
    data['Class'] = y  # 0 for legitimate, 1 for fraud
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    with open(data_set.path, 'w') as f:
        data.to_csv(f)