from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact, Metrics
from typing import Dict
import os

from .preprocess_component import preprocess_data
from .generate_data import generate_synthetic_data
from .train_component import train_model
from .evaluate_component import evaluate_model
from.deploy_model import deploy_model

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kubernetes", "model-registry"]
)
def register_model(
    model: Input[Model],
    model_name: str,
    model_version: str
) -> str:
    """Register the model in the Kubeflow Model Registry."""
    from model_registry import ModelRegistry
    import json
    
    # Print inputs for debugging
    print(f"Registering model with path: {model.path}")
    print(f"Model name: {model_name}")
    print(f"Model version: {model_version}")
    
    try:
        # Initialize the Model Registry client
        registry = ModelRegistry(
            server_address="http://model-registry-service.kubeflow-user-example-com.svc.cluster.local",
            port=8080,
            author="Prashanth Josyula",
            is_secure=False
        )
        
        # Register model
        registered_model = registry.register_model(
            model_name,
            model.path,
            model_format_name="sklearn",
            model_format_version="1",
            version=model_version, 
            description="Fraud detection model",
            metadata={
                "accuracy": 3.14,
                "license": "BSD 3-Clause License",
            }
        )
        
        print(f"Successfully registered model: {model_name} version: {model_version}")
        return f"{model_name}-{model_version}"
        
    except Exception as e:
        print(f"Error registering model: {e}")
        # Return a value even on error to satisfy the function signature
        return f"Error-{model_name}-{model_version}"


@dsl.pipeline(
    name="Fraud Detection Model Pipeline",
    description="A pipeline to train and register a fraud detection model."
)
def fraud_detection_pipeline(
    model_name: str = "fraud-detection-model",
    model_version: str = "v1"
):
    
    # Generate Datamodel_version
    generate_data_op = generate_synthetic_data()
    
    # Preprocess data
    preprocess_op = preprocess_data(data=generate_data_op.outputs["data_set"])
    preprocess_op.after(generate_data_op)  # Ensure data generation completes first
    
    # Set resource requests
    preprocess_op.set_cpu_request('1')
    preprocess_op.set_memory_request('2G')
    
    
    # Train model
    train_op = train_model(
        x_train=preprocess_op.outputs["x_train"],
        y_train=preprocess_op.outputs["y_train"],
        feature_names=preprocess_op.outputs["feature_names"]
    )
    train_op.after(preprocess_op)  # Ensure preprocessing completes first
    
    # Set resource requests
    train_op.set_cpu_request('2')
    train_op.set_memory_request('4G')
    
    
    # Evaluate model
    evaluate_op = evaluate_model(
        model=train_op.outputs["model"],
        x_test=preprocess_op.outputs["x_test"],
        y_test=preprocess_op.outputs["y_test"],
        feature_names=preprocess_op.outputs["feature_names"]
    )
    evaluate_op.after(train_op)  # Ensure training completes first
    
    # Set resource requests
    evaluate_op.set_cpu_request('1')
    evaluate_op.set_memory_request('2G')
    
    
    # Register model
    register_op = register_model(
        model=train_op.outputs["model"], 
        model_name=model_name,
        model_version=model_version
    )
    register_op.after(evaluate_op)  # Ensure evaluation completes first
    
    
    # Set resource requests
    register_op.set_cpu_request('0.5')
    register_op.set_memory_request('1G')

    # Deploy model
    deploy_op = deploy_model(
        model_name=model_name,
        model_version=model_version
    )
    deploy_op.after(register_op)  # Ensure evaluation completes first
    
    
    # Set resource requests
    deploy_op.set_cpu_request('0.5')
    deploy_op.set_memory_request('1G')



