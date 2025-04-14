from kfp import dsl
from typing import Dict
import os

from .preprocess_component import preprocess_data
from .train_component import train_model
from .evaluate_component import evaluate_model

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kubernetes"]
)
def register_model(
    model_path: str,
    metrics: Dict[str, float],
    model_name: str,
    model_version: str
) -> str:
    """Register the model in the Kubeflow Model Registry."""
    import os
    import json
    from kubernetes import client, config
    
    # Load in-cluster config
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
    
    # Create model registry entry
    k8s_client = client.CustomObjectsApi()
    
    model_registry_entry = {
        "apiVersion": "serving.kubeflow.org/v1alpha1",
        "kind": "TrainedModel",
        "metadata": {
            "name": f"{model_name}-{model_version}",
            "namespace": "kubeflow"
        },
        "spec": {
            "model": {
                "name": model_name,
                "version": model_version,
                "framework": "sklearn",
                "storageUri": model_path,
                "metadata": {
                    "metrics": metrics
                }
            }
        }
    }
    
    # Create TrainedModel custom resource
    try:
        response = k8s_client.create_namespaced_custom_object(
            group="serving.kubeflow.org",
            version="v1alpha1",
            namespace="kubeflow",
            plural="trainedmodels",
            body=model_registry_entry
        )
        return f"{model_name}-{model_version}"
    except Exception as e:
        print(f"Error registering model: {e}")
        raise e

@dsl.pipeline(
    name="Fraud Detection Model Pipeline",
    description="A pipeline to train and register a fraud detection model."
)
def fraud_detection_pipeline(
    data_path: str,
    model_name: str = "fraud-detection",
    model_version: str = "v1"
):
    # Define the PVC to use for artifacts
    pvc_name = "kubeflow-artifact-storage"
    
    # Preprocess data
    preprocess_op = preprocess_data(data_path=data_path)
    
    # Set resource requests
    preprocess_op.set_cpu_request('1')
    preprocess_op.set_memory_request('2G')
    
    # Train model
    train_op = train_model(
        x_train=preprocess_op.outputs["x_train"],
        y_train=preprocess_op.outputs["y_train"],
        feature_names=preprocess_op.outputs["feature_names"]
    )
    
    # Set resource requests
    train_op.set_cpu_request('2')
    train_op.set_memory_request('4G')
    
    # Add volume mounting - using the correct KFP 2.0 method
    pvc_volume = dsl.PipelineVolume(pvc=pvc_name)
    train_op = train_op.add_volume(pvc_volume)
    train_op = train_op.add_volume_mount(pvc_volume, mount_path="/mnt/artifacts")
    
    # Copy model to PVC for KServe
    train_op = train_op.append_or_extend_command([
        'mkdir -p /mnt/artifacts/{}/{} && '.format(model_name, model_version),
        'cp {} /mnt/artifacts/{}/{}/model.joblib && '.format(
            train_op.outputs["model"].path, model_name, model_version),
        'cp {} /mnt/artifacts/{}/{}/model_config.json'.format(
            train_op.outputs["model_config"].path, model_name, model_version)
    ])
    
    # Evaluate model
    evaluate_op = evaluate_model(
        model=train_op.outputs["model"],
        x_test=preprocess_op.outputs["x_test"],
        y_test=preprocess_op.outputs["y_test"],
        feature_names=preprocess_op.outputs["feature_names"]
    )
    
    # Set resource requests
    evaluate_op.set_cpu_request('1')
    evaluate_op.set_memory_request('2G')
    
    # Add volume mounting for evaluate task
    evaluate_op = evaluate_op.add_volume(pvc_volume)
    evaluate_op = evaluate_op.add_volume_mount(pvc_volume, mount_path="/mnt/artifacts")
    
    # Copy metrics to PVC
    evaluate_op = evaluate_op.append_or_extend_command([
        'mkdir -p /mnt/artifacts/{}/{}/metrics && '.format(model_name, model_version),
        'cp {} /mnt/artifacts/{}/{}/metrics/metrics.json && '.format(
            evaluate_op.outputs["metrics_output"].path, model_name, model_version),
        'cp {} /mnt/artifacts/{}/{}/metrics/confusion_matrix.csv && '.format(
            evaluate_op.outputs["confusion_matrix"].path, model_name, model_version),
        'cp {} /mnt/artifacts/{}/{}/metrics/shap_values.npy'.format(
            evaluate_op.outputs["shap_values"].path, model_name, model_version)
    ])
    
    # Register model
    register_op = register_model(
        model_path=f"pvc://{pvc_name}/{model_name}/{model_version}",
        metrics=evaluate_op.outputs,
        model_name=model_name,
        model_version=model_version
    )
    
    # Set resource requests
    register_op.set_cpu_request('0.5')
    register_op.set_memory_request('1G')