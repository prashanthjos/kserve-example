from kfp import dsl

@dsl.component(
    base_image="python:3.11.13",
    packages_to_install=["kserve", "kubernetes", "model-registry", "boto3", "minio", "joblib", "scikit-learn"]
)
def deploy_model(
    model_name: str, 
    model_version: str,
    scaler: dsl.Input[dsl.Artifact],
    feature_names: dsl.Input[dsl.Dataset],
    feature_stats: dsl.Input[dsl.Dataset]
):

    import kserve
    from model_registry import ModelRegistry
    from kubernetes import client
    from minio import Minio
    import os
    import tempfile
    import joblib

    def convert_minio_path_to_s3_uri(minio_path):
        """
        Convert a local MinIO path to an S3 URI format for KServe.
        
        Args:
            minio_path (str): The local MinIO path, e.g., 
                            "/minio/mlpipeline/v2/artifacts/fraud-detection-model-pipeline/..."
        
        Returns:
            str: The S3 URI format, e.g., 
                "s3://mlpipeline/v2/artifacts/fraud-detection-model-pipeline/..."
        """
        # Remove the leading '/minio/' prefix if present
        if minio_path.startswith('/minio/'):
            path_without_prefix = minio_path[7:]  # Skip '/minio/'
        else:
            path_without_prefix = minio_path.lstrip('/')
        
        # Create the S3 URI by adding the 's3://' prefix
        s3_uri = f"s3://{path_without_prefix}"
        
        return s3_uri

    def parse_minio_uri(uri):
        """
        Parse a MinIO URI into bucket and object path.
        
        Args:
            uri (str): The MinIO URI, e.g., 
                "/minio/mlpipeline/v2/artifacts/fraud-detection-model-pipeline/..."
                
        Returns:
            tuple: (bucket_name, object_path)
        """
        # Remove the leading '/minio/' prefix
        if uri.startswith('/minio/'):
            path = uri[7:]  # Skip '/minio/'
        else:
            path = uri.lstrip('/')
            
        # First part is the bucket name
        parts = path.split('/', 1)
        bucket = parts[0]
        
        # Rest is the object path
        obj_path = parts[1] if len(parts) > 1 else ""
        
        return bucket, obj_path

    registry = ModelRegistry(
                server_address="http://model-registry-service.kubeflow-user-example-com.svc.cluster.local",
                port=8080,
                author="Prashanth Josyula",
                is_secure=False
            )

    model = registry.get_registered_model(model_name)
    print("Registered Model:", model, "with ID", model.id)

    version = registry.get_model_version(model_name, model_version)
    print("Model Version:", version, "with ID", version.id)

    artifact = registry.get_model_artifact(model_name, model_version)
    print("Model Artifact:", artifact, "with ID", artifact.id)
    print(f"Original model URI: {artifact.uri}")
    
    # Connect to MinIO
    try:
        print("Connecting to MinIO...")
        minio_client = Minio(
            "minio-service.kubeflow:9000",  # Adjust if your MinIO endpoint is different
            access_key="minio",
            secret_key="minio123",
            secure=False  # Set to True if using HTTPS
        )
        
        # Parse the original URI to get bucket and object path
        bucket, object_path = parse_minio_uri(artifact.uri)
        print(f"Parsed bucket: {bucket}, object path: {object_path}")
        
        # Download the model to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        print(f"Downloading model from bucket: {bucket}, object: {object_path} to temporary file: {temp_path}")
        minio_client.fget_object(bucket, object_path, temp_path)
        
        # Load the model
        print(f"Loading model from: {temp_path}")
        model_obj = joblib.load(temp_path)
        
        # Create directory structure to separate predictor and transformer artifacts
        # Get the base directory from the object path
        if '/' in object_path:
            base_dir = os.path.dirname(object_path)
        else:
            base_dir = ""
        
        # Predictor directory - contains ONLY model.joblib (sklearn server requirement)
        predictor_dir = f"{base_dir}/pred-model" if base_dir else "model"
        model_object_path = f"{predictor_dir}/model.joblib"
        
        # Save the model with .joblib extension to a temp file
        temp_joblib_file = tempfile.NamedTemporaryFile(suffix='.joblib', delete=False)
        temp_joblib_path = temp_joblib_file.name
        temp_joblib_file.close()
        
        print(f"Saving model to temporary joblib file: {temp_joblib_path}")
        joblib.dump(model_obj, temp_joblib_path)
        
        # Upload model.joblib to predictor directory
        print(f"Uploading model to bucket: {bucket}, object: {model_object_path}")
        minio_client.fput_object(bucket, model_object_path, temp_joblib_path)
        print(f"Successfully uploaded model.joblib to predictor directory")
        
        # Upload transformer/explainer artifacts to base directory
        scaler_object_path = f"{base_dir}/scaler.joblib" if base_dir else "scaler.joblib"
        print(f"Uploading scaler to bucket: {bucket}, object: {scaler_object_path}")
        minio_client.fput_object(bucket, scaler_object_path, scaler.path)
        print(f"Successfully uploaded scaler.joblib")
        
        feature_names_object_path = f"{base_dir}/feature_names.json" if base_dir else "feature_names.json"
        print(f"Uploading feature names to bucket: {bucket}, object: {feature_names_object_path}")
        minio_client.fput_object(bucket, feature_names_object_path, feature_names.path)
        print(f"Successfully uploaded feature_names.json")
        
        feature_stats_object_path = f"{base_dir}/feature_stats.json" if base_dir else "feature_stats.json"
        print(f"Uploading feature stats to bucket: {bucket}, object: {feature_stats_object_path}")
        minio_client.fput_object(bucket, feature_stats_object_path, feature_stats.path)
        print(f"Successfully uploaded feature_stats.json")
        
        # Return URIs for different components
        # IMPORTANT: storageUri should point to the model FILE directly
        predictor_uri = f"s3://{bucket}/{model_object_path}"  # Points to model.joblib file
        transformer_uri = f"s3://{bucket}/{base_dir}" if base_dir else f"s3://{bucket}"
        
        joblib_uri = predictor_uri
        print(f"Predictor S3 URI: {predictor_uri} (points to model.joblib file)")
        print(f"Transformer/Explainer S3 URI: {transformer_uri} (contains all artifacts)")
        
    except Exception as e:
        print(f"Error working with MinIO: {e}")
        # Continue with original URI if there's an error
        joblib_uri = convert_minio_path_to_s3_uri(artifact.uri)
    finally:
        # Clean up temporary files
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        if 'temp_joblib_path' in locals() and os.path.exists(temp_joblib_path):
            os.unlink(temp_joblib_path)

    # Use the joblib URI for the InferenceService if available
    storage_uri = joblib_uri if 'joblib_uri' in locals() else convert_minio_path_to_s3_uri(artifact.uri)
    
    isvc = kserve.V1beta1InferenceService(
        api_version=kserve.constants.KSERVE_GROUP + "/v1beta1",
        kind=kserve.constants.KSERVE_KIND_INFERENCESERVICE,
        metadata=client.V1ObjectMeta(
            name=model_name,
            namespace=kserve.utils.get_default_target_namespace(),
            annotations={
                "sidecar.istio.io/inject": "false"
            }
        ),
        spec=kserve.V1beta1InferenceServiceSpec(
            predictor=kserve.V1beta1PredictorSpec(
                service_account_name="s3-sa",
                model=kserve.V1beta1ModelSpec(
                    storage_uri=storage_uri,
                    model_format=kserve.V1beta1ModelFormat(
                        name=artifact.model_format_name, version=artifact.model_format_version
                    ),
                    runtime="kserve-sklearnserver",
                    protocol_version="v2"
                )
            )
        ),
    )
    ks_client = kserve.KServeClient()
    ks_client.create(isvc)