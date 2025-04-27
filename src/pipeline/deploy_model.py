from kfp import dsl

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kserve", "kubernetes", "model-registry", "boto3", "minio", "joblib", "scikit-learn"]
)
def deploy_model(model_name: str, model_version: str):

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
        
        # Create the new object path with .joblib extension
        # Get the directory part of the object path
        if '/' in object_path:
            # If there's a directory structure, preserve it
            object_dir = os.path.dirname(object_path)
            new_object_path = f"{object_dir}/model.joblib"
        else:
            # If it's just a filename, use model.joblib
            new_object_path = "model.joblib"
        
        # Save the model with .joblib extension to a temp file
        temp_joblib_file = tempfile.NamedTemporaryFile(suffix='.joblib', delete=False)
        temp_joblib_path = temp_joblib_file.name
        temp_joblib_file.close()
        
        print(f"Saving model to temporary joblib file: {temp_joblib_path}")
        joblib.dump(model_obj, temp_joblib_path)
        
        # Upload the .joblib file to MinIO
        print(f"Uploading to bucket: {bucket}, object: {new_object_path} from file: {temp_joblib_path}")
        minio_client.fput_object(bucket, new_object_path, temp_joblib_path)
        print(f"Successfully uploaded model.joblib to MinIO")
        
        # Create the S3 URI for the new .joblib file
        joblib_uri = f"s3://{bucket}/{new_object_path}"
        print(f"Model.joblib S3 URI: {joblib_uri}")
        
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