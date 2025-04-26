
from kfp import dsl


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kserve", "kubernetes" , "model-registry"]
)
def deploy_model(model_name: str, model_version: str):

    import kserve
    from model_registry import ModelRegistry
    from kubernetes import client

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

    isvc = kserve.V1beta1InferenceService(
        api_version=kserve.constants.KSERVE_GROUP + "/v1beta1",
        kind=kserve.constants.KSERVE_KIND_INFERENCESERVICE,
        metadata=client.V1ObjectMeta(
            name="fraud-detection-model",
            namespace=kserve.utils.get_default_target_namespace(),
            labels={
                "modelregistry/registered-model-id": model.id,
                "modelregistry/model-version-id": version.id,
            },
        ),
        spec=kserve.V1beta1InferenceServiceSpec(
            predictor=kserve.V1beta1PredictorSpec(
                service_account_name="s3-sa",
                model=kserve.V1beta1ModelSpec(
                    storage_uri=convert_minio_path_to_s3_uri(artifact.uri),
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

