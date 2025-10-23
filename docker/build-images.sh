#!/bin/bash

# Build script for KServe custom images
# This script builds Docker images and loads them into kind cluster

set -e

# Configuration
DOCKER_HUB_REPO="${DOCKER_HUB_REPO:-prashanthchaitanya715/kubeflowbook}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-latest}"
PUSH_TO_DOCKER_HUB="${PUSH_TO_DOCKER_HUB:-true}"

echo "================================================"
echo "Building KServe Custom Images"
echo "================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Docker Hub Repo: $DOCKER_HUB_REPO"
echo "Version: $VERSION"
echo "Push to Docker Hub: $PUSH_TO_DOCKER_HUB"
echo ""

# Function to build and load image
build_and_load() {
    local component=$1
    local dockerfile=$2
    local image_name="fraud-detection-${component}"
    local local_image="${image_name}:${VERSION}"
    local dockerhub_image="${DOCKER_HUB_REPO}:${component}-${VERSION}"
    
    echo "Building ${local_image}..."
    
    docker build \
        -f "${PROJECT_ROOT}/docker/${dockerfile}" \
        -t "${local_image}" \
        -t "${dockerhub_image}" \
        "${PROJECT_ROOT}"
    
    if [ "$PUSH_TO_DOCKER_HUB" = "true" ]; then
        echo "Pushing ${dockerhub_image} to Docker Hub..."
        docker push "${dockerhub_image}"
        echo "✓ Pushed to Docker Hub"
    fi
    
    echo "Loading ${local_image} into kind cluster..."
    kind load docker-image "${local_image}" --name kubeflow
    
    echo "✓ ${local_image} built and loaded"
    echo ""
}

# Build all images
# Note: Predictor uses built-in kserve-sklearnserver runtime, no custom image needed

echo "Building Transformer..."
build_and_load "transformer" "Dockerfile.transformer"

echo "Building Explainer..."
build_and_load "explainer" "Dockerfile.explainer"

echo "Building Aggregator..."
build_and_load "aggregator" "Dockerfile.aggregator"

echo "================================================"
echo "All images built and loaded successfully!"
echo "================================================"
echo ""
echo "Local images:"
docker images | grep fraud-detection

if [ "$PUSH_TO_DOCKER_HUB" = "true" ]; then
    echo ""
    echo "Docker Hub images pushed:"
    echo "  - ${DOCKER_HUB_REPO}:transformer-${VERSION}"
    echo "  - ${DOCKER_HUB_REPO}:explainer-${VERSION}"
    echo "  - ${DOCKER_HUB_REPO}:aggregator-${VERSION}"
fi

echo ""
echo "Images in kind cluster:"
docker exec -it kubeflow-control-plane crictl images | grep fraud-detection || echo "Note: Images may take a moment to appear in crictl"
