# setup.py
from setuptools import setup, find_packages

setup(
    name="kserve-iris",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "kfp>=1.8.0",
        "kserve>=0.9.0",
        "kubernetes>=12.0.0",
        "joblib>=1.1.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="KServe example with Iris dataset",
    keywords="kubeflow,kserve,machine learning,iris",
    python_requires=">=3.7",
)