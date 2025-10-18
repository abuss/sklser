#!/usr/bin/env python3
"""
Train sklearn pipelines and serialize them to JSON files on disk.
This script demonstrates how to save trained pipelines for later use.
Pipelines combine multiple preprocessing steps with final estimators.
"""

import json
import os
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import sklser


def create_sample_data():
    """Create sample datasets for training pipeline models."""
    print("Creating sample datasets...")

    # Classification dataset with more features for preprocessing demonstration
    X_clf, y_clf = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=20, noise=0.1, random_state=42
    )

    # Split datasets
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Save test data for later validation
    test_data = {
        "classification": {
            "X_test": X_clf_test.tolist(),
            "y_test": y_clf_test.tolist(),
        },
        "regression": {
            "X_test": X_reg_test.tolist(),
            "y_test": y_reg_test.tolist(),
        },
    }

    os.makedirs("models", exist_ok=True)
    with open("models/pipeline_test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print("✓ Test data saved to models/pipeline_test_data.json")

    return {
        "classification": (X_clf_train, X_clf_test, y_clf_train, y_clf_test),
        "regression": (X_reg_train, X_reg_test, y_reg_train, y_reg_test),
    }


def train_and_serialize_pipelines():
    """Train various pipeline models and serialize them."""

    # Create datasets
    datasets = create_sample_data()
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = datasets["classification"]
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = datasets["regression"]

    # Define pipelines to test
    pipelines = {
        # Simple preprocessing pipeline
        "simple_pipeline": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
                ]
            ),
            "data": (X_clf_train, y_clf_train),
            "type": "classification",
        },
        # Complex preprocessing pipeline
        "complex_pipeline": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=10, random_state=42)),
                    ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
                ]
            ),
            "data": (X_clf_train, y_clf_train),
            "type": "classification",
        },
        # Feature selection pipeline
        "feature_selection_pipeline": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(f_classif, k=10)),
                    ("classifier", SVC(random_state=42, probability=True)),
                ]
            ),
            "data": (X_clf_train, y_clf_train),
            "type": "classification",
        },
        # FeatureUnion pipeline
        "feature_union_pipeline": {
            "model": Pipeline(
                [
                    (
                        "features",
                        FeatureUnion(
                            [
                                ("scaled", StandardScaler()),
                                ("pca", PCA(n_components=5, random_state=42)),
                            ]
                        ),
                    ),
                    ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
                ]
            ),
            "data": (X_clf_train, y_clf_train),
            "type": "classification",
        },
        # Regression pipeline
        "regression_pipeline": {
            "model": Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    ("pca", PCA(n_components=15, random_state=42)),
                    ("regressor", LinearRegression()),
                ]
            ),
            "data": (X_reg_train, y_reg_train),
            "type": "regression",
        },
        # Neural network pipeline
        "neural_pipeline": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(f_classif, k=15)),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(50, 25), random_state=42, max_iter=500
                        ),
                    ),
                ]
            ),
            "data": (X_clf_train, y_clf_train),
            "type": "classification",
        },
    }

    results = {}
    successful_models = []
    failed_models = []

    print(f"\nTraining and serializing {len(pipelines)} pipelines...")
    print("=" * 60)

    for pipeline_name, config in pipelines.items():
        print(f"\nTraining {pipeline_name}...")

        try:
            # Train the pipeline
            start_time = time.perf_counter()
            config["model"].fit(*config["data"])
            train_time = (time.perf_counter() - start_time) * 1000

            print(f"✓ {pipeline_name} trained successfully ({train_time:.1f}ms)")

            # Serialize the pipeline
            print(f"Serializing {pipeline_name}...")
            start_time = time.perf_counter()
            serialized = sklser.serialize_json(config["model"])
            serialize_time = (time.perf_counter() - start_time) * 1000

            # Save to file
            filename = f"models/{pipeline_name}_model.json"
            with open(filename, "w") as f:
                f.write(serialized)

            print(
                f"✓ {pipeline_name} serialized to {filename} ({serialize_time:.1f}ms)"
            )

            # Test a quick prediction to ensure the model works
            if config["type"] == "classification":
                test_prediction = config["model"].predict(X_clf_test[:5])
                test_score = config["model"].score(X_clf_test, y_clf_test)
            else:
                test_prediction = config["model"].predict(X_reg_test[:5])
                test_score = config["model"].score(X_reg_test, y_reg_test)

            results[pipeline_name] = {
                "status": "success",
                "train_time_ms": train_time,
                "serialize_time_ms": serialize_time,
                "test_score": test_score,
                "file_size_kb": round(len(serialized) / 1024, 2),
                "type": config["type"],
            }

            successful_models.append(pipeline_name)
            print(f"✓ Test score: {test_score:.4f}")

        except Exception as e:
            print(f"❌ Failed to train/serialize {pipeline_name}: {str(e)}")
            results[pipeline_name] = {
                "status": "failed",
                "error": str(e),
                "type": config["type"],
            }
            failed_models.append(pipeline_name)

    # Save results summary
    with open("models/pipeline_serialization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SERIALIZATION SUMMARY")
    print("=" * 60)

    total_pipelines = len(pipelines)
    successful_count = len(successful_models)
    success_rate = (successful_count / total_pipelines) * 100

    print(f"Total pipelines: {total_pipelines}")
    print(f"Successfully serialized: {successful_count}")
    print(f"Failed: {len(failed_models)}")
    print(f"Success rate: {success_rate:.1f}%")

    if successful_models:
        print("\n✓ Successfully serialized pipelines:")
        for model in successful_models:
            score = results[model]["test_score"]
            size = results[model]["file_size_kb"]
            pipeline_type = results[model]["type"]
            print(f"  - {model} ({pipeline_type}): score={score:.4f}, size={size}KB")

    if failed_models:
        print("\n❌ Failed pipelines:")
        for model in failed_models:
            print(f"  - {model}: {results[model]['error']}")

    print("\n📁 Results saved to models/pipeline_serialization_results.json")
    print("📁 Test data saved to models/pipeline_test_data.json")
    print("📁 Model files saved to models/")


if __name__ == "__main__":
    print("============================================================")
    print("SKLEARN PIPELINE TRAINING AND SERIALIZATION")
    print("============================================================")
    train_and_serialize_pipelines()

