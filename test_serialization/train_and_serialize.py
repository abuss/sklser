#!/usr/bin/env python3
"""
Train sklearn models and serialize them to JSON files on disk.
This script demonstrates how to save trained models for later use.
"""

import json
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklser
import os


def create_sample_data():
    """Create sample datasets for training models."""
    print("Creating sample datasets...")

    # Classification dataset
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
        "regression": {"X_test": X_reg_test.tolist(), "y_test": y_reg_test.tolist()},
    }

    with open("test_data.json", "w") as f:
        json.dump(test_data, f)

    print("✓ Test data saved to test_data.json")

    return {
        "classification": (X_clf_train, y_clf_train),
        "regression": (X_reg_train, y_reg_train),
    }


def train_and_serialize_models(datasets):
    """Train multiple models and serialize them to JSON files."""

    X_clf_train, y_clf_train = datasets["classification"]
    X_reg_train, y_reg_train = datasets["regression"]

    # Define models to train
    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "linear_regression": LinearRegression(),
        "svc": SVC(probability=True, random_state=42),
        "knn_classifier": KNeighborsClassifier(n_neighbors=5),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(50,), max_iter=500, random_state=42
        ),
        "random_forest": RandomForestClassifier(n_estimators=10, random_state=42),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        try:
            # Train model on appropriate dataset
            if model_name in ["linear_regression"]:
                model.fit(X_reg_train, y_reg_train)
                model_type = "regression"
            else:
                model.fit(X_clf_train, y_clf_train)
                model_type = "classification"

            print(f"✓ {model_name} trained successfully")

            # Serialize model
            print(f"Serializing {model_name}...")
            serialized_json = sklser.serialize_json(model)

            # Save to file
            filename = f"{model_name}_model.json"
            with open(filename, "w") as f:
                f.write(serialized_json)

            print(f"✓ {model_name} serialized to {filename}")

            results[model_name] = {
                "filename": filename,
                "model_type": model_type,
                "status": "success",
            }

        except Exception as e:
            print(f"✗ Failed to train/serialize {model_name}: {e}")
            results[model_name] = {
                "filename": None,
                "model_type": None,
                "status": "failed",
                "error": str(e),
            }

    # Save results summary
    with open("serialization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Serialization results saved to serialization_results.json")

    return results


def main():
    """Main function to run the training and serialization process."""
    print("=" * 60)
    print("SKLEARN MODEL TRAINING AND SERIALIZATION")
    print("=" * 60)

    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    os.chdir("models")

    # Create datasets
    datasets = create_sample_data()

    # Train and serialize models
    results = train_and_serialize_models(datasets)

    # Print summary
    print("\n" + "=" * 60)
    print("SERIALIZATION SUMMARY")
    print("=" * 60)

    successful = [name for name, info in results.items() if info["status"] == "success"]
    failed = [name for name, info in results.items() if info["status"] == "failed"]

    print(f"✓ Successfully serialized models ({len(successful)}):")
    for model_name in successful:
        print(f"  - {model_name}")

    if failed:
        print(f"\n✗ Failed to serialize models ({len(failed)}):")
        for model_name in failed:
            print(f"  - {model_name}")

    print(
        f"\nSuccess rate: {len(successful)}/{len(results)} ({len(successful) / len(results) * 100:.1f}%)"
    )
    print(f"\nModel files saved in: {os.getcwd()}")


if __name__ == "__main__":
    main()

