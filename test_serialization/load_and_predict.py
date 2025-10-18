#!/usr/bin/env python3
"""
Load serialized sklearn models from JSON files and test predictions.
This script demonstrates how to load and use previously saved models.
"""

import numpy as np
import json
from sklearn.metrics import accuracy_score, mean_squared_error
import sklser


def load_test_data():
    """Load test data from file."""
    try:
        with open("test_data.json", "r") as f:
            test_data = json.load(f)

        # Convert back to numpy arrays
        X_clf_test = np.array(test_data["classification"]["X_test"])
        y_clf_test = np.array(test_data["classification"]["y_test"])
        X_reg_test = np.array(test_data["regression"]["X_test"])
        y_reg_test = np.array(test_data["regression"]["y_test"])

        print("✓ Test data loaded successfully")
        return {
            "classification": (X_clf_test, y_clf_test),
            "regression": (X_reg_test, y_reg_test),
        }
    except FileNotFoundError:
        print("✗ test_data.json not found. Run train_and_serialize.py first.")
        return None
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return None


def load_serialization_results():
    """Load the serialization results to know which models were successfully saved."""
    try:
        with open("serialization_results.json", "r") as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(
            "✗ serialization_results.json not found. Run train_and_serialize.py first."
        )
        return None


def load_and_test_model(model_filename, model_info, test_datasets):
    """Load a single model from JSON and test its predictions."""
    print(f"\n=== Testing {model_filename} ===")

    try:
        # Load serialized model
        with open(model_filename, "r") as f:
            serialized_model = f.read()

        print("✓ JSON file loaded successfully")

        # Deserialize model
        model = sklser.deserialize_object(serialized_model)
        print("✓ Model deserialized successfully")

        # Get appropriate test data
        model_type = model_info["model_type"]
        if model_type == "classification":
            X_test, y_test = test_datasets["classification"]
        else:
            X_test, y_test = test_datasets["regression"]

        # Make predictions
        predictions = model.predict(X_test)
        print("✓ Predictions generated successfully")

        # Calculate metrics
        if model_type == "classification":
            accuracy = accuracy_score(y_test, predictions)
            print(f"✓ Accuracy: {accuracy:.4f}")
            return {"status": "success", "metric": "accuracy", "value": accuracy}
        else:
            mse = mean_squared_error(y_test, predictions)
            print(f"✓ MSE: {mse:.4f}")
            return {"status": "success", "metric": "mse", "value": mse}

    except FileNotFoundError:
        print(f"✗ Model file {model_filename} not found")
        return {"status": "file_not_found", "error": f"File {model_filename} not found"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": "error", "error": str(e)}


def test_all_models():
    """Test all successfully serialized models."""
    print("=" * 60)
    print("SKLEARN MODEL DESERIALIZATION AND TESTING")
    print("=" * 60)

    # Load test data
    test_datasets = load_test_data()
    if test_datasets is None:
        return

    # Load serialization results
    serialization_results = load_serialization_results()
    if serialization_results is None:
        return

    # Test each successfully serialized model
    test_results = {}
    successful_models = {
        name: info
        for name, info in serialization_results.items()
        if info["status"] == "success"
    }

    if not successful_models:
        print("No successfully serialized models found to test.")
        return

    print(f"Found {len(successful_models)} models to test...")

    for model_name, model_info in successful_models.items():
        result = load_and_test_model(model_info["filename"], model_info, test_datasets)
        test_results[model_name] = result

    # Generate summary
    print("\n" + "=" * 60)
    print("TESTING SUMMARY")
    print("=" * 60)

    successful_tests = [
        name for name, result in test_results.items() if result["status"] == "success"
    ]
    failed_tests = [
        name for name, result in test_results.items() if result["status"] != "success"
    ]

    print(f"✓ Successfully tested models ({len(successful_tests)}):")
    for model_name in successful_tests:
        result = test_results[model_name]
        metric_name = result["metric"]
        metric_value = result["value"]
        print(f"  - {model_name}: {metric_name} = {metric_value:.4f}")

    if failed_tests:
        print(f"\n✗ Failed to test models ({len(failed_tests)}):")
        for model_name in failed_tests:
            result = test_results[model_name]
            print(f"  - {model_name}: {result.get('error', 'Unknown error')}")

    print(
        f"\nTesting success rate: {len(successful_tests)}/{len(test_results)} ({len(successful_tests) / len(test_results) * 100:.1f}%)"
    )

    # Save test results
    with open("testing_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("Test results saved to testing_results.json")

    return test_results


def test_single_model(model_name):
    """Test a single specific model by name."""
    print(f"Testing single model: {model_name}")

    # Load test data
    test_datasets = load_test_data()
    if test_datasets is None:
        return

    # Load serialization results
    serialization_results = load_serialization_results()
    if serialization_results is None:
        return

    # Check if model exists
    if model_name not in serialization_results:
        print(f"Model '{model_name}' not found in serialization results.")
        available_models = list(serialization_results.keys())
        print(f"Available models: {', '.join(available_models)}")
        return

    model_info = serialization_results[model_name]
    if model_info["status"] != "success":
        print(f"Model '{model_name}' was not successfully serialized.")
        return

    # Test the model
    result = load_and_test_model(model_info["filename"], model_info, test_datasets)
    return result


def main():
    """Main function - can test all models or a specific one."""
    import sys

    if len(sys.argv) > 1:
        # Test specific model
        model_name = sys.argv[1]
        test_single_model(model_name)
    else:
        # Test all models
        test_all_models()


if __name__ == "__main__":
    main()

