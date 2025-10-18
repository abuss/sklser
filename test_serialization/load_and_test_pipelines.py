#!/usr/bin/env python3
"""
Load previously serialized sklearn pipelines from JSON files and test them.
This script demonstrates how to load and use saved pipeline models.
"""

import json
import os
import sys
import time
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import sklser


def load_test_data():
    """Load the test data that was saved during training."""
    try:
        with open("models/pipeline_test_data.json", "r") as f:
            data = json.load(f)

        # Convert lists back to numpy arrays
        test_data = {
            "classification": {
                "X_test": np.array(data["classification"]["X_test"]),
                "y_test": np.array(data["classification"]["y_test"]),
            },
            "regression": {
                "X_test": np.array(data["regression"]["X_test"]),
                "y_test": np.array(data["regression"]["y_test"]),
            },
        }

        print("✓ Test data loaded successfully")
        return test_data

    except FileNotFoundError:
        print(
            "❌ Test data file not found. Please run train_and_serialize_pipelines.py first."
        )
        return None
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None


def test_pipeline(pipeline_name, model_file, test_data):
    """Test a single pipeline model."""
    print(f"\n=== Testing {model_file} ===")

    try:
        # Load JSON file
        with open(f"models/{model_file}", "r") as f:
            json_content = f.read()
        print("✓ JSON file loaded successfully")

        # Deserialize model
        start_time = time.perf_counter()
        pipeline = sklser.deserialize_object(json_content)
        deserialize_time = (time.perf_counter() - start_time) * 1000

        if pipeline is None:
            print("❌ Model deserialization returned None")
            return {
                "status": "failed",
                "error": "Deserialization returned None",
                "deserialize_time_ms": deserialize_time,
            }

        print(f"✓ Model deserialized successfully ({deserialize_time:.1f}ms)")

        # Determine test data type based on pipeline name/model type
        if "regression" in pipeline_name.lower():
            X_test = test_data["regression"]["X_test"]
            y_test = test_data["regression"]["y_test"]
            task_type = "regression"
        else:
            X_test = test_data["classification"]["X_test"]
            y_test = test_data["classification"]["y_test"]
            task_type = "classification"

        # Make predictions
        start_time = time.perf_counter()
        predictions = pipeline.predict(X_test)
        predict_time = (time.perf_counter() - start_time) * 1000
        print(f"✓ Predictions generated successfully ({predict_time:.1f}ms)")

        # Calculate metrics
        if task_type == "regression":
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            print(f"✓ MSE: {mse:.6f}, R²: {r2:.6f}")
            metric_score = r2
        else:
            accuracy = accuracy_score(y_test, predictions)
            print(f"✓ Accuracy: {accuracy:.4f}")
            metric_score = accuracy

        # Test pipeline steps (if accessible)
        try:
            if hasattr(pipeline, "named_steps"):
                step_names = list(pipeline.named_steps.keys())
                print(f"✓ Pipeline steps: {' -> '.join(step_names)}")
            elif hasattr(pipeline, "steps"):
                step_names = [name for name, _ in pipeline.steps]
                print(f"✓ Pipeline steps: {' -> '.join(step_names)}")
        except Exception as e:
            print(f"⚠ Could not access pipeline steps: {e}")

        return {
            "status": "success",
            "deserialize_time_ms": deserialize_time,
            "predict_time_ms": predict_time,
            "metric_score": metric_score,
            "task_type": task_type,
            "predictions_sample": predictions[:5].tolist(),
        }

    except FileNotFoundError:
        print(f"❌ Model file {model_file} not found")
        return {"status": "failed", "error": "File not found"}

    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return {"status": "failed", "error": str(e)}


def main(specific_model=None):
    """Main function to test pipeline models."""
    print("============================================================")
    print("SKLEARN PIPELINE DESERIALIZATION AND TESTING")
    print("============================================================")

    # Load test data
    test_data = load_test_data()
    if test_data is None:
        return

    # Find pipeline model files
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found")
        return

    # Look for pipeline model files
    pipeline_files = [
        f
        for f in os.listdir(models_dir)
        if f.endswith("_model.json") and "pipeline" in f
    ]

    if not pipeline_files:
        print(
            "❌ No pipeline model files found. Please run train_and_serialize_pipelines.py first."
        )
        return

    # Filter for specific model if requested
    if specific_model:
        pipeline_files = [f for f in pipeline_files if specific_model in f]
        if not pipeline_files:
            print(f"❌ No pipeline model files found matching '{specific_model}'")
            return

    print(f"Found {len(pipeline_files)} pipeline models to test...")

    # Test each pipeline model
    results = {}
    successful_tests = 0

    for model_file in sorted(pipeline_files):
        pipeline_name = model_file.replace("_model.json", "")
        result = test_pipeline(pipeline_name, model_file, test_data)
        results[pipeline_name] = result

        if result["status"] == "success":
            successful_tests += 1

    # Save results
    with open("models/pipeline_testing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE TESTING SUMMARY")
    print("=" * 60)

    total_tests = len(pipeline_files)
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

    print(f"Total pipelines tested: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests: {total_tests - successful_tests}")
    print(f"Testing success rate: {success_rate:.1f}%")

    # Show successful tests
    successful_pipelines = [
        name for name, result in results.items() if result["status"] == "success"
    ]

    if successful_pipelines:
        print("\n✓ Successfully tested pipelines:")
        for pipeline_name in successful_pipelines:
            result = results[pipeline_name]
            score = result["metric_score"]
            task_type = result["task_type"]
            metric_name = "R²" if task_type == "regression" else "Accuracy"
            deserialize_time = result["deserialize_time_ms"]
            predict_time = result["predict_time_ms"]
            print(
                f"  - {pipeline_name}: {metric_name}={score:.4f}, "
                f"deserialize={deserialize_time:.1f}ms, predict={predict_time:.1f}ms"
            )

    # Show failed tests
    failed_pipelines = [
        name for name, result in results.items() if result["status"] == "failed"
    ]

    if failed_pipelines:
        print("\n❌ Failed pipeline tests:")
        for pipeline_name in failed_pipelines:
            error = results[pipeline_name]["error"]
            print(f"  - {pipeline_name}: {error}")

    print("\n📁 Results saved to models/pipeline_testing_results.json")

    if success_rate == 100.0:
        print("🎉 All pipeline tests passed!")
    elif success_rate >= 80.0:
        print("👍 Most pipeline tests passed!")
    else:
        print("⚠️ Many pipeline tests failed. Check the errors above.")


if __name__ == "__main__":
    # Allow testing specific model via command line argument
    specific_model = sys.argv[1] if len(sys.argv) > 1 else None
    main(specific_model)

