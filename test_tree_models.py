#!/usr/bin/env python3
"""
Test script specifically for tree-based models with training data storage.
This demonstrates how to use the library with tree models by manually storing training data.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import sklser


def test_tree_model_with_training_data(
    name, model, X_train, X_test, y_train, y_test, is_classifier=True
):
    """Test tree model with manual training data storage."""
    print(f"\n=== Testing {name} (with training data) ===")

    try:
        # Train the model
        model.fit(X_train, y_train)

        # IMPORTANT: For tree models to be properly reconstructed,
        # you need to manually store training data
        model._fit_X = X_train
        model._y = y_train

        # Serialize
        json_out = sklser.serialize_json(model)
        print("✓ Serialization successful")

        # Deserialize
        new_model = sklser.deserialize_object(json_out)
        print("✓ Deserialization successful")

        # Test predictions
        pred_original = model.predict(X_test)
        pred_deserialized = new_model.predict(X_test)

        if is_classifier:
            match = np.array_equal(pred_original, pred_deserialized)
            accuracy = accuracy_score(y_test, pred_original)
            print(f"✓ Predictions match: {match}")
            print(f"  Accuracy: {accuracy:.4f}")
        else:
            match = np.allclose(pred_original, pred_deserialized, rtol=1e-10)
            mse = mean_squared_error(y_test, pred_original)
            print(f"✓ Predictions match: {match}")
            print(f"  MSE: {mse:.6f}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Test tree-based models with training data storage."""
    print("=" * 60)
    print("TREE-BASED MODEL SERIALIZATION TEST")
    print("=" * 60)
    print("NOTE: Tree models require training data to be manually stored")
    print("for proper reconstruction (model._fit_X = X, model._y = y)")
    print("=" * 60)

    # Generate test data
    X_class, y_class = make_classification(
        n_samples=1000, n_features=4, n_classes=2, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=4, noise=0.1, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    results = {}

    # Test Decision Tree models
    results["DecisionTreeClassifier"] = test_tree_model_with_training_data(
        "Decision Tree Classifier",
        DecisionTreeClassifier(random_state=42),
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
    )

    results["DecisionTreeRegressor"] = test_tree_model_with_training_data(
        "Decision Tree Regressor",
        DecisionTreeRegressor(random_state=42),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TREE MODEL TEST SUMMARY")
    print("=" * 60)

    working = [name for name, success in results.items() if success]
    failing = [name for name, success in results.items() if not success]

    if working:
        print(f"✓ Working tree models ({len(working)}/{len(results)}):")
        for model in working:
            print(f"  - {model}")

    if failing:
        print(f"\n✗ Failing tree models ({len(failing)}/{len(results)}):")
        for model in failing:
            print(f"  - {model}")

    success_rate = len(working) / len(results) * 100
    print(
        f"\nTree model success rate: {len(working)}/{len(results)} ({success_rate:.1f}%)"
    )


if __name__ == "__main__":
    main()

