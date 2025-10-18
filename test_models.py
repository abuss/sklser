#!/usr/bin/env python3
"""
Comprehensive test suite for sklearn model serialization.
Tests various sklearn models to determine which ones work with the current implementation.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import sklser


def test_model(name, model, X_train, X_test, y_train, y_test, is_classifier=True):
    """Test serialization/deserialization for a single model."""
    print(f"\n=== Testing {name} ===")

    try:
        # Train the model
        model.fit(X_train, y_train)

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
    """Run tests on various sklearn models."""
    print("=" * 60)
    print("SKLEARN MODEL SERIALIZATION TEST SUITE")
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

    # Linear Models
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

    results["LinearRegression"] = test_model(
        "Linear Regression",
        LinearRegression(),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    results["LogisticRegression"] = test_model(
        "Logistic Regression",
        LogisticRegression(random_state=42),
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
    )

    results["Ridge"] = test_model(
        "Ridge Regression",
        Ridge(random_state=42),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    results["Lasso"] = test_model(
        "Lasso Regression",
        Lasso(random_state=42),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    # Support Vector Machines
    from sklearn.svm import SVC, SVR

    results["SVC"] = test_model(
        "Support Vector Classifier",
        SVC(random_state=42),
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
    )

    results["SVR"] = test_model(
        "Support Vector Regressor",
        SVR(),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    # Tree Models
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    results["DecisionTreeClassifier"] = test_model(
        "Decision Tree Classifier",
        DecisionTreeClassifier(random_state=42),
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
    )

    results["DecisionTreeRegressor"] = test_model(
        "Decision Tree Regressor",
        DecisionTreeRegressor(random_state=42),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    # Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    results["KNeighborsClassifier"] = test_model(
        "K-Neighbors Classifier",
        KNeighborsClassifier(),
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
    )

    results["KNeighborsRegressor"] = test_model(
        "K-Neighbors Regressor",
        KNeighborsRegressor(),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    # Neural Networks
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    results["MLPClassifier"] = test_model(
        "MLP Classifier",
        MLPClassifier(random_state=42, max_iter=100),
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
    )

    results["MLPRegressor"] = test_model(
        "MLP Regressor",
        MLPRegressor(random_state=42, max_iter=100),
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        is_classifier=False,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    working = [name for name, success in results.items() if success]
    failing = [name for name, success in results.items() if not success]

    print(f"✓ Working models ({len(working)}/{len(results)}):")
    for model in working:
        print(f"  - {model}")

    if failing:
        print(f"\n✗ Failing models ({len(failing)}/{len(results)}):")
        for model in failing:
            print(f"  - {model}")

    print(
        f"\nSuccess rate: {len(working)}/{len(results)} ({100 * len(working) / len(results):.1f}%)"
    )


if __name__ == "__main__":
    main()

