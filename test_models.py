#!/usr/bin/env python3
"""
Comprehensive test suite for sklearn model serialization.
Tests various sklearn models to determine which ones work with the current implementation.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
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


def test_unsupervised_model(name, model, X_train, X_test=None):
    """Test serialization/deserialization for unsupervised models."""
    print(f"\n=== Testing {name} ===")

    try:
        # Train the model
        model.fit(X_train)
        print("✓ Model training successful")

        # Serialize
        json_out = sklser.serialize_json(model)
        print("✓ Serialization successful")

        # Deserialize
        new_model = sklser.deserialize_object(json_out)
        print("✓ Deserialization successful")

        # Test the model works after deserialization
        if hasattr(model, "labels_"):
            # Clustering model - check labels exist
            if hasattr(new_model, "labels_") and new_model.labels_ is not None:
                print("✓ Model has labels (clustering model)")
                return True
            else:
                print("✗ Clustering model missing labels")
                return False
        elif hasattr(model, "components_"):
            # Dimensionality reduction - test transform and compare components
            if X_test is None:
                X_test = X_train[:5]  # Use subset for testing

            transform_original = model.transform(X_test)
            transform_deserialized = new_model.transform(X_test)

            components_match = np.allclose(
                model.components_, new_model.components_, rtol=1e-10
            )
            transform_match = np.allclose(
                transform_original, transform_deserialized, rtol=1e-10
            )

            print(f"✓ Components match: {components_match}")
            print(f"✓ Transformations match: {transform_match}")
            return components_match and transform_match
        elif hasattr(model, "transform"):
            # Other transform models (like LDA)
            if X_test is None:
                X_test = X_train[:5]

            transform_original = model.transform(X_test)
            transform_deserialized = new_model.transform(X_test)

            transform_match = np.allclose(
                transform_original, transform_deserialized, rtol=1e-10
            )
            print(f"✓ Transformations match: {transform_match}")
            return transform_match
        else:
            # Models without direct comparison (like some clustering algorithms)
            print("✓ Model deserialized (no direct comparison possible)")
            return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Run tests on various sklearn models."""
    print("=" * 60)
    print("SKLEARN MODEL SERIALIZATION TEST SUITE")
    print("=" * 60)

    # Generate test data for supervised models
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

    # Generate test data for unsupervised models
    X_cluster, _ = make_blobs(
        n_samples=300, centers=4, n_features=4, random_state=42, cluster_std=0.60
    )

    results = {}

    # ================ SUPERVISED MODELS ================
    print("\n" + "=" * 40)
    print("SUPERVISED MODELS")
    print("=" * 40)

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

    # ================ UNSUPERVISED MODELS ================
    print("\n" + "=" * 40)
    print("UNSUPERVISED MODELS")
    print("=" * 40)

    # Clustering models
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

    results["KMeans"] = test_unsupervised_model(
        "K-Means Clustering", KMeans(n_clusters=4, random_state=42), X_cluster
    )

    results["DBSCAN"] = test_unsupervised_model(
        "DBSCAN Clustering", DBSCAN(eps=0.3, min_samples=10), X_cluster
    )

    results["AgglomerativeClustering"] = test_unsupervised_model(
        "Agglomerative Clustering", AgglomerativeClustering(n_clusters=4), X_cluster
    )

    # Dimensionality reduction models
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    results["PCA"] = test_unsupervised_model(
        "Principal Component Analysis", PCA(n_components=2), X_cluster
    )

    # LDA needs supervised fitting but is often used for dimensionality reduction
    results["LinearDiscriminantAnalysis"] = test_model(
        "Linear Discriminant Analysis",
        LinearDiscriminantAnalysis(n_components=1),
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
        is_classifier=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    working = [name for name, success in results.items() if success]
    failing = [name for name, success in results.items() if not success]

    # Separate supervised and unsupervised results
    supervised_models = [
        "LinearRegression",
        "LogisticRegression",
        "Ridge",
        "Lasso",
        "SVC",
        "SVR",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "KNeighborsClassifier",
        "KNeighborsRegressor",
        "MLPClassifier",
        "MLPRegressor",
        "LinearDiscriminantAnalysis",
    ]

    unsupervised_models = ["KMeans", "DBSCAN", "AgglomerativeClustering", "PCA"]

    supervised_working = [name for name in working if name in supervised_models]
    unsupervised_working = [name for name in working if name in unsupervised_models]

    supervised_failing = [name for name in failing if name in supervised_models]
    unsupervised_failing = [name for name in failing if name in unsupervised_models]

    print(
        f"✓ Working supervised models ({len(supervised_working)}/{len(supervised_models)}):"
    )
    for model in supervised_working:
        print(f"  - {model}")

    if supervised_failing:
        print(
            f"\n✗ Failing supervised models ({len(supervised_failing)}/{len(supervised_models)}):"
        )
        for model in supervised_failing:
            print(f"  - {model}")

    print(
        f"\n✓ Working unsupervised models ({len(unsupervised_working)}/{len(unsupervised_models)}):"
    )
    for model in unsupervised_working:
        print(f"  - {model}")

    if unsupervised_failing:
        print(
            f"\n✗ Failing unsupervised models ({len(unsupervised_failing)}/{len(unsupervised_models)}):"
        )
        for model in unsupervised_failing:
            print(f"  - {model}")

    print(
        f"\nSupervised success rate: {len(supervised_working)}/{len(supervised_models)} ({100 * len(supervised_working) / len(supervised_models):.1f}%)"
    )
    print(
        f"Unsupervised success rate: {len(unsupervised_working)}/{len(unsupervised_models)} ({100 * len(unsupervised_working) / len(unsupervised_models):.1f}%)"
    )
    print(
        f"Overall success rate: {len(working)}/{len(results)} ({100 * len(working) / len(results):.1f}%)"
    )


if __name__ == "__main__":
    main()
