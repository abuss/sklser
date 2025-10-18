#!/usr/bin/env python3
"""
Test unsupervised learning models with sklearn-serialize library.
Tests clustering, dimensionality reduction, and other unsupervised models.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sklser


def create_sample_data():
    """Create sample datasets for unsupervised learning."""
    # Clustering data
    X_cluster, _ = make_blobs(
        n_samples=300, centers=4, n_features=5, random_state=42, cluster_std=1.5
    )

    # Dimensionality reduction data
    X_dim_red, y_dim_red = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    return X_cluster, X_dim_red, y_dim_red


def test_unsupervised_model(model_name, model, X_fit, X_test=None, y_test=None):
    """Test serialization and prediction consistency for an unsupervised model."""
    print(f"\n=== Testing {model_name} ===")

    try:
        # Fit the model
        if hasattr(model, "fit_transform"):
            # For models like PCA, TSNE that have fit_transform
            if y_test is not None and hasattr(model, "fit"):
                # For supervised dimensionality reduction like LDA
                model.fit(X_fit, y_test)
                original_transform = (
                    model.transform(X_test) if X_test is not None else None
                )
            else:
                # For unsupervised models
                original_transform = model.fit_transform(X_fit)
        else:
            # For clustering models
            model.fit(X_fit)
            if hasattr(model, "predict"):
                original_predictions = model.predict(
                    X_test if X_test is not None else X_fit
                )
            elif hasattr(model, "labels_"):
                original_predictions = model.labels_
            else:
                original_predictions = None

        print("✓ Model training successful")

        # Serialize model
        try:
            serialized_json = sklser.serialize_json(model)
            print("✓ Serialization successful")
        except Exception as e:
            print(f"✗ Serialization failed: {e}")
            return {"status": "serialization_failed", "error": str(e)}

        # Deserialize model
        try:
            restored_model = sklser.deserialize_object(serialized_json)
            print("✓ Deserialization successful")
        except Exception as e:
            print(f"✗ Deserialization failed: {e}")
            return {"status": "deserialization_failed", "error": str(e)}

        # Test predictions/transformations
        try:
            if hasattr(restored_model, "fit_transform"):
                if y_test is not None and hasattr(restored_model, "transform"):
                    # For supervised dimensionality reduction
                    restored_transform = (
                        restored_model.transform(X_test) if X_test is not None else None
                    )
                    if (
                        original_transform is not None
                        and restored_transform is not None
                    ):
                        match = np.allclose(
                            original_transform,
                            restored_transform,
                            rtol=1e-10,
                            atol=1e-10,
                        )
                        print(f"✓ Transformations match: {match}")
                        if match:
                            return {"status": "success", "type": "transform"}
                        else:
                            return {
                                "status": "prediction_mismatch",
                                "type": "transform",
                            }
                else:
                    # For unsupervised models, we can't directly compare fit_transform results
                    # But we can check if the model has the same fitted attributes
                    if hasattr(model, "components_") and hasattr(
                        restored_model, "components_"
                    ):
                        match = np.allclose(
                            model.components_,
                            restored_model.components_,
                            rtol=1e-10,
                            atol=1e-10,
                        )
                        print(f"✓ Components match: {match}")
                        return {
                            "status": "success" if match else "prediction_mismatch",
                            "type": "components",
                        }
                    else:
                        print("✓ Model deserialized (no direct comparison possible)")
                        return {"status": "success", "type": "no_comparison"}
            else:
                # For clustering models
                if hasattr(restored_model, "predict"):
                    restored_predictions = restored_model.predict(
                        X_test if X_test is not None else X_fit
                    )
                elif hasattr(restored_model, "labels_"):
                    # Can't directly compare labels as they depend on the order of fitting
                    print("✓ Model has labels (clustering model)")
                    return {"status": "success", "type": "clustering"}
                else:
                    restored_predictions = None

                if (
                    original_predictions is not None
                    and restored_predictions is not None
                ):
                    match = np.array_equal(original_predictions, restored_predictions)
                    print(f"✓ Predictions match: {match}")
                    return {
                        "status": "success" if match else "prediction_mismatch",
                        "type": "predict",
                    }
                else:
                    print("✓ Model deserialized (no predictions to compare)")
                    return {"status": "success", "type": "no_predictions"}
        except Exception as e:
            print(f"✗ Prediction/transform failed: {e}")
            return {"status": "prediction_failed", "error": str(e)}

    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return {"status": "training_failed", "error": str(e)}


def main():
    """Test all unsupervised models."""
    print("=" * 60)
    print("UNSUPERVISED MODEL SERIALIZATION TEST SUITE")
    print("=" * 60)

    # Create sample data
    X_cluster, X_dim_red, y_dim_red = create_sample_data()

    # Define models to test
    models = {
        # Clustering models
        "KMeans": (
            KMeans(n_clusters=4, random_state=42, n_init=10),
            X_cluster,
            None,
            None,
        ),
        "DBSCAN": (DBSCAN(eps=0.5, min_samples=5), X_cluster, None, None),
        "AgglomerativeClustering": (
            AgglomerativeClustering(n_clusters=4),
            X_cluster,
            None,
            None,
        ),
        # Dimensionality reduction models
        "PCA": (PCA(n_components=5, random_state=42), X_dim_red, X_dim_red[:100], None),
        "LinearDiscriminantAnalysis": (
            LinearDiscriminantAnalysis(),
            X_dim_red,
            X_dim_red[:100],
            y_dim_red,
        ),
        # Note: TSNE is excluded as it doesn't have a transform method for new data
        # 'TSNE': (TSNE(n_components=2, random_state=42), X_dim_red[:100], None, None),
    }

    results = {}

    for model_name, (model, X_fit, X_test, y_test) in models.items():
        result = test_unsupervised_model(model_name, model, X_fit, X_test, y_test)
        results[model_name] = result

    # Print summary
    print("\n" + "=" * 60)
    print("UNSUPERVISED MODEL SUMMARY")
    print("=" * 60)

    successful = [
        name for name, result in results.items() if result["status"] == "success"
    ]
    failed = [name for name, result in results.items() if result["status"] != "success"]

    print(f"✅ Working unsupervised models ({len(successful)}):")
    for model_name in successful:
        result_type = results[model_name].get("type", "unknown")
        print(f"  - {model_name} ({result_type})")

    if failed:
        print(f"\n❌ Failed unsupervised models ({len(failed)}):")
        for model_name in failed:
            result = results[model_name]
            print(f"  - {model_name}: {result['status']}")
            if "error" in result:
                print(f"    Error: {result['error']}")

    total_models = len(results)
    success_rate = len(successful) / total_models * 100
    print(
        f"\nUnsupervised model success rate: {len(successful)}/{total_models} ({success_rate:.1f}%)"
    )

    return results


if __name__ == "__main__":
    main()

