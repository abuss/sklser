#!/usr/bin/env python3
"""
Test script for sklearn-serialize Pipeline support.
"""

import time
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sklser import serialize_json, deserialize_object


def test_simple_pipeline():
    """Test a simple pipeline with scaler + classifier."""
    print("Testing Simple Pipeline (StandardScaler + LogisticRegression)...")
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and fit pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    original_score = pipeline.score(X_test, y_test)
    original_prediction = pipeline.predict(X_test[:5])
    
    # Serialize
    start_time = time.perf_counter()
    serialized = serialize_json(pipeline)
    serialize_time = (time.perf_counter() - start_time) * 1000
    
    # Deserialize
    start_time = time.perf_counter()
    deserialized_pipeline = deserialize_object(serialized)
    deserialize_time = (time.perf_counter() - start_time) * 1000
    
    if deserialized_pipeline is None:
        print("❌ FAILED: Could not deserialize pipeline")
        return False
    
    # Test deserialized pipeline
    try:
        deserialized_score = deserialized_pipeline.score(X_test, y_test)
        deserialized_prediction = deserialized_pipeline.predict(X_test[:5])
        
        score_match = abs(original_score - deserialized_score) < 1e-10
        prediction_match = np.array_equal(original_prediction, deserialized_prediction)
        
        if score_match and prediction_match:
            print(f"✅ SUCCESS: Simple Pipeline works perfectly")
            print(f"   Serialization time: {serialize_time:.3f}ms")
            print(f"   Deserialization time: {deserialize_time:.3f}ms")
            print(f"   Score: {original_score:.6f} -> {deserialized_score:.6f}")
            return True
        else:
            print(f"❌ FAILED: Score or prediction mismatch")
            print(f"   Score: {original_score:.6f} -> {deserialized_score:.6f} (match: {score_match})")
            print(f"   Prediction match: {prediction_match}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Error during testing: {e}")
        return False


def test_complex_pipeline():
    """Test a complex pipeline with multiple preprocessing steps."""
    print("\nTesting Complex Pipeline (StandardScaler + PCA + LogisticRegression)...")
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and fit complex pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5, random_state=42)),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    original_score = pipeline.score(X_test, y_test)
    original_prediction = pipeline.predict(X_test[:5])
    
    # Serialize
    start_time = time.perf_counter()
    serialized = serialize_json(pipeline)
    serialize_time = (time.perf_counter() - start_time) * 1000
    
    # Deserialize
    start_time = time.perf_counter()
    deserialized_pipeline = deserialize_object(serialized)
    deserialize_time = (time.perf_counter() - start_time) * 1000
    
    if deserialized_pipeline is None:
        print("❌ FAILED: Could not deserialize complex pipeline")
        return False
    
    # Test deserialized pipeline
    try:
        deserialized_score = deserialized_pipeline.score(X_test, y_test)
        deserialized_prediction = deserialized_pipeline.predict(X_test[:5])
        
        score_match = abs(original_score - deserialized_score) < 1e-10
        prediction_match = np.array_equal(original_prediction, deserialized_prediction)
        
        if score_match and prediction_match:
            print(f"✅ SUCCESS: Complex Pipeline works perfectly")
            print(f"   Serialization time: {serialize_time:.3f}ms")
            print(f"   Deserialization time: {deserialize_time:.3f}ms")
            print(f"   Score: {original_score:.6f} -> {deserialized_score:.6f}")
            return True
        else:
            print(f"❌ FAILED: Score or prediction mismatch")
            print(f"   Score: {original_score:.6f} -> {deserialized_score:.6f} (match: {score_match})")
            print(f"   Prediction match: {prediction_match}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Error during testing: {e}")
        return False


def test_feature_union_pipeline():
    """Test a pipeline with FeatureUnion."""
    print("\nTesting FeatureUnion Pipeline...")
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create pipeline with FeatureUnion
    feature_union = FeatureUnion([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3, random_state=42))
    ])
    
    pipeline = Pipeline([
        ('features', feature_union),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    original_score = pipeline.score(X_test, y_test)
    original_prediction = pipeline.predict(X_test[:5])
    
    # Serialize
    start_time = time.perf_counter()
    serialized = serialize_json(pipeline)
    serialize_time = (time.perf_counter() - start_time) * 1000
    
    # Deserialize
    start_time = time.perf_counter()
    deserialized_pipeline = deserialize_object(serialized)
    deserialize_time = (time.perf_counter() - start_time) * 1000
    
    if deserialized_pipeline is None:
        print("❌ FAILED: Could not deserialize FeatureUnion pipeline")
        return False
    
    # Test deserialized pipeline
    try:
        deserialized_score = deserialized_pipeline.score(X_test, y_test)
        deserialized_prediction = deserialized_pipeline.predict(X_test[:5])
        
        score_match = abs(original_score - deserialized_score) < 1e-10
        prediction_match = np.array_equal(original_prediction, deserialized_prediction)
        
        if score_match and prediction_match:
            print(f"✅ SUCCESS: FeatureUnion Pipeline works perfectly")
            print(f"   Serialization time: {serialize_time:.3f}ms")
            print(f"   Deserialization time: {deserialize_time:.3f}ms")
            print(f"   Score: {original_score:.6f} -> {deserialized_score:.6f}")
            return True
        else:
            print(f"❌ FAILED: Score or prediction mismatch")
            print(f"   Score: {original_score:.6f} -> {deserialized_score:.6f} (match: {score_match})")
            print(f"   Prediction match: {prediction_match}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Error during testing: {e}")
        return False


def main():
    """Run all pipeline tests."""
    print("🧪 Testing sklearn-serialize Pipeline Support")
    print("=" * 50)
    
    results = []
    
    # Test simple pipeline
    results.append(test_simple_pipeline())
    
    # Test complex pipeline
    results.append(test_complex_pipeline())
    
    # Test FeatureUnion pipeline
    results.append(test_feature_union_pipeline())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 PIPELINE TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate == 100.0:
        print("🎉 All pipeline tests passed! Pipeline support is working perfectly.")
    else:
        print("⚠️  Some pipeline tests failed. Pipeline support needs debugging.")
    
    return success_rate == 100.0


if __name__ == "__main__":
    main()