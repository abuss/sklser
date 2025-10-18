# sklearn-serialize (sklser)

A Python library for serializing and deserializing scikit-learn models to/from JSON format.

## Features

- ✅ **JSON serialization**: Convert sklearn models to human-readable JSON
- ✅ **Model reconstruction**: Deserialize JSON back to functional sklearn models  
- ✅ **Numpy support**: Proper handling of numpy arrays and scalar types
- ✅ **Type preservation**: Maintains data types during serialization/deserialization
- ✅ **Prediction consistency**: Deserialized models produce identical predictions
- ✅ **Complex object handling**: Support for sklearn internal objects (LabelBinarizer, Tree structures)
- ✅ **Training data preservation**: Automatic rebuilding of internal structures for models like KNeighbors
- ✅ **High compatibility**: 88.2% success rate across supervised and unsupervised sklearn models

## Installation

```bash
uv add sklser
# or
pip install sklser
```

## Quick Start

```python
import sklser
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Create and train a model
X, y = make_regression(n_samples=100, n_features=3, random_state=42)
model = LinearRegression()
model.fit(X, y)

# Serialize to JSON
json_string = sklser.serialize_json(model)

# Deserialize back to model
restored_model = sklser.deserialize_object(json_string)

# Both models produce identical predictions
assert np.array_equal(model.predict(X), restored_model.predict(X))
```

## Supported Models

**Current success rate: 15/17 models (88.2%)** - Covers both supervised and unsupervised learning!

### ✅ Fully Supported Supervised Models
All these models serialize/deserialize with **perfect prediction preservation**:

#### Linear Models
- ✅ **LinearRegression** - Complete coefficient and intercept preservation
- ✅ **LogisticRegression** - Full parameter serialization including class handling
- ✅ **Ridge** - Ridge regression with regularization parameters
- ✅ **Lasso** - Lasso regression with L1 regularization

#### Support Vector Models
- ✅ **SVC** - Support Vector Classifier with all parameters
- ✅ **SVR** - Support Vector Regressor with kernel support

#### Neighbor Models
- ✅ **KNeighborsClassifier** - K-nearest neighbors classifier
- ✅ **KNeighborsRegressor** - K-nearest neighbors regressor

#### Neural Networks
- ✅ **MLPClassifier** - Multi-layer perceptron classifier
- ✅ **MLPRegressor** - Multi-layer perceptron regressor

#### Discriminant Analysis
- ✅ **LinearDiscriminantAnalysis** - Linear discriminant analysis for classification and dimensionality reduction

### ✅ Fully Supported Unsupervised Models
All unsupervised models work perfectly with **complete state preservation**:

#### Clustering Models
- ✅ **KMeans** - K-means clustering with centroids and labels
- ✅ **DBSCAN** - Density-based clustering with noise detection
- ✅ **AgglomerativeClustering** - Hierarchical clustering

#### Dimensionality Reduction
- ✅ **PCA** - Principal Component Analysis with components and explained variance

### ❌ Currently Unsupported (Supervised Models)
- **DecisionTreeClassifier/Regressor** - Complex tree structure serialization issues

### 🔧 Recent Improvements
- **Added unsupervised learning support** - Now supports clustering and dimensionality reduction models
- **Fixed LinearDiscriminantAnalysis** - Added missing internal attributes for complete serialization
- **Fixed SVC/SVR models** - Added sklearn internal attributes serialization
- **Fixed MLP models** - Enhanced list serialization for numpy arrays
- **Fixed KNeighbors models** - Implemented training data preservation and tree rebuilding
- **Fixed MLPClassifier** - Added LabelBinarizer object serialization support

## Usage Examples

### Basic Serialization

```python
import sklser
from sklearn.linear_model import LogisticRegression

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Serialize
json_data = sklser.serialize_json(model)
print(json_data)  # Human-readable JSON

# Save to file
with open('model.json', 'w') as f:
    f.write(json_data)
```

### Deserialization

```python
# Load from file
with open('model.json', 'r') as f:
    json_data = f.read()

# Deserialize
model = sklser.deserialize_object(json_data)

# Use restored model
predictions = model.predict(X_test)
```

### Unsupervised Model Serialization

```python
import sklser
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Create clustering data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Train and serialize a clustering model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Serialize the clustering model
json_data = sklser.serialize_json(kmeans)

# Deserialize and verify
restored_kmeans = sklser.deserialize_object(json_data)
assert hasattr(restored_kmeans, 'labels_')  # Labels preserved

# Train and serialize a dimensionality reduction model  
pca = PCA(n_components=2)
pca.fit(X)

# Serialize the PCA model
pca_json = sklser.serialize_json(pca)

# Deserialize and verify transformation consistency
restored_pca = sklser.deserialize_object(pca_json)
original_transform = pca.transform(X[:5])
restored_transform = restored_pca.transform(X[:5])
assert np.allclose(original_transform, restored_transform)
```

### Testing Model Compatibility

Run the comprehensive test suite to check which models work in your environment:

```bash
# Test all supported models
python test_models.py

# Test real serialization to disk workflow
cd test_serialization
python train_and_serialize.py  # Train and save models to JSON files
python load_and_predict.py     # Load models and test predictions
```

## API Reference

### `serialize_json(model) -> str`

Serialize a trained sklearn model to JSON string.

**Parameters:**
- `model`: A fitted sklearn estimator

**Returns:**
- `str`: JSON string representation of the model

**Example:**
```python
json_str = sklser.serialize_json(trained_model)
```

### `deserialize_object(json_str) -> object`

Deserialize a JSON string back to a sklearn model.

**Parameters:**
- `json_str`: JSON string created by `serialize_json()`

**Returns:**
- `object`: Reconstructed sklearn estimator

**Example:**
```python
model = sklser.deserialize_object(json_str)
```

### `show_members(obj)`

Debug utility to inspect model attributes (useful for development).

**Parameters:**
- `obj`: Any Python object to inspect

## JSON Format

The serialization format includes:

```json
{
    "__class__": "LinearRegression",
    "__module__": "sklearn.linear_model._base",
    "members": {
        "coef_": {
            "value": [1.5, 2.3, -0.8],
            "type": "numpy.ndarray",
            "dtype": "float64",
            "shape": [3]
        },
        "intercept_": {
            "value": 0.5,
            "type": "<class 'numpy.float64'>",
            "numpy_type": true
        }
    }
}
```

## Limitations

1. **Decision Trees**: Complex tree structures in DecisionTree models are not fully supported due to sklearn's internal tree representation
2. **Custom objects**: Only standard sklearn models and numpy types are supported  
3. **Version compatibility**: Serialized models may not work across different sklearn versions
4. **Memory usage**: JSON format is less memory-efficient than binary formats like pickle

## What's Fixed

Recent improvements have resolved many previous limitations:
- ✅ **Unsupervised learning support** - Added clustering and dimensionality reduction models
- ✅ **LinearDiscriminantAnalysis** - Fixed missing internal attributes for complete serialization
- ✅ **Training data preservation** - KNeighbors models now rebuild internal structures
- ✅ **Complex sklearn objects** - LabelBinarizer and other internal objects are supported
- ✅ **SVM internal attributes** - Support vector models now serialize completely
- ✅ **Neural network weights** - MLP models with complex weight structures work perfectly

## Development

### Running Tests

```bash
# Run comprehensive model tests
uv run python test_models.py

# Test specific models
uv run python models/linear_regression.py
```

### Project Structure

```
sklearn-serialize/
├── src/sklser/              # Main library code
│   ├── __init__.py          # Core serialization functions
│   └── py.typed             # Type hint marker
├── test_serialization/      # Real-world serialization tests
│   ├── train_and_serialize.py  # Train and save models to JSON
│   ├── load_and_predict.py     # Load and test saved models
│   └── README.md            # Test suite documentation
├── models/                  # Example model scripts
├── test_models.py           # Comprehensive test suite
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## Contributing

Contributions are welcome! Areas for improvement:

1. **Support for ensemble models** (RandomForest, GradientBoosting)
2. **Preprocessing objects** (StandardScaler, LabelEncoder)
3. **Binary serialization** for better performance
4. **Cross-version compatibility** handling

## License

This project is licensed under the MIT License.