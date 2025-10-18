# sklearn-serialize (sklser)

A Python library for serializing and deserializing scikit-learn models to/from JSON format.

## Features

- ✅ **JSON serialization**: Convert sklearn models to human-readable JSON
- ✅ **Model reconstruction**: Deserialize JSON back to functional sklearn models
- ✅ **Numpy support**: Proper handling of numpy arrays and scalar types
- ✅ **Type preservation**: Maintains data types during serialization/deserialization
- ✅ **Prediction consistency**: Deserialized models produce identical predictions

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

Currently working models (✅ = fully supported):

### Linear Models
- ✅ **LinearRegression** - Linear regression with perfect reconstruction
- ✅ **LogisticRegression** - Logistic regression with all parameters
- ✅ **Ridge** - Ridge regression with regularization
- ✅ **Lasso** - Lasso regression with L1 regularization

### Models with Limitations
- ⚠️ **SVC/SVR** - Support Vector models (missing some internal attributes)
- ⚠️ **DecisionTree** - Tree models (tree structure not fully serialized)
- ⚠️ **KNeighbors** - Neighbor models (training data not serialized)
- ⚠️ **MLP** - Neural networks (weights partially supported)
- ❌ **Ensemble models** - RandomForest, GradientBoosting (complex internal structures)

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

### Testing Model Compatibility

Run the test suite to check which models work in your environment:

```bash
python test_models.py
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

1. **Complex internal structures**: Models with complex internal objects (like tree structures, estimator arrays) are not fully supported
2. **Training data**: Models that store training data (like KNeighbors) lose this information
3. **Custom objects**: Only standard sklearn models and numpy types are supported
4. **Version compatibility**: Serialized models may not work across different sklearn versions

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
├── src/sklser/          # Main library code
│   ├── __init__.py      # Core serialization functions
│   └── py.typed         # Type hint marker
├── models/              # Example model scripts
├── test_models.py       # Comprehensive test suite
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

## Contributing

Contributions are welcome! Areas for improvement:

1. **Support for ensemble models** (RandomForest, GradientBoosting)
2. **Preprocessing objects** (StandardScaler, LabelEncoder)
3. **Binary serialization** for better performance
4. **Cross-version compatibility** handling

## License

This project is licensed under the MIT License.