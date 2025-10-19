# Core Functions

The sklearn-serialize library provides two primary functions that handle serialization and deserialization of scikit-learn objects.

## serialize_json()

Convert any sklearn object to a JSON string representation.

### Signature

```python
def serialize_json(obj: object) -> str
```

### Parameters

- **obj** (`object`): Any scikit-learn model, transformer, or pipeline to serialize

### Returns

- **str**: JSON string representation of the object

### Description

The `serialize_json()` function takes any sklearn object and converts it to a JSON string that can be stored in files, databases, or transmitted over networks. The function handles complex nested structures including pipelines, feature unions, and column transformers.

### Example Usage

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklser import serialize_json

# Simple transformer
scaler = StandardScaler()
scaler.fit([[1, 2], [3, 4]])
json_str = serialize_json(scaler)

# Complex pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)
pipeline_json = serialize_json(pipeline)
```

### Error Handling

If serialization fails, the function returns a JSON error object instead of raising an exception:

```python
{
    "error": "Serialization failed: [error message]",
    "type": "[object type]"
}
```

### Supported Objects

- **Linear Models**: LinearRegression, LogisticRegression, Ridge, Lasso, etc.
- **Tree Models**: DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor, etc.
- **SVM**: SVC, SVR with various kernels
- **Neural Networks**: MLPClassifier, MLPRegressor
- **Preprocessing**: StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, etc.
- **Pipelines**: Pipeline, FeatureUnion, ColumnTransformer
- **Complex Structures**: Nested pipelines with multiple transformers

---

## deserialize_object()

Convert a JSON string back to the original sklearn object.

### Signature

```python
def deserialize_object(json_str: str) -> object
```

### Parameters

- **json_str** (`str`): JSON string created by `serialize_json()`

### Returns

- **object**: Reconstructed sklearn object, or `None` if deserialization fails

### Description

The `deserialize_object()` function takes a JSON string produced by `serialize_json()` and reconstructs the original sklearn object with all its fitted parameters, state, and configuration.

### Example Usage

```python
from sklser import serialize_json, deserialize_object

# Serialize
original_model = LogisticRegression()
original_model.fit(X_train, y_train)
json_str = serialize_json(original_model)

# Deserialize
reconstructed_model = deserialize_object(json_str)

# Use reconstructed model
predictions = reconstructed_model.predict(X_test)
```

### Error Handling

If deserialization fails, the function prints an error message and returns `None`:

```python
model = deserialize_object(invalid_json)  # Returns None
# Prints: "Deserialization error: [error details]"
```

### Verification

After deserialization, you can verify the object was reconstructed correctly:

```python
import numpy as np

# Compare predictions
original_pred = original_model.predict(X_test)
reconstructed_pred = reconstructed_model.predict(X_test)
assert np.allclose(original_pred, reconstructed_pred)

# Compare parameters
assert original_model.get_params() == reconstructed_model.get_params()
```

---

## Complete Workflow Example

Here's a complete example showing the full serialize → save → load → deserialize workflow:

```python
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklser import serialize_json, deserialize_object

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])
pipeline.fit(X_train, y_train)

# Serialize to JSON
json_str = serialize_json(pipeline)

# Save to file
with open('model.json', 'w') as f:
    f.write(json_str)

# Load from file
with open('model.json', 'r') as f:
    loaded_json = f.read()

# Deserialize back to object
reconstructed_pipeline = deserialize_object(loaded_json)

# Verify functionality
original_score = pipeline.score(X_test, y_test)
reconstructed_score = reconstructed_pipeline.score(X_test, y_test)

print(f"Original accuracy: {original_score:.4f}")
print(f"Reconstructed accuracy: {reconstructed_score:.4f}")
print(f"Scores match: {abs(original_score - reconstructed_score) < 1e-10}")
```

---

## Performance Notes

- **Memory Usage**: Large models (like RandomForest with many trees) will produce large JSON strings
- **Serialization Speed**: Complex pipelines take longer to serialize due to nested structure analysis
- **File Size**: JSON format is human-readable but not optimized for size. For production, consider compressing the JSON

## Best Practices

1. **Always verify**: Test that deserialized objects produce the same predictions
2. **Handle errors**: Check if `deserialize_object()` returns `None`
3. **Version compatibility**: Models serialized with one sklearn version may not work with another
4. **Store metadata**: Consider storing sklearn version, model type, and creation date alongside the JSON

## Limitations

- **Custom objects**: Only sklearn objects and basic Python types are supported
- **Version dependencies**: Deserialization requires the same sklearn modules that were used during serialization
- **Memory requirements**: Very large models may consume significant memory during serialization/deserialization