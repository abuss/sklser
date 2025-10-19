# API Reference

Welcome to the sklearn-serialize API reference. This library provides a simple and powerful way to serialize and deserialize scikit-learn models and transformers to/from JSON format.

## Quick Start

```python
from sklearn.preprocessing import StandardScaler
from sklser import serialize_json, deserialize_object

# Create and fit a model
scaler = StandardScaler()
scaler.fit([[1, 2], [3, 4]])

# Serialize to JSON
json_str = serialize_json(scaler)

# Deserialize back to object
reconstructed_scaler = deserialize_object(json_str)
```

## Core API

The sklearn-serialize library exposes two main functions that handle most use cases:

### Primary Functions

- **[`serialize_json(obj)`](core-functions.md#serialize_json)** - Convert any sklearn object to JSON string
- **[`deserialize_object(json_str)`](core-functions.md#deserialize_object)** - Convert JSON string back to sklearn object

### Advanced Components

For advanced users who need to customize serialization behavior:

- **[Handler System](handlers.md)** - Extensible handler architecture for custom serialization
- **[Exception Handling](exceptions.md)** - Custom exceptions and error handling

## Supported Objects

sklearn-serialize supports a wide range of scikit-learn objects:

### Basic Models
- Linear models (LinearRegression, LogisticRegression, etc.)
- Tree-based models (DecisionTree, RandomForest, etc.)
- Support Vector Machines
- Naive Bayes classifiers
- Clustering algorithms

### Preprocessors
- StandardScaler, MinMaxScaler, RobustScaler
- LabelEncoder, LabelBinarizer, OneHotEncoder
- PCA, TruncatedSVD
- Polynomial features

### Complex Structures
- **Pipelines** - Multi-step workflows
- **FeatureUnion** - Parallel feature processing
- **ColumnTransformer** - Column-specific transformations
- **FunctionTransformer** - Custom function wrappers

## Architecture

The library uses a handler-based architecture where each type of object has a specialized handler:

```
sklearn-serialize
├── Core API (serialize_json, deserialize_object)
├── Handler System
│   ├── BaseTypeHandler (abstract base)
│   ├── NumPy handlers (arrays, scalars)
│   ├── sklearn handlers (models, transformers)
│   ├── Advanced handlers (pipelines, complex structures)
│   └── Generic handlers (fallback)
└── Type Serializer (orchestrates handlers)
```

## Navigation

- **[Core Functions](core-functions.md)** - Detailed documentation of main API functions
- **[Handler System](handlers.md)** - Advanced customization and extension
- **[Exceptions](exceptions.md)** - Error handling and troubleshooting
- **[Examples](../examples/)** - Practical usage examples
- **[Guides](../guides/)** - Best practices and troubleshooting

## Version Information

This documentation covers sklearn-serialize version 1.0.0+, compatible with:
- Python 3.7+
- scikit-learn 0.24+
- NumPy 1.19+