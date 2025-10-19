---
layout: home
title: sklearn-serialize
---

# sklearn-serialize

A comprehensive Python library for serializing scikit-learn models to JSON format, enabling easy model storage, sharing, and deployment.

## Overview

sklearn-serialize provides a robust solution for converting trained scikit-learn models into JSON format, allowing you to:

- **Save trained models** to disk in a human-readable format
- **Load models** from JSON files for inference
- **Share models** across different environments and platforms
- **Version control** your models with Git and other VCS
- **Deploy models** in web applications, microservices, and cloud environments

## Key Features

### ✅ Wide Model Support
- **Linear Models**: LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
- **Tree Models**: DecisionTree, RandomForest, ExtraTrees, GradientBoosting
- **Support Vector Machines**: SVC, SVR, LinearSVC, LinearSVR
- **Neural Networks**: MLPClassifier, MLPRegressor
- **Neighbors**: KNeighborsClassifier, KNeighborsRegressor
- **Ensemble Methods**: AdaBoost, Bagging, VotingClassifier, VotingRegressor
- **Preprocessing**: StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
- **Pipelines**: Full support for sklearn Pipelines and FeatureUnions
- **Advanced**: ColumnTransformer, FunctionTransformer

### ✅ Pipeline Support
Complete support for complex sklearn pipelines with multiple preprocessing steps:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import sklser

# Create and train a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

# Serialize to JSON
json_data = sklser.serialize_json(pipeline)

# Load from JSON
loaded_pipeline = sklser.deserialize_object(json_data)
```

### ✅ Easy to Use
Simple API with just two main functions:

```python
import sklser

# Serialize any sklearn object to JSON
json_data = sklser.serialize_json(model)

# Deserialize from JSON back to sklearn object
model = sklser.deserialize_object(json_data)
```

## Quick Start

### Installation

```bash
pip install sklearn-serialize
```

### Basic Usage

```python
import sklser
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create and train a model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = LogisticRegression()
model.fit(X, y)

# Serialize to JSON
json_data = sklser.serialize_json(model)

# Save to file
with open('model.json', 'w') as f:
    f.write(json_data)

# Load from file
with open('model.json', 'r') as f:
    json_data = f.read()

# Deserialize back to sklearn model
loaded_model = sklser.deserialize_object(json_data)

# Use the loaded model
predictions = loaded_model.predict(X[:5])
print(f"Predictions: {predictions}")
```

## Documentation Structure

<div class="grid-container">
  <div class="grid-item">
    <h3><a href="api/">📚 API Reference</a></h3>
    <p>Complete API documentation with function signatures, parameters, and return types.</p>
    <ul>
      <li><a href="api/core-functions.html">Core Functions</a></li>
      <li><a href="api/handlers.html">Handler System</a></li>
      <li><a href="api/exceptions.html">Exceptions</a></li>
    </ul>
  </div>

  <div class="grid-item">
    <h3><a href="examples/">💡 Examples & Tutorials</a></h3>
    <p>Step-by-step guides and practical examples for common use cases.</p>
    <ul>
      <li><a href="examples/basic-usage.html">Basic Usage</a></li>
      <li><a href="examples/pipeline-serialization.html">Pipeline Serialization</a></li>
      <li><a href="examples/advanced-features.html">Advanced Features</a></li>
      <li><a href="examples/model-deployment.html">Model Deployment</a></li>
    </ul>
  </div>

  <div class="grid-item">
    <h3><a href="guides/">📖 Guides</a></h3>
    <p>In-depth guides covering advanced topics and best practices.</p>
    <ul>
      <li><a href="guides/supported-models.html">Supported Models</a></li>
      <li><a href="guides/troubleshooting.html">Troubleshooting</a></li>
      <li><a href="guides/performance.html">Performance Tips</a></li>
      <li><a href="guides/contributing.html">Contributing</a></li>
    </ul>
  </div>
</div>

## Real-World Examples

### Web Application Deployment
```python
# Flask web service example
from flask import Flask, request, jsonify
import sklser

app = Flask(__name__)

# Load model once at startup
with open('trained_model.json', 'r') as f:
    model = sklser.deserialize_object(f.read())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': prediction[0]})
```

### Model Versioning with Git
```bash
# Save model to version control
git add models/v1.2.3/classifier.json
git commit -m "Add trained classifier v1.2.3 with 94.2% accuracy"
git tag model-v1.2.3
```

### Cross-Platform Model Sharing
```python
# Train on Linux server
model = train_model()
json_data = sklser.serialize_json(model)

# Deploy on Windows production server
model = sklser.deserialize_object(json_data)
predictions = model.predict(production_data)
```

## Performance

sklearn-serialize is designed for performance:

- **Fast serialization**: Optimized JSON encoding
- **Compact output**: Minimal JSON size
- **Quick deserialization**: Efficient object reconstruction
- **Memory efficient**: Streaming support for large models

## Compatibility

- **Python**: 3.7+
- **scikit-learn**: 0.24+
- **NumPy**: 1.17+
- **Cross-platform**: Windows, macOS, Linux

## Community & Support

- **GitHub**: [sklearn-serialize](https://github.com/your-username/sklearn-serialize)
- **Issues**: [Report bugs or request features](https://github.com/your-username/sklearn-serialize/issues)
- **Discussions**: [Community discussions](https://github.com/your-username/sklearn-serialize/discussions)

## License

sklearn-serialize is released under the MIT License. See [LICENSE](https://github.com/your-username/sklearn-serialize/blob/main/LICENSE) for details.

---

<style>
.grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.grid-item {
  border: 1px solid #e1e4e8;
  border-radius: 6px;
  padding: 20px;
  background-color: #f8f9fa;
}

.grid-item h3 {
  margin-top: 0;
  color: #0366d6;
}

.grid-item ul {
  margin-bottom: 0;
}

.grid-item a {
  text-decoration: none;
}

.grid-item a:hover {
  text-decoration: underline;
}
</style>