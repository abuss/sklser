# Sklearn Model Serialization Test Suite

This directory contains test scripts that demonstrate the complete workflow of training sklearn models, serializing them to JSON files on disk, and then loading and testing them later.

## Files

### `train_and_serialize.py`
- Trains multiple sklearn models on sample datasets
- Serializes each model to a JSON file using the sklser library
- Saves test data for later validation
- Creates a summary of serialization results

### `load_and_predict.py`
- Loads previously serialized models from JSON files
- Tests predictions on saved test data
- Calculates performance metrics
- Can test all models or a specific model

### `train_and_serialize_pipelines.py`
- Trains multiple sklearn pipeline models on sample datasets
- Demonstrates various pipeline types (simple, complex, feature selection, feature union, regression, neural)
- Serializes each pipeline to a JSON file using the sklser library
- Saves test data for later validation

### `load_and_test_pipelines.py`
- Loads previously serialized pipeline models from JSON files
- Tests predictions on saved test data
- Validates pipeline steps and performance
- Demonstrates pipeline deserialization functionality

### `README.md`
- This documentation file

## Usage

### 1. First, train and serialize models:

```bash
# Activate virtual environment (if using one)
source ../venv/bin/activate

# Train regular models and save to JSON files
python train_and_serialize.py

# Train pipeline models and save to JSON files
python train_and_serialize_pipelines.py
```

This will create:
- Individual model JSON files (e.g., `logistic_regression_model.json`)
- Individual pipeline JSON files (e.g., `simple_pipeline_model.json`)
- `test_data.json` and `pipeline_test_data.json` - Test data for validation
- `serialization_results.json` and `pipeline_serialization_results.json` - Summary of successful serializations

### 2. Then, load and test the serialized models:

```bash
# Test all regular models
python load_and_predict.py

# Test all pipeline models
python load_and_test_pipelines.py

# Or test a specific regular model
python load_and_predict.py logistic_regression
```

This will:
- Load the serialized models/pipelines from JSON files
- Deserialize them back to sklearn model/pipeline objects
- Make predictions on test data
- Calculate accuracy/MSE metrics
- Save results to `testing_results.json` and `pipeline_testing_results.json`

## Example Output

### Training and Serialization:
```
============================================================
SKLEARN MODEL TRAINING AND SERIALIZATION
============================================================
Creating sample datasets...
✓ Test data saved to test_data.json

Training logistic_regression...
✓ logistic_regression trained successfully
Serializing logistic_regression...
✓ logistic_regression serialized to logistic_regression_model.json

Training linear_regression...
✓ linear_regression trained successfully
...

Success rate: 5/6 (83.3%)
```

### Pipeline Training and Serialization:
```
============================================================
SKLEARN PIPELINE TRAINING AND SERIALIZATION
============================================================
Creating sample datasets...
✓ Test data saved to models/pipeline_test_data.json

Training simple_pipeline...
✓ simple_pipeline trained successfully (3.5ms)
Serializing simple_pipeline...
✓ simple_pipeline serialized to models/simple_pipeline_model.json (0.6ms)
✓ Test score: 0.6550

...

Success rate: 6/6 (100.0%)
```

### Loading and Testing:
```
============================================================
SKLEARN MODEL DESERIALIZATION AND TESTING
============================================================
✓ Test data loaded successfully
Found 5 models to test...

=== Testing logistic_regression_model.json ===
✓ JSON file loaded successfully
✓ Model deserialized successfully
✓ Predictions generated successfully
✓ Accuracy: 0.8850

...

Testing success rate: 5/5 (100.0%)
```

### Pipeline Loading and Testing:
```
============================================================
SKLEARN PIPELINE DESERIALIZATION AND TESTING
============================================================
✓ Test data loaded successfully
Found 6 pipeline models to test...

=== Testing simple_pipeline_model.json ===
✓ JSON file loaded successfully
✓ Model deserialized successfully (0.3ms)
✓ Predictions generated successfully (0.2ms)
✓ Accuracy: 0.6550
✓ Pipeline steps: scaler -> classifier

...

Testing success rate: 6/6 (100.0%)
```

## Generated Files

After running both scripts, you'll have:

```
test_serialization/
├── models/
│   ├── logistic_regression_model.json
│   ├── linear_regression_model.json
│   ├── svc_model.json
│   ├── knn_classifier_model.json
│   ├── mlp_classifier_model.json
│   ├── simple_pipeline_model.json
│   ├── complex_pipeline_model.json
│   ├── feature_selection_pipeline_model.json
│   ├── feature_union_pipeline_model.json
│   ├── regression_pipeline_model.json
│   ├── neural_pipeline_model.json
│   ├── test_data.json
│   ├── pipeline_test_data.json
│   ├── serialization_results.json
│   ├── pipeline_serialization_results.json
│   ├── testing_results.json
│   └── pipeline_testing_results.json
├── train_and_serialize.py
├── train_and_serialize_pipelines.py
├── load_and_predict.py
├── load_and_test_pipelines.py
└── README.md
```

## What This Demonstrates

1. **Real-world serialization**: Models and pipelines are saved as JSON files on disk, not just in memory
2. **Persistent storage**: Models can be trained once and used later
3. **Pipeline support**: Complete sklearn pipelines with multiple steps can be serialized and deserialized
4. **Validation**: Test data is preserved to validate that deserialized models work correctly
5. **Error handling**: Graceful handling of models that fail to serialize/deserialize
6. **Metrics**: Performance comparison between original and deserialized models
7. **Complex workflows**: Support for feature selection, feature union, and multi-step pipelines

## Requirements

- Python 3.11+
- scikit-learn
- numpy
- sklearn-serialize (sklser) library

## Notes

- The scripts use `make_classification` and `make_regression` to generate sample datasets
- Models that fail to serialize will be skipped during testing
- Test data is saved separately to ensure fair validation
- All results are logged and saved for analysis