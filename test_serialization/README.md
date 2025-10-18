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

### `README.md`
- This documentation file

## Usage

### 1. First, train and serialize models:

```bash
# Activate virtual environment (if using one)
source ../venv/bin/activate

# Train models and save to JSON files
python train_and_serialize.py
```

This will create:
- Individual model JSON files (e.g., `logistic_regression_model.json`)
- `test_data.json` - Test data for validation
- `serialization_results.json` - Summary of which models were successfully serialized

### 2. Then, load and test the serialized models:

```bash
# Test all models
python load_and_predict.py

# Or test a specific model
python load_and_predict.py logistic_regression
```

This will:
- Load the serialized models from JSON files
- Deserialize them back to sklearn model objects
- Make predictions on test data
- Calculate accuracy/MSE metrics
- Save results to `testing_results.json`

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
│   ├── test_data.json
│   ├── serialization_results.json
│   └── testing_results.json
├── train_and_serialize.py
├── load_and_predict.py
└── README.md
```

## What This Demonstrates

1. **Real-world serialization**: Models are saved as JSON files on disk, not just in memory
2. **Persistent storage**: Models can be trained once and used later
3. **Validation**: Test data is preserved to validate that deserialized models work correctly
4. **Error handling**: Graceful handling of models that fail to serialize/deserialize
5. **Metrics**: Performance comparison between original and deserialized models

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