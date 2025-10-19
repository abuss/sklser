# Exception Handling

The sklearn-serialize library uses standard Python exceptions and graceful error handling patterns rather than custom exception classes. This document covers common error scenarios and how to handle them.

## Error Handling Philosophy

The library follows a **graceful degradation** approach:

1. **Serialization errors** return error objects instead of raising exceptions
2. **Deserialization errors** return `None` and print error messages
3. **Handler failures** fall back to simpler serialization methods

This design ensures that the library doesn't crash your application, but you should always verify results.

## Serialization Error Handling

### Error Object Format

When `serialize_json()` encounters an error, it returns a JSON string containing an error object:

```python
{
    "error": "Serialization failed: [detailed error message]",
    "type": "[object type that failed]"
}
```

### Example Error Scenarios

#### Unsupported Object Type
```python
from sklser import serialize_json

class UnsupportedClass:
    def __init__(self):
        self.data = "some data"

obj = UnsupportedClass()
result = serialize_json(obj)
print(result)
# Output: {"error": "Serialization failed: ...", "type": "<class '__main__.UnsupportedClass'>"}
```

#### Circular References
```python
class CircularRef:
    def __init__(self):
        self.ref = self  # Circular reference

obj = CircularRef()
result = serialize_json(obj)
# Returns error object instead of infinite recursion
```

#### Complex Object Attributes
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.custom_attribute = lambda x: x  # Function objects can't be serialized
result = serialize_json(model)
# May return error object or fall back to partial serialization
```

### Detecting Serialization Errors

```python
import json
from sklser import serialize_json

def safe_serialize(obj):
    result = serialize_json(obj)
    try:
        parsed = json.loads(result)
        if "error" in parsed:
            print(f"Serialization failed: {parsed['error']}")
            return None
        return result
    except json.JSONDecodeError:
        print("Invalid JSON returned")
        return None

# Usage
serialized = safe_serialize(my_object)
if serialized is not None:
    # Serialization successful
    pass
```

## Deserialization Error Handling

### Error Behavior

When `deserialize_object()` encounters an error:

1. Prints an error message to stdout: `"Deserialization error: [details]"`
2. Returns `None`

### Example Error Scenarios

#### Invalid JSON
```python
from sklser import deserialize_object

result = deserialize_object("invalid json string")
# Prints: Deserialization error: Expecting value: line 1 column 1 (char 0)
# Returns: None
```

#### Missing Required Modules
```python
# If sklearn is not installed or wrong version
json_str = '{"type": "sklearn.linear_model.LinearRegression", ...}'
result = deserialize_object(json_str)
# Prints: Deserialization error: No module named 'sklearn.linear_model'
# Returns: None
```

#### Corrupted Serialization Data
```python
json_str = '{"type": "sklearn.linear_model.LinearRegression", "value": {"invalid": "data"}}'
result = deserialize_object(json_str)
# Prints: Deserialization error: [specific attribute error]
# Returns: None
```

### Safe Deserialization Pattern

```python
from sklser import deserialize_object

def safe_deserialize(json_str, expected_type=None):
    """Safely deserialize with validation."""
    try:
        result = deserialize_object(json_str)
        
        if result is None:
            print("Deserialization returned None")
            return None
            
        if expected_type and not isinstance(result, expected_type):
            print(f"Expected {expected_type}, got {type(result)}")
            return None
            
        return result
        
    except Exception as e:
        print(f"Unexpected error during deserialization: {e}")
        return None

# Usage
from sklearn.linear_model import LinearRegression

model = safe_deserialize(json_string, LinearRegression)
if model is not None:
    predictions = model.predict(X_test)
```

## Handler-Specific Errors

### NumPy Handler Errors

```python
# TypeError or ValueError during array conversion
{
    "error": "Array serialization failed: could not convert string to float: 'invalid'",
    "type": "numpy.ndarray"
}
```

### sklearn Handler Errors

```python
# Missing attributes or incompatible sklearn versions
{
    "error": "Sklearn object serialization failed: 'LinearRegression' object has no attribute 'new_feature'",
    "type": "sklearn.linear_model._base.LinearRegression"
}
```

### Pipeline Handler Errors

```python
# Step serialization failures
{
    "error": "Pipeline serialization failed: Step 'custom_transformer' could not be serialized",
    "type": "sklearn.pipeline.Pipeline"
}
```

## Common Error Scenarios and Solutions

### Version Compatibility Issues

**Problem:** Model serialized with sklearn 1.0 fails to deserialize with sklearn 0.24

**Solution:**
```python
import sklearn
print(f"sklearn version: {sklearn.__version__}")

# Check version compatibility before deserializing
def check_sklearn_compatibility(json_str):
    import json
    data = json.loads(json_str)
    if "sklearn_version" in data:
        print(f"Model created with sklearn {data['sklearn_version']}")
        print(f"Current sklearn version: {sklearn.__version__}")
```

### Large Model Memory Issues

**Problem:** RandomForest with 1000+ trees causes memory errors

**Solution:**
```python
import sys
import json

def estimate_memory_usage(json_str):
    """Estimate memory usage of serialized model."""
    size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
    print(f"Serialized size: {size_mb:.2f} MB")
    
    # Rough estimate: deserialization uses 2-3x serialized size
    estimated_memory = size_mb * 3
    print(f"Estimated memory needed: {estimated_memory:.2f} MB")
    
    return estimated_memory < 100  # Safe threshold

# Check before deserializing
if estimate_memory_usage(json_string):
    model = deserialize_object(json_string)
```

### Custom Attribute Handling

**Problem:** sklearn objects with custom attributes fail to serialize

**Solution:**
```python
def clean_sklearn_object(obj):
    """Remove problematic custom attributes before serialization."""
    # List of safe sklearn attributes
    safe_attrs = ['fit', 'predict', 'transform', 'score', 'get_params', 'set_params']
    
    # Create copy and remove custom attributes
    import copy
    cleaned = copy.deepcopy(obj)
    
    for attr in dir(cleaned):
        if not attr.startswith('_') and attr not in safe_attrs:
            if hasattr(cleaned, attr) and not callable(getattr(cleaned, attr)):
                try:
                    delattr(cleaned, attr)
                except AttributeError:
                    pass
    
    return cleaned

# Usage
cleaned_model = clean_sklearn_object(my_model)
json_str = serialize_json(cleaned_model)
```

## Debugging Error Messages

### Enable Verbose Error Reporting

```python
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)

def verbose_serialize(obj):
    """Serialize with full error tracking."""
    try:
        from sklser import _type_serializer
        return _type_serializer.serialize_value(obj)
    except Exception as e:
        print(f"Serialization error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return None

def verbose_deserialize(json_str):
    """Deserialize with full error tracking."""
    try:
        from sklser import _type_serializer
        import json
        data = json.loads(json_str)
        return _type_serializer.deserialize_value(data)
    except Exception as e:
        print(f"Deserialization error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return None
```

### Inspect Serialization Output

```python
import json

def inspect_serialization(obj):
    """Inspect what gets serialized."""
    result = serialize_json(obj)
    try:
        data = json.loads(result)
        print("Serialization structure:")
        print(json.dumps(data, indent=2)[:500] + "..." if len(result) > 500 else data)
        
        # Check for error markers
        if "error" in data:
            print(f"❌ Serialization failed: {data['error']}")
        else:
            print("✅ Serialization successful")
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON output: {e}")

# Usage
inspect_serialization(my_model)
```

## Best Practices for Error Handling

1. **Always check return values**: Never assume serialization/deserialization succeeded
2. **Use wrapper functions**: Implement safe_serialize/safe_deserialize patterns
3. **Log errors appropriately**: Don't let silent failures hide issues
4. **Test thoroughly**: Verify round-trip serialization in your test suite
5. **Handle version differences**: Consider sklearn version compatibility
6. **Monitor resource usage**: Large models can cause memory issues

## Error Prevention

### Pre-serialization Validation

```python
def validate_for_serialization(obj):
    """Check if object is likely to serialize successfully."""
    
    # Check for basic sklearn interface
    if hasattr(obj, 'get_params'):
        try:
            params = obj.get_params()
            print(f"✅ Object has {len(params)} parameters")
        except Exception as e:
            print(f"❌ get_params() failed: {e}")
            return False
    
    # Check if fitted (for models)
    if hasattr(obj, 'predict') or hasattr(obj, 'transform'):
        fitted_attrs = ['coef_', 'intercept_', 'classes_', 'n_features_in_']
        is_fitted = any(hasattr(obj, attr) for attr in fitted_attrs)
        print(f"✅ Model appears {'fitted' if is_fitted else 'unfitted'}")
    
    return True

# Usage
if validate_for_serialization(my_model):
    json_str = serialize_json(my_model)
```

The sklearn-serialize library prioritizes stability and graceful degradation over strict error reporting, making it safe to use in production environments while still providing enough information to debug issues when they occur.