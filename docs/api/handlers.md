# Handler System

The sklearn-serialize library uses an extensible handler-based architecture to serialize different types of objects. Each handler is responsible for serializing and deserializing specific types of objects.

## Architecture Overview

The handler system consists of:

1. **BaseTypeHandler** - Abstract base class defining the handler interface
2. **Specialized Handlers** - Concrete implementations for specific object types
3. **CombinedTypeSerializer** - Orchestrates all handlers and routes objects to appropriate handlers

## Handler Interface

All handlers inherit from `BaseTypeHandler` and implement four key methods:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTypeHandler(ABC):
    @abstractmethod
    def can_handle(self, value: Any) -> bool:
        """Check if this handler can handle the given value type."""
        pass
    
    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        """Check if this handler can deserialize the given value dict."""
        return False
    
    @abstractmethod
    def serialize(self, value: Any) -> Dict[str, Any]:
        """Serialize the value to a dictionary representation."""
        pass
    
    @abstractmethod
    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        """Deserialize the dictionary back to the original value."""
        pass
```

## Built-in Handlers

### NumPy Handlers

#### NumpyArrayHandler
Handles NumPy ndarray objects by converting them to lists with metadata.

```python
# Handles: numpy.ndarray
# Output format:
{
    "value": [[1, 2], [3, 4]],  # Array as nested lists
    "type": "numpy.ndarray",
    "dtype": "float64",
    "shape": [2, 2]
}
```

#### NumpyScalarHandler
Handles NumPy scalar types (integers, floats).

```python
# Handles: numpy.integer, numpy.floating
# Output format:
{
    "value": 42.0,
    "type": "<class 'numpy.float64'>",
    "numpy_type": true
}
```

#### ListHandler
Recursively handles Python lists containing serializable objects.

### sklearn Handlers

#### SklearnObjectHandler
General-purpose handler for most sklearn objects (models, transformers).

**Capabilities:**
- Extracts all object attributes
- Handles fitted and unfitted objects
- Preserves class information for reconstruction

**Example objects:** LinearRegression, StandardScaler, PCA, etc.

#### LabelBinarizerHandler
Specialized handler for sklearn's LabelBinarizer with specific attribute handling.

#### TreeHandler
Specialized handler for tree-based models with optimized tree structure serialization.

**Example objects:** DecisionTreeClassifier, RandomForestRegressor, etc.

#### ComplexObjectHandler
Fallback handler for sklearn objects that require deep attribute inspection.

### Advanced Handlers

#### PipelineHandler
Handles sklearn Pipeline objects by serializing each step.

```python
# Example serialization:
{
    "steps": [
        ["scaler", {"type": "StandardScaler", "value": {...}}],
        ["classifier", {"type": "LogisticRegression", "value": {...}}]
    ],
    "type": "sklearn.pipeline.Pipeline"
}
```

#### FeatureUnionHandler
Handles sklearn FeatureUnion objects that combine multiple transformers.

#### ColumnTransformerHandler
Handles sklearn ColumnTransformer for column-specific transformations.

#### FunctionTransformerHandler
Handles sklearn FunctionTransformer with custom functions.

### Generic Handlers

#### GenericHandler
Handles basic Python types (int, float, str, bool, None).

#### TupleHandler
Handles Python tuples by converting to lists with type information.

#### FallbackHandler
Last-resort handler that attempts basic serialization for unknown types.

## Handler Priority

Handlers are applied in a specific order (advanced handlers first):

1. **Advanced Handlers** (PipelineHandler, FeatureUnionHandler, etc.)
2. **NumPy Handlers** (NumpyArrayHandler, NumpyScalarHandler)
3. **sklearn Handlers** (LabelBinarizerHandler, TreeHandler, SklearnObjectHandler)
4. **Generic Handlers** (ComplexObjectHandler, ListHandler, TupleHandler, GenericHandler)
5. **FallbackHandler** (always last)

## Creating Custom Handlers

You can extend the library by creating custom handlers:

```python
from sklser.handlers.base import BaseTypeHandler
from typing import Any, Dict

class MyCustomHandler(BaseTypeHandler):
    def can_handle(self, value: Any) -> bool:
        # Return True if this handler should process this value
        return isinstance(value, MyCustomClass)
    
    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        # Return True if this handler can deserialize this dict
        return value_dict.get("type") == "MyCustomClass"
    
    def serialize(self, value: Any) -> Dict[str, Any]:
        # Convert your object to a JSON-serializable dict
        return {
            "value": {
                "attr1": value.attr1,
                "attr2": value.attr2.tolist() if hasattr(value.attr2, 'tolist') else value.attr2
            },
            "type": "MyCustomClass"
        }
    
    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        # Reconstruct your object from the dict
        obj = MyCustomClass()
        data = value_dict["value"]
        obj.attr1 = data["attr1"]
        obj.attr2 = np.array(data["attr2"]) if isinstance(data["attr2"], list) else data["attr2"]
        return obj
```

### Integrating Custom Handlers

To use custom handlers, you'll need to modify the `CombinedTypeSerializer`:

```python
from sklser import CombinedTypeSerializer

# Create custom serializer with your handler
class CustomTypeSerializer(CombinedTypeSerializer):
    def __init__(self):
        super().__init__()
        # Add your custom handler at the beginning (highest priority)
        self.all_handlers.insert(0, MyCustomHandler())

# Use custom serializer
custom_serializer = CustomTypeSerializer()
serialized = custom_serializer.serialize_value(my_object)
```

## Handler Development Guidelines

### Best Practices

1. **Specific `can_handle()` logic** - Be as specific as possible to avoid conflicts
2. **Robust error handling** - Wrap serialization/deserialization in try-catch blocks
3. **Preserve metadata** - Include type, module, and version information
4. **Handle nested objects** - Use the main serializer for nested objects
5. **Test thoroughly** - Verify round-trip serialization works correctly

### Common Patterns

#### Type Detection
```python
def can_handle(self, value: Any) -> bool:
    return (
        hasattr(value, '__class__') and 
        value.__class__.__module__ == 'my_module' and
        value.__class__.__name__ == 'MyClass'
    )
```

#### Nested Object Handling
```python
def serialize(self, value: Any) -> Dict[str, Any]:
    from sklser import _type_serializer  # Access global serializer
    
    return {
        "nested_obj": _type_serializer.serialize_value(value.nested_obj),
        "simple_attr": value.simple_attr,
        "type": "MyComplexClass"
    }
```

#### Error Recovery
```python
def serialize(self, value: Any) -> Dict[str, Any]:
    try:
        # Attempt complex serialization
        return self._complex_serialize(value)
    except Exception as e:
        # Fallback to basic serialization
        return {
            "value": "SERIALIZATION_FAILED",
            "type": str(type(value)),
            "error": str(e)
        }
```

## Debugging Handlers

### Common Issues

1. **Handler conflicts** - Multiple handlers claim the same object type
2. **Missing imports** - Required modules not available during deserialization
3. **Version incompatibility** - Object structure changed between sklearn versions
4. **Circular references** - Objects that reference themselves

### Debugging Tips

```python
# Enable handler debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check which handler is selected
from sklser import _type_serializer
for handler in _type_serializer.all_handlers:
    if handler.can_handle(my_object):
        print(f"Handler: {handler.__class__.__name__}")
        break

# Inspect serialized output
import json
serialized = serialize_json(my_object)
print(json.dumps(json.loads(serialized), indent=2))
```

## Performance Considerations

- **Handler order matters** - More specific handlers should come first
- **Complex objects** - Large sklearn objects (RandomForest with many trees) create large JSON
- **Memory usage** - Serialization temporarily doubles memory usage
- **Nested structures** - Deep pipelines increase processing time

The handler system provides flexibility and extensibility while maintaining a clean separation of concerns for different object types.