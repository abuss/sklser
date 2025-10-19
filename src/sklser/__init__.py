"""
sklearn-serialize: A library for serializing and deserializing scikit-learn objects.

This package provides simple functions to serialize scikit-learn models and transformers
to JSON format, enabling easy storage and transfer of trained models.

Main functions:
- serialize_json(obj): Serialize a sklearn object to JSON string
- deserialize_object(json_str): Deserialize a JSON string back to sklearn object

Example:
    from sklearn.preprocessing import StandardScaler
    from sklser import serialize_json, deserialize_object

    scaler = StandardScaler()
    scaler.fit([[1, 2], [3, 4]])

    # Serialize
    json_str = serialize_json(scaler)

    # Deserialize
    reconstructed_scaler = deserialize_object(json_str)
"""

import json
from typing import Any, Dict

# Import all handler classes to maintain backward compatibility for direct imports
from .handlers.base import BaseTypeHandler
from .handlers.numpy_handlers import NumpyArrayHandler, NumpyScalarHandler, ListHandler
from .handlers.sklearn_handlers import (
    LabelBinarizerHandler,
    TreeHandler,
    SklearnObjectHandler,
    ComplexObjectHandler,
)
from .handlers.sklearn_advanced_handlers import (
    PipelineHandler,
    FeatureUnionHandler,
    ColumnTransformerHandler,
    FunctionTransformerHandler,
)
from .handlers.generic_handlers import GenericHandler, TupleHandler, FallbackHandler


# Create a combined TypeSerializer with all handlers
class CombinedTypeSerializer:
    """Enhanced TypeSerializer that includes both modular and local handlers."""

    def __init__(self):
        # Create handlers directly
        self.base_handlers = [
            NumpyArrayHandler(),
            NumpyScalarHandler(),
            LabelBinarizerHandler(),
            TreeHandler(),
            SklearnObjectHandler(),  # Handle sklearn objects before falling back to ComplexObjectHandler
            ComplexObjectHandler(),
            ListHandler(),
            TupleHandler(),
            GenericHandler(),
            FallbackHandler(),  # Always last
        ]

        # Add our advanced handlers
        self.advanced_handlers = [
            PipelineHandler(),
            FeatureUnionHandler(),
            ColumnTransformerHandler(),
            FunctionTransformerHandler(),
        ]

        # Combine handlers - advanced handlers first for priority
        self.all_handlers = self.advanced_handlers + self.base_handlers

    def serialize_value(self, value: Any) -> Dict[str, Any]:
        """Serialize a value using the first applicable handler."""
        for handler in self.all_handlers:
            if handler.can_handle(value):
                return handler.serialize(value)

        # Fallback if no handler matches
        return {"value": "UNSERIALIZABLE", "type": str(type(value))}

    def deserialize_value(self, value_dict: Dict[str, Any]) -> Any:
        """Deserialize a value using the appropriate handler."""
        # Find the right handler based on type
        for handler in self.all_handlers:
            if hasattr(handler, "can_deserialize") and handler.can_deserialize(
                value_dict
            ):
                return handler.deserialize(value_dict)

        # Fallback
        return value_dict.get("value")


# Create global serializer instance
_type_serializer = CombinedTypeSerializer()


# Main API functions
def serialize_json(obj: object) -> str:
    """Serialize a sklearn model to JSON string."""
    try:
        # Use the combined type serializer to handle the object
        serialized_dict = _type_serializer.serialize_value(obj)

        # Add class information for top-level object if it's not already there
        if "__class__" not in serialized_dict:
            serialized_dict["__class__"] = obj.__class__.__name__
            serialized_dict["__module__"] = obj.__class__.__module__

        # Convert to JSON string
        return json.dumps(serialized_dict, indent=4)
    except Exception as e:
        return json.dumps(
            {"error": f"Serialization failed: {str(e)}", "type": str(type(obj))},
            indent=4,
        )


def deserialize_object(json_str: str) -> object:
    """Deserialize a JSON string back to a sklearn model."""
    try:
        # Parse JSON string
        obj_dict = json.loads(json_str)

        # Use the combined type serializer to handle the object
        return _type_serializer.deserialize_value(obj_dict)
    except Exception as e:
        print(f"Deserialization error: {e}")
        return None


# Export the main API for convenience
__all__ = [
    "serialize_json",
    "deserialize_object",
    # Handler classes for advanced usage
    "BaseTypeHandler",
    "NumpyArrayHandler",
    "NumpyScalarHandler",
    "ListHandler",
    "LabelBinarizerHandler",
    "TreeHandler",
    "SklearnObjectHandler",
    "ComplexObjectHandler",
    "GenericHandler",
    "TupleHandler",
    "FallbackHandler",
    "PipelineHandler",
    "FeatureUnionHandler",
    "ColumnTransformerHandler",
    "FunctionTransformerHandler",
]

