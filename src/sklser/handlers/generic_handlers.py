"""Generic fallback handler for basic types."""

import json
from typing import Any, Dict

from .base import BaseTypeHandler


class GenericHandler(BaseTypeHandler):
    """Handler for basic JSON-serializable types."""

    def can_handle(self, value: Any) -> bool:
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        # Generic handler can handle most basic types
        return not any(
            [
                value_dict.get("numpy_type", False),
                value_dict.get("special_list", False),
                value_dict.get("label_binarizer", False),
                value_dict.get("tree_structure", False),
                value_dict.get("pipeline", False),
                value_dict.get("feature_union", False),
                value_dict.get("column_transformer", False),
                value_dict.get("function_transformer", False),
                value_dict.get("type") == "numpy.ndarray",
            ]
        )

    def serialize(self, value: Any) -> Dict[str, Any]:
        return {"value": value, "type": str(type(value))}

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        return value_dict["value"]


class TupleHandler(BaseTypeHandler):
    """Handler for tuples."""

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, tuple)

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("tuple_data", False)

    def serialize(self, value: tuple) -> Dict[str, Any]:
        # Import the global serializer to handle recursive serialization
        from .. import _type_serializer

        serialized_items = []
        for item in value:
            serialized_item = _type_serializer.serialize_value(item)
            serialized_items.append(serialized_item)

        return {
            "value": serialized_items,
            "type": str(type(value)),
            "tuple_data": True,
        }

    def deserialize(self, value_dict: Dict[str, Any]) -> tuple:
        # Import the global serializer to handle recursive deserialization
        from .. import _type_serializer

        deserialized_items = []
        for item_data in value_dict["value"]:
            deserialized_item = _type_serializer.deserialize_value(item_data)
            deserialized_items.append(deserialized_item)

        return tuple(deserialized_items)


class FallbackHandler(BaseTypeHandler):
    """Handler for unserializable objects."""

    def can_handle(self, value: Any) -> bool:
        return True  # Always handles as last resort

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return True  # Always handles as last resort

    def serialize(self, value: Any) -> Dict[str, Any]:
        return {"value": "UNSERIALIZABLE", "type": str(type(value))}

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        if value_dict["value"] in [
            "COMPLEX_OBJECT",
            "COMPLEX_LIST",
            "UNSERIALIZABLE",
            "COMPLEX_TREE_OBJECT",
            "COMPLEX_LABELBINARIZER",
        ]:
            return None
        return value_dict["value"]

