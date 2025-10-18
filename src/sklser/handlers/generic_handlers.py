"""Generic fallback handler for basic types."""
import json
from typing import Any, Dict

from sklser.handlers.base import BaseTypeHandler


class GenericHandler(BaseTypeHandler):
    """Handler for basic JSON-serializable types."""
    
    def can_handle(self, value: Any) -> bool:
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def serialize(self, value: Any) -> Dict[str, Any]:
        return {"value": value, "type": str(type(value))}
    
    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        return value_dict["value"]


class FallbackHandler(BaseTypeHandler):
    """Handler for unserializable objects."""
    
    def can_handle(self, value: Any) -> bool:
        return True  # Always handles as last resort
    
    def serialize(self, value: Any) -> Dict[str, Any]:
        return {"value": "UNSERIALIZABLE", "type": str(type(value))}
    
    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        if value_dict["value"] in [
            "COMPLEX_OBJECT",
            "COMPLEX_LIST", 
            "UNSERIALIZABLE",
            "COMPLEX_TREE_OBJECT",
            "COMPLEX_LABELBINARIZER"
        ]:
            return None
        return value_dict["value"]