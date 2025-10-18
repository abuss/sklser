"""Handlers for numpy types."""
import json
from typing import Any, Dict
import numpy as np
from numpy import ndarray

from sklser.handlers.base import BaseTypeHandler


class NumpyArrayHandler(BaseTypeHandler):
    """Handler for numpy arrays."""
    
    def can_handle(self, value: Any) -> bool:
        return isinstance(value, ndarray)
    
    def serialize(self, value: ndarray) -> Dict[str, Any]:
        return {
            "value": value.tolist(),
            "type": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": value.shape,
        }
    
    def deserialize(self, value_dict: Dict[str, Any]) -> ndarray:
        dtype = value_dict.get("dtype", "float64")
        shape = value_dict.get("shape")
        arr = np.array(value_dict["value"], dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr


class NumpyScalarHandler(BaseTypeHandler):
    """Handler for numpy scalar types."""
    
    def can_handle(self, value: Any) -> bool:
        return isinstance(value, (np.integer, np.floating))
    
    def serialize(self, value: Any) -> Dict[str, Any]:
        return {"value": value.item(), "type": str(type(value)), "numpy_type": True}
    
    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        value_type = value_dict["type"]
        value = value_dict["value"]
        
        if "int" in value_type:
            return np.int64(value)
        elif "float" in value_type:
            return np.float64(value)
        return value


class ListHandler(BaseTypeHandler):
    """Handler for lists that might contain complex objects."""
    
    def can_handle(self, value: Any) -> bool:
        return isinstance(value, list)
    
    def serialize(self, value: list) -> Dict[str, Any]:
        try:
            json.dumps(value)
            return {"value": value, "type": str(type(value))}
        except (TypeError, ValueError):
            try:
                serialized_list = []
                for item in value:
                    if isinstance(item, ndarray):
                        serialized_list.append(
                            {
                                "item_value": item.tolist(),
                                "item_type": "numpy.ndarray",
                                "dtype": str(item.dtype),
                                "shape": item.shape,
                            }
                        )
                    else:
                        json.dumps(item)
                        serialized_list.append(
                            {
                                "item_value": item,
                                "item_type": str(type(item)),
                            }
                        )

                return {
                    "value": serialized_list,
                    "type": str(type(value)),
                    "special_list": True,
                }
            except (TypeError, ValueError):
                return {
                    "value": "COMPLEX_LIST",
                    "type": str(type(value)),
                    "length": len(value),
                }
    
    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        if value_dict.get("special_list"):
            try:
                deserialized_list = []
                for item_dict in value_dict["value"]:
                    if item_dict.get("item_type") == "numpy.ndarray":
                        dtype = item_dict.get("dtype", "float64")
                        shape = item_dict.get("shape")
                        arr = np.array(item_dict["item_value"], dtype=dtype)
                        if shape:
                            arr = arr.reshape(shape)
                        deserialized_list.append(arr)
                    else:
                        deserialized_list.append(item_dict["item_value"])
                return deserialized_list
            except Exception as e:
                print(f"Warning: Could not deserialize special list: {e}")
                return None
        elif value_dict["value"] == "COMPLEX_LIST":
            return None
        else:
            return value_dict["value"]