"""Type serialization manager that coordinates different handlers."""

import json
from typing import Any, Dict
import numpy as np
from numpy import ndarray

# Import handlers directly without base classes for now to avoid circular imports
# We'll add proper inheritance later


class TypeSerializer:
    """Manages all type handlers for serialization/deserialization."""

    def __init__(self):
        # Initialize all handlers in order of priority
        self.handlers = [
            NumpyArrayHandler(),
            NumpyScalarHandler(),
            LabelBinarizerHandler(),
            TreeHandler(),
            ListHandler(),
            GenericHandler(),
            FallbackHandler(),  # Always last
        ]

    def serialize_value(self, value: Any) -> Dict[str, Any]:
        """Serialize a value using the first applicable handler."""
        for handler in self.handlers:
            if handler.can_handle(value):
                return handler.serialize(value)

        # Fallback if no handler matches
        return {"value": "UNSERIALIZABLE", "type": str(type(value))}

    def deserialize_value(self, value_dict: Dict[str, Any]) -> Any:
        """Deserialize a value using the appropriate handler."""
        value_type = value_dict.get("type", "")

        # Find the right handler based on type
        for handler in self.handlers:
            if handler.can_deserialize(value_dict):
                return handler.deserialize(value_dict)

        # Fallback
        return value_dict.get("value")


# Handler classes
class NumpyArrayHandler:
    """Handler for numpy arrays."""

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, ndarray)

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("type") == "numpy.ndarray"

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


class NumpyScalarHandler:
    """Handler for numpy scalar types."""

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, (np.integer, np.floating))

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("numpy_type", False)

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


class LabelBinarizerHandler:
    """Handler for sklearn LabelBinarizer objects."""

    def can_handle(self, value: Any) -> bool:
        return hasattr(
            value, "__class__"
        ) and "sklearn.preprocessing._label.LabelBinarizer" in str(type(value))

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("type") == "sklearn.preprocessing._label.LabelBinarizer"

    def serialize(self, value: Any) -> Dict[str, Any]:
        try:
            binarizer_data = {
                "classes_": value.classes_,
                "neg_label": value.neg_label,
                "pos_label": value.pos_label,
                "sparse_input_": value.sparse_input_,
                "sparse_output": value.sparse_output,
                "y_type_": value.y_type_,
            }
            return {
                "value": {
                    k: v.tolist() if isinstance(v, ndarray) else v
                    for k, v in binarizer_data.items()
                },
                "type": "sklearn.preprocessing._label.LabelBinarizer",
                "label_binarizer": True,
            }
        except Exception as e:
            return {
                "value": "COMPLEX_LABELBINARIZER",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        try:
            # Dynamic import to avoid issues
            import importlib

            preprocessing_module = importlib.import_module("sklearn.preprocessing")
            LabelBinarizer = getattr(preprocessing_module, "LabelBinarizer")

            binarizer_data = value_dict["value"]
            label_binarizer = LabelBinarizer()

            label_binarizer.classes_ = (
                np.array(binarizer_data["classes_"])
                if isinstance(binarizer_data["classes_"], list)
                else binarizer_data["classes_"]
            )
            label_binarizer.neg_label = binarizer_data["neg_label"]
            label_binarizer.pos_label = binarizer_data["pos_label"]
            label_binarizer.sparse_input_ = binarizer_data["sparse_input_"]
            label_binarizer.sparse_output = binarizer_data["sparse_output"]
            label_binarizer.y_type_ = binarizer_data["y_type_"]

            return label_binarizer
        except Exception as e:
            print(f"Warning: Could not reconstruct LabelBinarizer object: {e}")
            return None


class TreeHandler:
    """Handler for sklearn Tree objects."""

    def can_handle(self, value: Any) -> bool:
        return hasattr(value, "__class__") and "sklearn.tree._tree.Tree" in str(
            type(value)
        )

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("type") == "sklearn.tree._tree.Tree"

    def serialize(self, value: Any) -> Dict[str, Any]:
        try:
            tree_data = {
                "children_left": value.children_left,
                "children_right": value.children_right,
                "feature": value.feature,
                "threshold": value.threshold,
                "impurity": value.impurity,
                "n_node_samples": value.n_node_samples,
                "weighted_n_node_samples": value.weighted_n_node_samples,
                "value": value.value,
                "capacity": value.capacity,
                "node_count": value.node_count,
                "max_depth": value.max_depth,
                "n_features": value.n_features,
                "n_classes": value.n_classes,
                "n_outputs": value.n_outputs,
            }
            return {
                "value": {
                    k: v.tolist() if isinstance(v, ndarray) else v
                    for k, v in tree_data.items()
                },
                "type": "sklearn.tree._tree.Tree",
                "tree_structure": True,
            }
        except Exception as e:
            return {
                "value": "COMPLEX_TREE_OBJECT",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        try:
            import importlib

            tree_module = importlib.import_module("sklearn.tree._tree")
            Tree = getattr(tree_module, "Tree")

            tree_data = value_dict["value"]

            children_left = np.array(tree_data["children_left"], dtype=np.intp)
            children_right = np.array(tree_data["children_right"], dtype=np.intp)
            feature = np.array(tree_data["feature"], dtype=np.intp)
            threshold = np.array(tree_data["threshold"], dtype=np.float64)
            impurity = np.array(tree_data["impurity"], dtype=np.float64)
            n_node_samples = np.array(tree_data["n_node_samples"], dtype=np.intp)
            weighted_n_node_samples = np.array(
                tree_data["weighted_n_node_samples"], dtype=np.float64
            )
            value_array = np.array(tree_data["value"], dtype=np.float64)

            n_features = tree_data["n_features"]
            n_classes = np.array([tree_data["n_classes"]], dtype=np.intp)
            n_outputs = tree_data["n_outputs"]

            tree = Tree(n_features, n_classes, n_outputs)

            tree.children_left = children_left
            tree.children_right = children_right
            tree.feature = feature
            tree.threshold = threshold
            tree.impurity = impurity
            tree.n_node_samples = n_node_samples
            tree.weighted_n_node_samples = weighted_n_node_samples
            tree.value = value_array
            tree.capacity = tree_data["capacity"]
            tree.node_count = tree_data["node_count"]
            tree.max_depth = tree_data["max_depth"]

            return tree
        except Exception as e:
            print(f"Warning: Could not reconstruct Tree object: {e}")
            return None


class ListHandler:
    """Handler for lists that might contain complex objects."""

    def can_handle(self, value: Any) -> bool:
        return isinstance(value, list)

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("type", "").startswith(
            "<class 'list'>"
        ) or value_dict.get("special_list", False)

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


class GenericHandler:
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
                value_dict.get("type") == "numpy.ndarray",
            ]
        )

    def serialize(self, value: Any) -> Dict[str, Any]:
        return {"value": value, "type": str(type(value))}

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        return value_dict["value"]


class FallbackHandler:
    """Handler for unserializable objects and complex objects."""

    def can_handle(self, value: Any) -> bool:
        # Check if it's a complex object
        if hasattr(value, "__dict__") and hasattr(value, "__class__"):
            return True
        return True  # Always handles as last resort

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return True  # Always can deserialize as fallback

    def serialize(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "__dict__") and hasattr(value, "__class__"):
            return {
                "value": "COMPLEX_OBJECT",
                "type": str(type(value)),
                "class_name": value.__class__.__name__,
                "module": value.__class__.__module__,
            }
        else:
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


# Create a global instance
_type_serializer = TypeSerializer()


def serialize_value(value: Any) -> Dict[str, Any]:
    """Serialize a single value with proper type handling."""
    return _type_serializer.serialize_value(value)


def deserialize_value(value_dict: Dict[str, Any]) -> Any:
    """Deserialize a single value with proper type reconstruction."""
    return _type_serializer.deserialize_value(value_dict)

