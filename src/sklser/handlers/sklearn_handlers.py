"""Handlers for sklearn-specific types."""

import importlib
from typing import Any, Dict
import numpy as np
from numpy import ndarray

from .base import BaseTypeHandler


class LabelBinarizerHandler(BaseTypeHandler):
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
            from sklearn.preprocessing import LabelBinarizer

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


class TreeHandler(BaseTypeHandler):
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


class SklearnObjectHandler(BaseTypeHandler):
    """Handler for general sklearn objects using introspection."""

    def can_handle(self, value: Any) -> bool:
        # Check if it's a sklearn object
        module_name = getattr(value.__class__, "__module__", "")
        return (
            hasattr(value, "__dict__")
            and hasattr(value, "__class__")
            and module_name.startswith("sklearn.")
        )

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("sklearn_object", False)

    def serialize(self, value: Any) -> Dict[str, Any]:
        try:
            # Get class information
            class_name = value.__class__.__name__
            module_name = value.__class__.__module__

            # Serialize all non-callable, non-private attributes
            attributes = {}
            for attr_name in dir(value):
                if not attr_name.startswith("_") and not callable(
                    getattr(value, attr_name)
                ):
                    attr_value = getattr(value, attr_name)

                    # Use global serializer for recursive serialization of complex attributes
                    from .. import _type_serializer

                    attributes[attr_name] = _type_serializer.serialize_value(attr_value)

            return {
                "__class__": class_name,
                "__module__": module_name,
                "attributes": attributes,
                "sklearn_object": True,
            }
        except Exception as e:
            return {
                "value": "COMPLEX_OBJECT",
                "type": str(type(value)),
                "class_name": value.__class__.__name__,
                "module": value.__class__.__module__,
                "error": str(e),
            }

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        try:
            # Import the class
            module_name = value_dict["__module__"]
            class_name = value_dict["__class__"]

            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

            # Create instance
            obj = cls()

            # Restore attributes
            from .. import _type_serializer

            for attr_name, attr_data in value_dict["attributes"].items():
                attr_value = _type_serializer.deserialize_value(attr_data)
                setattr(obj, attr_name, attr_value)

            return obj
        except Exception as e:
            print(
                f"Warning: Could not reconstruct {value_dict.get('__class__', 'Unknown')} object: {e}"
            )
            return None


class ComplexObjectHandler(BaseTypeHandler):
    """Fallback handler for complex objects that can't be serialized."""

    def can_handle(self, value: Any) -> bool:
        # Only handle objects that aren't handled by other specialized handlers
        # This should be very restrictive and only catch truly unserializable objects
        return (
            hasattr(value, "__dict__")
            and hasattr(value, "__class__")
            and not getattr(value.__class__, "__module__", "").startswith("sklearn.")
        )

    def serialize(self, value: Any) -> Dict[str, Any]:
        return {
            "value": "COMPLEX_OBJECT",
            "type": str(type(value)),
            "class_name": value.__class__.__name__,
            "module": value.__class__.__module__,
        }

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        return None

