import inspect
import json
import importlib
from typing import Any, Dict
import numpy as np
from numpy import ndarray


def show_memebers(obj: object):
    members = inspect.getmembers(obj)
    print("==" * 30)
    for member in members:
        print(member)
    print("==" * 30)

    print(obj.__class__.__name__)
    for name, value in members:
        if not callable(value) and not name.startswith("_"):
            print(name, value, type(value))
    print("-" * 30)


def _serialize_value(value: Any) -> Dict[str, Any]:
    """Serialize a single value with proper type handling."""
    if isinstance(value, ndarray):
        return {
            "value": value.tolist(),
            "type": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": value.shape,
        }
    elif isinstance(value, (np.integer, np.floating)):
        return {"value": value.item(), "type": str(type(value)), "numpy_type": True}
    elif hasattr(
        value, "__class__"
    ) and "sklearn.preprocessing._label.LabelBinarizer" in str(type(value)):
        # Special handling for sklearn LabelBinarizer objects
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
    elif hasattr(value, "__class__") and "sklearn.tree._tree.Tree" in str(type(value)):
        # Special handling for sklearn Tree objects
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
    elif isinstance(value, list):
        # Handle lists that might contain complex objects
        try:
            # Try to serialize the list directly first
            json.dumps(value)
            return {"value": value, "type": str(type(value))}
        except (TypeError, ValueError):
            # If list contains non-serializable objects, try to serialize each element
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
                        # Try to serialize other items directly
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
                # If still can't serialize, mark as complex
                return {
                    "value": "COMPLEX_LIST",
                    "type": str(type(value)),
                    "length": len(value),
                }
    elif hasattr(value, "__dict__") and hasattr(value, "__class__"):
        # Handle complex sklearn objects (like Tree objects, estimators in ensembles)
        return {
            "value": "COMPLEX_OBJECT",
            "type": str(type(value)),
            "class_name": value.__class__.__name__,
            "module": value.__class__.__module__,
        }
    else:
        try:
            # Try to serialize the value directly
            json.dumps(value)
            return {"value": value, "type": str(type(value))}
        except (TypeError, ValueError):
            # If not serializable, mark as complex
            return {"value": "UNSERIALIZABLE", "type": str(type(value))}


def serialize_json(obj: object) -> str:
    """Serialize a sklearn model to JSON string."""
    members = inspect.getmembers(obj)
    out_dict: Dict[str, Any] = {}

    # Store class information
    class_name = obj.__class__.__name__
    module_name = obj.__class__.__module__
    out_dict["__class__"] = class_name
    out_dict["__module__"] = module_name

    # Serialize members
    member_dict = {}

    # Define important sklearn internal attributes that should be serialized
    important_sklearn_internals = {
        "_sparse",
        "_gamma",
        "_impl",
        "_intercept_",
        "_dual_coef_",
        "_n_support",
        "_num_iter",
        "_probA",
        "_probB",
        "_sparse_kernels",
        "_estimator_type",
        "_coef_",
        "_intercepts_",
        "_weights",
        "_biases",
        "_tree",
        "_trees",
        "_estimators_",
        "_oob_score",
        "_label_binarizer",
        "_classes",
        "_enc",
        "_fit_method",
        "_coef_grads",
        "_intercept_grads",
        "_loss_history",
        "_no_improvement_count",
        "_fit_X",
        "_y",
    }

    for name, value in members:
        # Include if: not callable, doesn't start with __ (dunder), and either doesn't start with _
        # OR is an important sklearn internal attribute
        should_include = (
            not callable(value)
            and not name.startswith("__")
            and (not name.startswith("_") or name in important_sklearn_internals)
        )

        if should_include:
            member_dict[name] = _serialize_value(value)

    out_dict["members"] = member_dict
    return json.dumps(out_dict, indent=4)


def _deserialize_value(value_dict: Dict[str, Any]) -> Any:
    """Deserialize a single value with proper type reconstruction."""
    value_type = value_dict["type"]
    value = value_dict["value"]

    if value_dict.get("type") == "numpy.ndarray":
        dtype = value_dict.get("dtype", "float64")
        shape = value_dict.get("shape")
        arr = np.array(value, dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr
    elif value_dict.get("type") == "sklearn.tree._tree.Tree":
        # Special handling for sklearn Tree objects
        try:
            # Import Tree class dynamically
            tree_module = importlib.import_module("sklearn.tree._tree")
            Tree = getattr(tree_module, "Tree")

            # Extract tree structure data
            tree_data = value_dict["value"]

            # Create arrays from the serialized data
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

            # Create a new Tree object with the reconstructed data
            n_features = tree_data["n_features"]
            n_classes = np.array([tree_data["n_classes"]], dtype=np.intp)
            n_outputs = tree_data["n_outputs"]

            # Create tree with proper dimensions
            tree = Tree(n_features, n_classes, n_outputs)

            # Try to set the tree state manually
            # Note: This is a complex operation and may not work for all sklearn versions
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
    elif value_dict.get("type") == "sklearn.preprocessing._label.LabelBinarizer":
        # Special handling for sklearn LabelBinarizer objects
        try:
            # Import LabelBinarizer class dynamically
            from sklearn.preprocessing import LabelBinarizer

            # Extract the serialized LabelBinarizer data
            binarizer_data = value_dict["value"]

            # Create a new LabelBinarizer object
            label_binarizer = LabelBinarizer()

            # Set the attributes from the serialized data
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
    elif value_dict.get("special_list"):
        # Handle lists with special serialization (e.g., lists of numpy arrays)
        try:
            deserialized_list = []
            for item_dict in value:
                if item_dict.get("item_type") == "numpy.ndarray":
                    # Reconstruct numpy array
                    dtype = item_dict.get("dtype", "float64")
                    shape = item_dict.get("shape")
                    arr = np.array(item_dict["item_value"], dtype=dtype)
                    if shape:
                        arr = arr.reshape(shape)
                    deserialized_list.append(arr)
                else:
                    # Regular item
                    deserialized_list.append(item_dict["item_value"])
            return deserialized_list
        except Exception as e:
            print(f"Warning: Could not deserialize special list: {e}")
            return None
        # Handle numpy scalar types
        if "int" in value_type:
            return np.int64(value)
        elif "float" in value_type:
            return np.float64(value)
        return value
    elif value in [
        "COMPLEX_OBJECT",
        "COMPLEX_LIST",
        "UNSERIALIZABLE",
        "COMPLEX_TREE_OBJECT",
    ]:
        # Skip complex objects that can't be easily reconstructed
        return None
    elif value is None:
        # Handle explicit None values
        return None
    else:
        return value


def deserialize_object(json_str: str) -> object:
    """Deserialize a JSON string back to a sklearn model."""
    obj_dict = json.loads(json_str)
    class_name = obj_dict["__class__"]
    module_name = obj_dict.get("__module__")

    # Ensure we have module information
    if not module_name:
        raise ValueError(
            f"No module information available for class '{class_name}'. "
            f"The JSON was likely created with an older version that didn't store module names."
        )

    # Import the module and get the class
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import {class_name} from {module_name}: {e}")

    # Extract initialization parameters and fitted attributes
    members = obj_dict.get("members", {})
    init_params = {}
    fitted_attrs = {}

    # Separate constructor parameters from fitted attributes
    # Most sklearn models have these common fitted attributes that end with _
    fitted_attr_patterns = [
        "_",
        "coef_",
        "intercept_",
        "feature_importances_",
        "n_features_in_",
        "classes_",
        "singular_",
        "rank_",
    ]

    for name, value_dict in members.items():
        # Fitted attributes: end with _, start with _, or start with n_
        is_fitted_attr = (
            any(name.endswith(pattern) for pattern in fitted_attr_patterns)
            or name.startswith("_")
            or name.startswith("n_")
        )

        if is_fitted_attr:
            fitted_attrs[name] = _deserialize_value(value_dict)
        else:
            # These are constructor parameters
            deserialized_value = _deserialize_value(value_dict)
            if deserialized_value is not None:
                init_params[name] = deserialized_value

    # Create instance with constructor parameters
    try:
        obj = cls(**init_params)
    except TypeError:
        # Fallback: create with no params and set all attributes manually
        obj = cls()
        for name, value in init_params.items():
            try:
                setattr(obj, name, value)
            except Exception as e:
                print(f"Warning: Could not set parameter '{name}': {e}")

    # Set fitted attributes (including None values which some models need)
    for name, value in fitted_attrs.items():
        try:
            setattr(obj, name, value)
        except Exception as e:
            print(f"Warning: Could not set fitted attribute '{name}': {e}")

    # Special post-processing for certain model types that need internal structure rebuilding
    if hasattr(obj, "_fit_X") and hasattr(obj, "_y") and obj._fit_X is not None:
        # KNeighbors models need to rebuild their tree structures
        if (
            "neighbors" in obj.__class__.__module__
            and "KNeighbors" in obj.__class__.__name__
        ):
            try:
                # Save current state
                original_fit_x = obj._fit_X.copy()
                original_y = obj._y.copy()

                # Re-fit to rebuild internal structures (tree, etc.)
                obj.fit(original_fit_x, original_y)
            except Exception as e:
                print(f"Warning: Could not rebuild KNeighbors internal structures: {e}")

    return obj
