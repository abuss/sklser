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


# Modular Type Handlers
class TypeSerializer:
    """Manages all type handlers for serialization/deserialization."""

    def __init__(self):
        # Initialize all handlers in order of priority
        self.handlers = [
            NumpyArrayHandler(),
            NumpyScalarHandler(),
            PipelineHandler(),
            FeatureUnionHandler(),
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
        # Find the right handler based on type
        for handler in self.handlers:
            if handler.can_deserialize(value_dict):
                return handler.deserialize(value_dict)

        # Fallback
        return value_dict.get("value")


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
                "missing_go_to_left": getattr(value, "missing_go_to_left", None),
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
        # Tree objects cannot be reconstructed directly due to read-only attributes
        # Instead, we return a marker that tells the deserializer to reconstruct at model level
        return {
            "_tree_data": value_dict["value"],
            "_tree_marker": True,
            "_original_type": "sklearn.tree._tree.Tree",
        }


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
                value_dict.get("pipeline", False),
                value_dict.get("feature_union", False),
                value_dict.get("type") == "numpy.ndarray",
            ]
        )

    def serialize(self, value: Any) -> Dict[str, Any]:
        return {"value": value, "type": str(type(value))}

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        return value_dict["value"]


class PipelineHandler:
    """Handler for sklearn Pipeline objects."""

    def can_handle(self, value: Any) -> bool:
        return hasattr(value, "__class__") and "sklearn.pipeline.Pipeline" in str(type(value))

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("type") == "sklearn.pipeline.Pipeline"

    def serialize(self, value: Any) -> Dict[str, Any]:
        try:
            # Serialize each step in the pipeline
            serialized_steps = []
            for step_name, estimator in value.steps:
                # Use the main serialize_json function for each estimator
                # but parse it back to get the dict structure
                estimator_json = serialize_json(estimator)
                estimator_dict = json.loads(estimator_json)
                serialized_steps.append({
                    "name": step_name,
                    "estimator": estimator_dict
                })
            
            return {
                "value": {
                    "steps": serialized_steps,
                    "memory": getattr(value, "memory", None),
                    "verbose": getattr(value, "verbose", False)
                },
                "type": "sklearn.pipeline.Pipeline",
                "pipeline": True,
            }
        except Exception as e:
            return {
                "value": "COMPLEX_PIPELINE",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        try:
            # Dynamic import to avoid issues
            pipeline_module = importlib.import_module("sklearn.pipeline")
            Pipeline = getattr(pipeline_module, "Pipeline")

            pipeline_data = value_dict["value"]
            
            # Deserialize each step
            steps = []
            for step_data in pipeline_data["steps"]:
                step_name = step_data["name"]
                # Use the main deserialize_object function for each estimator
                # but convert it back to JSON string first
                estimator_json = json.dumps(step_data["estimator"])
                estimator = deserialize_object(estimator_json)
                steps.append((step_name, estimator))

            # Create the pipeline with the deserialized steps
            pipeline = Pipeline(
                steps=steps,
                memory=pipeline_data.get("memory"),
                verbose=pipeline_data.get("verbose", False)
            )

            return pipeline
        except Exception as e:
            print(f"Warning: Could not reconstruct Pipeline object: {e}")
            return None


class FeatureUnionHandler:
    """Handler for sklearn FeatureUnion objects."""

    def can_handle(self, value: Any) -> bool:
        return hasattr(value, "__class__") and "sklearn.pipeline.FeatureUnion" in str(type(value))

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        return value_dict.get("type") == "sklearn.pipeline.FeatureUnion"

    def serialize(self, value: Any) -> Dict[str, Any]:
        try:
            # Serialize each transformer in the FeatureUnion
            serialized_transformers = []
            for transformer_name, transformer in value.transformer_list:
                # Use the main serialize_json function for each transformer
                transformer_json = serialize_json(transformer)
                transformer_dict = json.loads(transformer_json)
                serialized_transformers.append({
                    "name": transformer_name,
                    "transformer": transformer_dict
                })
            
            return {
                "value": {
                    "transformer_list": serialized_transformers,
                    "n_jobs": getattr(value, "n_jobs", None),
                    "transformer_weights": getattr(value, "transformer_weights", None),
                    "verbose": getattr(value, "verbose", False)
                },
                "type": "sklearn.pipeline.FeatureUnion",
                "feature_union": True,
            }
        except Exception as e:
            return {
                "value": "COMPLEX_FEATUREUNION",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        try:
            # Dynamic import to avoid issues
            pipeline_module = importlib.import_module("sklearn.pipeline")
            FeatureUnion = getattr(pipeline_module, "FeatureUnion")

            featureunion_data = value_dict["value"]
            
            # Deserialize each transformer
            transformer_list = []
            for transformer_data in featureunion_data["transformer_list"]:
                transformer_name = transformer_data["name"]
                # Use the main deserialize_object function for each transformer
                transformer_json = json.dumps(transformer_data["transformer"])
                transformer = deserialize_object(transformer_json)
                transformer_list.append((transformer_name, transformer))

            # Create the FeatureUnion with the deserialized transformers
            feature_union = FeatureUnion(
                transformer_list=transformer_list,
                n_jobs=featureunion_data.get("n_jobs"),
                transformer_weights=featureunion_data.get("transformer_weights"),
                verbose=featureunion_data.get("verbose", False)
            )

            return feature_union
        except Exception as e:
            print(f"Warning: Could not reconstruct FeatureUnion object: {e}")
            return None


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
            "COMPLEX_PIPELINE",
            "COMPLEX_FEATUREUNION",
        ]:
            return None
        return value_dict["value"]


# Create a global instance
_type_serializer = TypeSerializer()


def _serialize_value(value: Any) -> Dict[str, Any]:
    """Serialize a single value with proper type handling."""
    return _type_serializer.serialize_value(value)


def _deserialize_value(value_dict: Dict[str, Any]) -> Any:
    """Deserialize a single value with proper type reconstruction."""
    return _type_serializer.deserialize_value(value_dict)


def serialize_json(obj: object) -> str:
    """Serialize a sklearn model to JSON string."""
    
    # Special handling for Pipeline and FeatureUnion objects
    if "sklearn.pipeline.Pipeline" in str(type(obj)):
        # Use PipelineHandler for serialization
        for handler in _type_serializer.handlers:
            if isinstance(handler, PipelineHandler):
                serialized_data = handler.serialize(obj)
                # Add class information for top-level object
                serialized_data["__class__"] = obj.__class__.__name__
                serialized_data["__module__"] = obj.__class__.__module__
                return json.dumps(serialized_data, indent=4)
    
    if "sklearn.pipeline.FeatureUnion" in str(type(obj)):
        # Use FeatureUnionHandler for serialization
        for handler in _type_serializer.handlers:
            if isinstance(handler, FeatureUnionHandler):
                serialized_data = handler.serialize(obj)
                # Add class information for top-level object
                serialized_data["__class__"] = obj.__class__.__name__
                serialized_data["__module__"] = obj.__class__.__module__
                return json.dumps(serialized_data, indent=4)
    
    # Regular sklearn object serialization
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
        "_max_components",
        "_n_features_out",
    }

    # Check if this is a tree-based model that needs special handling
    is_tree_model = (
        hasattr(obj, "tree_")
        or "tree" in obj.__class__.__module__
        or "Tree" in obj.__class__.__name__
    )

    # For tree models, we'll store a flag to indicate we need training data for reconstruction
    if is_tree_model:
        out_dict["__needs_refitting__"] = True

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


def deserialize_object(json_str: str) -> object:
    """Deserialize a JSON string back to a sklearn model."""
    obj_dict = json.loads(json_str)
    class_name = obj_dict["__class__"]
    module_name = obj_dict.get("__module__")
    
    # Special handling for Pipeline and FeatureUnion objects
    if obj_dict.get("pipeline", False) and class_name == "Pipeline":
        # Use PipelineHandler for deserialization
        for handler in _type_serializer.handlers:
            if isinstance(handler, PipelineHandler):
                return handler.deserialize(obj_dict)
    
    if obj_dict.get("feature_union", False) and class_name == "FeatureUnion":
        # Use FeatureUnionHandler for deserialization
        for handler in _type_serializer.handlers:
            if isinstance(handler, FeatureUnionHandler):
                return handler.deserialize(obj_dict)
    
    # Regular sklearn object deserialization
    needs_refitting = obj_dict.get("__needs_refitting__", False)

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
            # Special handling for tree objects that cannot be directly reconstructed
            if (
                isinstance(value, dict)
                and value.get("_tree_marker")
                and needs_refitting
            ):
                # For tree-based models that need refitting, skip tree_ for now
                if name == "tree_":
                    continue

            # Skip computed properties that don't have setters
            if name in [
                "feature_importances_",
                "n_support_",
                "probA_",
                "probB_",
                "sparse_coef_",
                "_n_features_out",
            ]:
                continue

            setattr(obj, name, value)
        except Exception as e:
            print(f"Warning: Could not set fitted attribute '{name}': {e}")

    # Special handling for models that need refitting to restore internal structures
    if needs_refitting:
        # Check if we have both training data and labels stored
        if (
            hasattr(obj, "_fit_X")
            and hasattr(obj, "_y")
            and obj._fit_X is not None
            and obj._y is not None
        ):
            try:
                # Save current state
                original_fit_x = obj._fit_X.copy()
                original_y = obj._y.copy()

                # Save other important fitted attributes that won't be lost during refitting
                saved_attrs = {}
                for attr_name in ["n_features_in_", "feature_names_in_", "classes_"]:
                    if hasattr(obj, attr_name):
                        saved_attrs[attr_name] = getattr(obj, attr_name)

                # Re-fit to rebuild internal structures (tree, etc.)
                obj.fit(original_fit_x, original_y)

                # Restore saved attributes that might have been overwritten
                for attr_name, attr_value in saved_attrs.items():
                    try:
                        setattr(obj, attr_name, attr_value)
                    except:
                        pass

            except Exception as e:
                print(f"Warning: Could not rebuild model internal structures: {e}")
        else:
            print(
                f"Warning: {class_name} requires training data to fully reconstruct tree structure"
            )

    # Handle KNeighbors models specifically (they also need rebuilding)
    elif hasattr(obj, "_fit_X") and hasattr(obj, "_y") and obj._fit_X is not None:
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
