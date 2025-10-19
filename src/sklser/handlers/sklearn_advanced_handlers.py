"""Advanced sklearn handlers for complex transformation objects."""

from typing import Any

from .base import BaseTypeHandler


class PipelineHandler(BaseTypeHandler):
    """Handler for sklearn Pipeline objects."""

    def can_handle(self, value: Any) -> bool:
        return "sklearn.pipeline.Pipeline" in str(type(value))

    def can_deserialize(self, value_dict) -> bool:
        return value_dict.get("pipeline", False)

    def serialize(self, value: Any):
        try:
            # Import the global serializer to handle recursion
            from .. import _type_serializer

            steps_data = []
            for step_name, step_obj in value.steps:
                # Use global serializer for recursion
                step_serialized = _type_serializer.serialize_value(step_obj)
                steps_data.append((step_name, step_serialized))

            return {
                "steps": steps_data,
                "memory": None,  # Memory objects are not serializable
                "verbose": value.verbose,
                "pipeline": True,
                "__class__": "Pipeline",
                "__module__": "sklearn.pipeline",
            }
        except Exception as e:
            return {
                "value": "COMPLEX_PIPELINE",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict):
        try:
            from sklearn.pipeline import Pipeline

            # Import the global serializer to handle recursion
            from .. import _type_serializer

            steps = []
            for step_name, step_data in value_dict["steps"]:
                # Use global serializer for recursion
                step_obj = _type_serializer.deserialize_value(step_data)
                if step_obj is not None:
                    steps.append((step_name, step_obj))

            pipeline = Pipeline(
                steps=steps,
                memory=value_dict.get("memory"),
                verbose=value_dict.get("verbose", False),
            )

            return pipeline
        except Exception as e:
            print(f"Warning: Could not deserialize Pipeline: {e}")
            return None


class FeatureUnionHandler(BaseTypeHandler):
    """Handler for sklearn FeatureUnion objects."""

    def can_handle(self, value: Any) -> bool:
        return "sklearn.pipeline.FeatureUnion" in str(type(value))

    def can_deserialize(self, value_dict) -> bool:
        return value_dict.get("feature_union", False)

    def serialize(self, value: Any):
        try:
            # Import the global serializer to handle recursion
            from .. import _type_serializer

            transformer_list = []
            for transformer_name, transformer_obj in value.transformer_list:
                transformer_serialized = _type_serializer.serialize_value(
                    transformer_obj
                )
                transformer_list.append((transformer_name, transformer_serialized))

            fitted_data = {
                "transformer_list": transformer_list,
                "n_jobs": value.n_jobs,
                "transformer_weights": value.transformer_weights,
                "verbose": value.verbose,
                "feature_union": True,
                "__class__": "FeatureUnion",
                "__module__": "sklearn.pipeline",
            }

            # Handle fitted attributes if the FeatureUnion is fitted
            try:
                if hasattr(value, "_validate_transformers"):
                    fitted_data["_transformers"] = _type_serializer.serialize_value(
                        value._transformers
                    )
                if hasattr(value, "transformer_") and value.transformer_ is not None:
                    fitted_data["transformer_"] = _type_serializer.serialize_value(
                        value.transformer_
                    )
            except Exception:
                pass

            return fitted_data
        except Exception as e:
            return {
                "value": "COMPLEX_FEATUREUNION",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict):
        try:
            from sklearn.pipeline import FeatureUnion

            # Import the global serializer to handle recursion
            from .. import _type_serializer

            transformer_list = []
            for transformer_name, transformer_data in value_dict["transformer_list"]:
                transformer_obj = _type_serializer.deserialize_value(transformer_data)
                if transformer_obj is not None:
                    transformer_list.append((transformer_name, transformer_obj))

            feature_union = FeatureUnion(
                transformer_list=transformer_list,
                n_jobs=value_dict.get("n_jobs", None),
                transformer_weights=value_dict.get("transformer_weights", None),
                verbose=value_dict.get("verbose", False),
            )

            # Restore fitted attributes if present
            if "_transformers" in value_dict:
                setattr(
                    feature_union,
                    "_transformers",
                    _type_serializer.deserialize_value(value_dict["_transformers"]),
                )
            if "transformer_" in value_dict:
                setattr(
                    feature_union,
                    "transformer_",
                    _type_serializer.deserialize_value(value_dict["transformer_"]),
                )

            return feature_union
        except Exception as e:
            print(f"Warning: Could not deserialize FeatureUnion: {e}")
            return None


class ColumnTransformerHandler(BaseTypeHandler):
    """Handler for sklearn ColumnTransformer objects."""

    def can_handle(self, value: Any) -> bool:
        return "sklearn.compose._column_transformer.ColumnTransformer" in str(
            type(value)
        )

    def can_deserialize(self, value_dict) -> bool:
        return value_dict.get("column_transformer", False)

    def serialize(self, value: Any):
        try:
            # Import the global serializer to handle recursion
            from .. import _type_serializer

            # Serialize the transformers list
            transformers_list = []
            for transformer_name, transformer_obj, columns in value.transformers:
                transformer_serialized = _type_serializer.serialize_value(
                    transformer_obj
                )

                # Handle columns serialization
                columns_serialized = _type_serializer.serialize_value(columns)

                transformers_list.append(
                    (transformer_name, transformer_serialized, columns_serialized)
                )

            fitted_data = {
                "transformers": transformers_list,
                "remainder": value.remainder,
                "sparse_threshold": value.sparse_threshold,
                "n_jobs": value.n_jobs,
                "transformer_weights": value.transformer_weights,
                "verbose": value.verbose,
                "column_transformer": True,
                "__class__": "ColumnTransformer",
                "__module__": "sklearn.compose",
            }

            # Handle fitted attributes if the ColumnTransformer is fitted
            try:
                if hasattr(value, "_transformers") and value._transformers is not None:
                    fitted_data["_transformers"] = _type_serializer.serialize_value(
                        value._transformers
                    )
                if hasattr(value, "transformers_") and value.transformers_ is not None:
                    fitted_data["transformers_"] = _type_serializer.serialize_value(
                        value.transformers_
                    )
                if hasattr(value, "_columns") and value._columns is not None:
                    fitted_data["_columns"] = _type_serializer.serialize_value(
                        value._columns
                    )

                if hasattr(value, "_remainder") and value._remainder is not None:
                    remainder_val = getattr(value, "_remainder")
                    fitted_data["_remainder"] = _type_serializer.serialize_value(
                        remainder_val
                    )

                fitted_data["output_indices_"] = _type_serializer.serialize_value(
                    value.output_indices_
                )

                # Add sparse_output_ if it exists (fitted ColumnTransformers have this)
                if hasattr(value, "sparse_output_"):
                    fitted_data["sparse_output_"] = _type_serializer.serialize_value(
                        value.sparse_output_
                    )

                if (
                    hasattr(value, "feature_names_in_")
                    and value.feature_names_in_ is not None
                ):
                    fitted_data["feature_names_in_"] = _type_serializer.serialize_value(
                        value.feature_names_in_
                    )

            except Exception as e:
                pass

            return fitted_data
        except Exception as e:
            return {
                "value": "COMPLEX_COLUMNTRANSFORMER",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict):
        try:
            from sklearn.compose import ColumnTransformer

            # Import the global serializer to handle recursion
            from .. import _type_serializer

            # Deserialize the transformers list
            transformers_list = []
            for transformer_name, transformer_data, columns_data in value_dict[
                "transformers"
            ]:
                transformer_obj = _type_serializer.deserialize_value(transformer_data)

                # Handle columns deserialization
                columns = _type_serializer.deserialize_value(columns_data)

                if transformer_obj is not None:
                    transformers_list.append(
                        (transformer_name, transformer_obj, columns)
                    )

            column_transformer = ColumnTransformer(
                transformers=transformers_list,
                remainder=value_dict.get("remainder", "drop"),
                sparse_threshold=value_dict.get("sparse_threshold", 0.3),
                n_jobs=value_dict.get("n_jobs", None),
                transformer_weights=value_dict.get("transformer_weights", None),
                verbose=value_dict.get("verbose", False),
            )

            # Restore fitted attributes if present
            fitted_attrs = [
                "_transformers",
                "transformers_",
                "_columns",
                "_remainder",
                "output_indices_",
                "sparse_output_",
                "feature_names_in_",
            ]
            for attr_name in fitted_attrs:
                if attr_name in value_dict:
                    try:
                        setattr(
                            column_transformer,
                            attr_name,
                            _type_serializer.deserialize_value(value_dict[attr_name]),
                        )
                    except Exception:
                        continue
                elif f"{attr_name}_" in value_dict:
                    setattr(
                        column_transformer,
                        attr_name,
                        _type_serializer.deserialize_value(value_dict[attr_name]),
                    )

            return column_transformer
        except Exception as e:
            print(f"Warning: Could not deserialize ColumnTransformer: {e}")
            return None


class FunctionTransformerHandler(BaseTypeHandler):
    """Handler for sklearn FunctionTransformer objects."""

    def can_handle(self, value: Any) -> bool:
        return "sklearn.preprocessing._function_transformer.FunctionTransformer" in str(
            type(value)
        )

    def can_deserialize(self, value_dict) -> bool:
        return value_dict.get("function_transformer", False)

    def serialize(self, value: Any):
        try:
            from sklearn.preprocessing import FunctionTransformer

            func_name = None
            inverse_func_name = None

            # Try to get function names
            if value.func is not None:
                try:
                    func_name = value.func.__name__
                except AttributeError:
                    func_name = str(value.func)

            if value.inverse_func is not None:
                try:
                    inverse_func_name = value.inverse_func.__name__
                except AttributeError:
                    inverse_func_name = str(value.inverse_func)

            return {
                "func": func_name,
                "inverse_func": inverse_func_name,
                "validate": value.validate,
                "accept_sparse": value.accept_sparse,
                "check_inverse": value.check_inverse,
                "feature_names_out": value.feature_names_out,
                "kw_args": value.kw_args,
                "inv_kw_args": value.inv_kw_args,
                "function_transformer": True,
                "__class__": "FunctionTransformer",
                "__module__": "sklearn.preprocessing",
            }
        except Exception as e:
            return {
                "value": "COMPLEX_FUNCTIONTRANSFORMER",
                "type": str(type(value)),
                "error": str(e),
            }

    def deserialize(self, value_dict):
        try:
            from sklearn.preprocessing import FunctionTransformer
            import numpy as np

            # Try to resolve function by name (basic functions only)
            func = None
            inverse_func = None

            func_name = value_dict.get("func")
            if func_name == "log1p":
                func = np.log1p
            elif func_name == "sqrt":
                func = np.sqrt
            # Add other basic functions as needed

            inverse_func_name = value_dict.get("inverse_func")
            if inverse_func_name == "expm1":
                inverse_func = np.expm1
            elif inverse_func_name == "square":
                inverse_func = np.square
            # Add other basic inverse functions as needed

            function_transformer = FunctionTransformer(
                func=func,
                inverse_func=inverse_func,
                validate=value_dict.get("validate", False),
                accept_sparse=value_dict.get("accept_sparse", False),
                check_inverse=value_dict.get("check_inverse", True),
                feature_names_out=value_dict.get("feature_names_out", None),
                kw_args=value_dict.get("kw_args", None),
                inv_kw_args=value_dict.get("inv_kw_args", None),
            )

            return function_transformer
        except Exception as e:
            print(f"Warning: Could not deserialize FunctionTransformer: {e}")
            return None
