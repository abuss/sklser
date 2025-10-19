"""Base classes for type handlers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTypeHandler(ABC):
    """Base class for all type handlers."""

    @abstractmethod
    def can_handle(self, value: Any) -> bool:
        """Check if this handler can handle the given value type."""
        pass

    def can_deserialize(self, value_dict: Dict[str, Any]) -> bool:
        """Check if this handler can deserialize the given value dict."""
        # Default implementation based on type field
        return False

    @abstractmethod
    def serialize(self, value: Any) -> Dict[str, Any]:
        """Serialize the value to a dictionary representation."""
        pass

    @abstractmethod
    def deserialize(self, value_dict: Dict[str, Any]) -> Any:
        """Deserialize the dictionary back to the original value."""
        pass


class BaseSerializationHandler(BaseTypeHandler):
    """Base class for serialization-specific handlers."""

    pass


class BaseDeserializationHandler(BaseTypeHandler):
    """Base class for deserialization-specific handlers."""

    pass
