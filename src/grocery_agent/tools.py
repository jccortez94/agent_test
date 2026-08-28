"""Strict OpenAI function schemas and an allow-listed grocery dispatcher."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Optional, Set

from .models import Cart, FrequentItem, Order, Preference
from .service import GroceryError, GroceryService


class ToolInputError(ValueError):
    """Raised when a model-provided tool call does not match its contract."""


class UnknownToolError(ToolInputError):
    """Raised when a tool name is not on the application allow list."""


def _object_schema(properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _tool(
    name: str, description: str, properties: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": _object_schema(properties),
    }


TOOL_DEFINITIONS = (
    _tool(
        "add_cart_item",
        "Add a positive quantity of one product to the user's active cart.",
        {
            "product_name": {
                "type": "string",
                "description": "Specific product name, including a stated variant or brand.",
            },
            "quantity": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Amount to add.",
            },
            "unit": {
                "type": ["string", "null"],
                "description": "Unit such as liters or packages, or null when unitless.",
            },
        },
    ),
    _tool(
        "remove_cart_item",
        "Remove one named product completely from the active cart.",
        {
            "product_name": {
                "type": "string",
                "description": "Product name to remove.",
            }
        },
    ),
    _tool(
        "set_cart_item_quantity",
        "Replace the quantity of an existing product in the active cart.",
        {
            "product_name": {
                "type": "string",
                "description": "Existing product name.",
            },
            "quantity": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "New total quantity, not an amount to add.",
            },
        },
    ),
    _tool(
        "get_cart",
        "Read the authoritative contents of the user's active cart.",
        {},
    ),
    _tool(
        "confirm_order",
        "Confirm the non-empty active cart as an order. Use only after explicit user intent.",
        {},
    ),
    _tool(
        "get_recent_orders",
        "Read the user's recent confirmed orders, newest first.",
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of orders to return.",
            }
        },
    ),
    _tool(
        "copy_order_to_cart",
        "Copy an owned previous order into the active cart.",
        {
            "order_id": {
                "type": "integer",
                "minimum": 1,
                "description": "Order identifier obtained from get_recent_orders.",
            },
            "replace_existing": {
                "type": "boolean",
                "description": "True to replace the cart; false to merge quantities.",
            },
        },
    ),
    _tool(
        "get_frequent_items",
        (
            "Read products and usual quantities derived from confirmed orders. "
            "For an underspecified product, use this before clarifying or "
            "mutating; frequency is evidence, not an explicit preference."
        ),
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of products to return.",
            }
        },
    ),
    _tool(
        "get_preferences",
        (
            "Read explicit stored preferences, optionally for one product or "
            "subject. For an underspecified product, use this before clarifying "
            "or mutating; explicit preferences outrank purchase frequency."
        ),
        {
            "subject": {
                "type": ["string", "null"],
                "description": "Product or topic to filter by, or null for all preferences.",
            }
        },
    ),
    _tool(
        "remember_preference",
        "Store an explicit user preference as subject, attribute, and value.",
        {
            "subject": {
                "type": "string",
                "description": "Product or preference topic, such as coffee.",
            },
            "attribute": {
                "type": "string",
                "description": "Preference dimension, such as brand or type.",
            },
            "value": {
                "type": "string",
                "description": "Explicit preferred value, such as Lavazza or oat.",
            },
        },
    ),
)


class GroceryToolDispatcher:
    """Executes only known functions for one application-owned user id."""

    def __init__(self, service: GroceryService, user_id: str) -> None:
        self.service = service
        self.user_id = user_id
        self._handlers: Mapping[str, Callable[[Dict[str, Any]], Any]] = {
            "add_cart_item": self._add_cart_item,
            "remove_cart_item": self._remove_cart_item,
            "set_cart_item_quantity": self._set_cart_item_quantity,
            "get_cart": self._get_cart,
            "confirm_order": self._confirm_order,
            "get_recent_orders": self._get_recent_orders,
            "copy_order_to_cart": self._copy_order_to_cart,
            "get_frequent_items": self._get_frequent_items,
            "get_preferences": self._get_preferences,
            "remember_preference": self._remember_preference,
        }

    def execute(self, name: str, raw_arguments: str) -> Dict[str, Any]:
        """Return a JSON-compatible success or expected-error envelope."""

        handler = self._handlers.get(name)
        if handler is None:
            return self._error(UnknownToolError(f"Unknown tool: {name!r}"))
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ToolInputError("Tool arguments must be a JSON object")
            result = handler(arguments)
            return {"ok": True, "result": result}
        except json.JSONDecodeError as error:
            return self._error(ToolInputError(f"Invalid JSON arguments: {error.msg}"))
        except (ToolInputError, GroceryError) as error:
            return self._error(error)

    @staticmethod
    def _error(error: Exception) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    @staticmethod
    def _exact_arguments(arguments: Dict[str, Any], expected: Set[str]) -> None:
        actual = set(arguments)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)!r}")
            if extra:
                details.append(f"unexpected {sorted(extra)!r}")
            raise ToolInputError("Invalid tool arguments: " + ", ".join(details))

    @staticmethod
    def _string(arguments: Dict[str, Any], name: str) -> str:
        value = arguments[name]
        if not isinstance(value, str):
            raise ToolInputError(f"{name} must be a string")
        return value

    @staticmethod
    def _number(arguments: Dict[str, Any], name: str) -> float:
        value = arguments[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolInputError(f"{name} must be a number")
        return value

    @staticmethod
    def _integer(arguments: Dict[str, Any], name: str) -> int:
        value = arguments[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolInputError(f"{name} must be an integer")
        return value

    @classmethod
    def _limit(cls, arguments: Dict[str, Any]) -> int:
        value = cls._integer(arguments, "limit")
        if not 1 <= value <= 20:
            raise ToolInputError("limit must be between 1 and 20")
        return value

    def _add_cart_item(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._exact_arguments(arguments, {"product_name", "quantity", "unit"})
        unit = arguments["unit"]
        if unit is not None and not isinstance(unit, str):
            raise ToolInputError("unit must be a string or null")
        cart = self.service.add_item(
            self.user_id,
            self._string(arguments, "product_name"),
            self._number(arguments, "quantity"),
            unit,
        )
        return self._cart(cart)

    def _remove_cart_item(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._exact_arguments(arguments, {"product_name"})
        cart = self.service.remove_item(
            self.user_id, self._string(arguments, "product_name")
        )
        return self._cart(cart)

    def _set_cart_item_quantity(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._exact_arguments(arguments, {"product_name", "quantity"})
        cart = self.service.set_item_quantity(
            self.user_id,
            self._string(arguments, "product_name"),
            self._number(arguments, "quantity"),
        )
        return self._cart(cart)

    def _get_cart(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._exact_arguments(arguments, set())
        return self._cart(self.service.get_cart(self.user_id))

    def _confirm_order(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._exact_arguments(arguments, set())
        return self._order(self.service.confirm_order(self.user_id))

    def _get_recent_orders(self, arguments: Dict[str, Any]) -> Any:
        self._exact_arguments(arguments, {"limit"})
        orders = self.service.get_recent_orders(
            self.user_id, self._limit(arguments)
        )
        return [self._order(order) for order in orders]

    def _copy_order_to_cart(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._exact_arguments(arguments, {"order_id", "replace_existing"})
        replace_existing = arguments["replace_existing"]
        if not isinstance(replace_existing, bool):
            raise ToolInputError("replace_existing must be a boolean")
        cart = self.service.copy_order_to_cart(
            self.user_id,
            self._integer(arguments, "order_id"),
            replace_existing,
        )
        return self._cart(cart)

    def _get_frequent_items(self, arguments: Dict[str, Any]) -> Any:
        self._exact_arguments(arguments, {"limit"})
        items = self.service.get_frequent_items(
            self.user_id, self._limit(arguments)
        )
        return [self._frequent_item(item) for item in items]

    def _get_preferences(self, arguments: Dict[str, Any]) -> Any:
        self._exact_arguments(arguments, {"subject"})
        subject = arguments["subject"]
        if subject is not None and not isinstance(subject, str):
            raise ToolInputError("subject must be a string or null")
        preferences = self.service.list_preferences(self.user_id, subject)
        return [self._preference(preference) for preference in preferences]

    def _remember_preference(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._exact_arguments(arguments, {"subject", "attribute", "value"})
        preference = self.service.remember_preference(
            self.user_id,
            self._string(arguments, "subject"),
            self._string(arguments, "attribute"),
            self._string(arguments, "value"),
        )
        return self._preference(preference)

    @staticmethod
    def _cart(cart: Cart) -> Dict[str, Any]:
        return {
            "cart_id": cart.id,
            "items": [
                {
                    "product_name": item.product_name,
                    "quantity": item.quantity,
                    "unit": item.unit,
                }
                for item in cart.items
            ],
        }

    @staticmethod
    def _order(order: Order) -> Dict[str, Any]:
        return {
            "order_id": order.id,
            "ordered_at": order.ordered_at.isoformat(),
            "items": [
                {
                    "product_name": item.product_name,
                    "quantity": item.quantity,
                    "unit": item.unit,
                }
                for item in order.items
            ],
        }

    @staticmethod
    def _frequent_item(item: FrequentItem) -> Dict[str, Any]:
        return {
            "product_name": item.product_name,
            "unit": item.unit,
            "order_count": item.order_count,
            "total_quantity": item.total_quantity,
            "average_quantity": item.average_quantity,
            "last_ordered_at": item.last_ordered_at.isoformat(),
        }

    @staticmethod
    def _preference(preference: Preference) -> Dict[str, Any]:
        return {
            "subject": preference.subject,
            "attribute": preference.attribute,
            "value": preference.value,
            "created_at": preference.created_at.isoformat(),
            "updated_at": preference.updated_at.isoformat(),
        }
