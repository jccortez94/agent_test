"""Deterministic cart, order, and structured-memory use cases.

The conversational model may request these operations, but it does not own or
directly mutate their state.
"""

from __future__ import annotations

import math
import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from .database import Database
from .models import Cart, CartItem, FrequentItem, Order, OrderItem, Preference


class GroceryError(ValueError):
    """Base class for expected domain errors."""


class InvalidProductNameError(GroceryError):
    """Raised when a product name is empty."""


class InvalidQuantityError(GroceryError):
    """Raised when a quantity is not a finite positive number."""


class ItemNotFoundError(GroceryError):
    """Raised when a requested product is not in the active cart."""


class UnitConflictError(GroceryError):
    """Raised when the same product is added using a different unit."""


class EmptyCartError(GroceryError):
    """Raised when attempting to confirm an empty cart."""


class OrderNotFoundError(GroceryError):
    """Raised when an order does not exist for the requested user."""


class InvalidPreferenceError(GroceryError):
    """Raised when a structured preference has an empty component."""


class InvalidLimitError(GroceryError):
    """Raised when a result limit is not a positive integer."""


Clock = Callable[[], datetime]


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _product_identity(product_name: object) -> Tuple[str, str]:
    display_name = _clean_text(product_name)
    if not display_name:
        raise InvalidProductNameError("Product name must not be empty")
    return display_name.casefold(), display_name


def _normalize_user_id(user_id: object) -> str:
    normalized = _clean_text(user_id)
    if not normalized:
        raise GroceryError("User id must not be empty")
    return normalized


def _normalize_unit(unit: object) -> Optional[str]:
    if unit is None:
        return None
    normalized = _clean_text(unit)
    return normalized.casefold() or None


def _normalize_quantity(quantity: object) -> float:
    if isinstance(quantity, bool):
        raise InvalidQuantityError("Quantity must be a finite number greater than zero")
    try:
        normalized = float(quantity)
    except (TypeError, ValueError):
        raise InvalidQuantityError(
            "Quantity must be a finite number greater than zero"
        ) from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise InvalidQuantityError("Quantity must be a finite number greater than zero")
    return normalized


def _normalize_limit(limit: object) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise InvalidLimitError("Limit must be a positive integer")
    return limit


def _normalize_order_id(order_id: object) -> int:
    if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id <= 0:
        raise OrderNotFoundError(f"Order {order_id!r} was not found")
    return order_id


def _preference_identity(value: object, label: str) -> Tuple[str, str]:
    display_value = _clean_text(value)
    if not display_value:
        raise InvalidPreferenceError(f"Preference {label} must not be empty")
    return display_value.casefold(), display_value


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class GroceryService:
    """Application boundary for cart, order, and structured-memory operations."""

    def __init__(self, database: Database, clock: Clock = _system_clock) -> None:
        self.database = database
        self._clock = clock

    def get_cart(self, user_id: str) -> Cart:
        normalized_user_id = _normalize_user_id(user_id)
        now = self._now()
        with self.database.connect() as connection, connection:
            self._ensure_user(connection, normalized_user_id, now)
            cart_id = self._get_or_create_active_cart(
                connection, normalized_user_id, now
            )
            return self._load_cart(connection, cart_id, normalized_user_id)

    def add_item(
        self,
        user_id: str,
        product_name: str,
        quantity: object,
        unit: Optional[str] = None,
    ) -> Cart:
        normalized_user_id = _normalize_user_id(user_id)
        product_key, display_name = _product_identity(product_name)
        normalized_quantity = _normalize_quantity(quantity)
        normalized_unit = _normalize_unit(unit)
        now = self._now()

        with self.database.connect() as connection, connection:
            self._ensure_user(connection, normalized_user_id, now)
            cart_id = self._get_or_create_active_cart(
                connection, normalized_user_id, now
            )
            self._add_or_merge_cart_item(
                connection,
                cart_id,
                product_key,
                display_name,
                normalized_quantity,
                normalized_unit,
                now,
            )

            return self._load_cart(connection, cart_id, normalized_user_id)

    def set_item_quantity(
        self, user_id: str, product_name: str, quantity: object
    ) -> Cart:
        normalized_user_id = _normalize_user_id(user_id)
        product_key, _ = _product_identity(product_name)
        normalized_quantity = _normalize_quantity(quantity)
        now = self._now()

        with self.database.connect() as connection, connection:
            self._ensure_user(connection, normalized_user_id, now)
            cart_id = self._get_or_create_active_cart(
                connection, normalized_user_id, now
            )
            result = connection.execute(
                """
                UPDATE cart_items
                SET quantity = ?, updated_at = ?
                WHERE cart_id = ? AND product_key = ?
                """,
                (normalized_quantity, now, cart_id, product_key),
            )
            if result.rowcount == 0:
                raise ItemNotFoundError(f"Product {product_name!r} is not in the cart")
            return self._load_cart(connection, cart_id, normalized_user_id)

    def remove_item(self, user_id: str, product_name: str) -> Cart:
        normalized_user_id = _normalize_user_id(user_id)
        product_key, _ = _product_identity(product_name)
        now = self._now()

        with self.database.connect() as connection, connection:
            self._ensure_user(connection, normalized_user_id, now)
            cart_id = self._get_or_create_active_cart(
                connection, normalized_user_id, now
            )
            result = connection.execute(
                "DELETE FROM cart_items WHERE cart_id = ? AND product_key = ?",
                (cart_id, product_key),
            )
            if result.rowcount == 0:
                raise ItemNotFoundError(f"Product {product_name!r} is not in the cart")
            return self._load_cart(connection, cart_id, normalized_user_id)

    def confirm_order(self, user_id: str) -> Order:
        """Atomically snapshot the active cart and replace it with an empty one."""

        normalized_user_id = _normalize_user_id(user_id)
        now = self._now()

        with self.database.connect() as connection, connection:
            self._ensure_user(connection, normalized_user_id, now)
            cart_id = self._get_or_create_active_cart(
                connection, normalized_user_id, now
            )
            item_rows = connection.execute(
                """
                SELECT product_key, product_name, quantity, unit
                FROM cart_items
                WHERE cart_id = ?
                ORDER BY product_key
                """,
                (cart_id,),
            ).fetchall()
            if not item_rows:
                raise EmptyCartError("Cannot confirm an empty cart")

            cursor = connection.execute(
                "INSERT INTO orders (user_id, ordered_at) VALUES (?, ?)",
                (normalized_user_id, now),
            )
            order_id = int(cursor.lastrowid)
            for line_number, row in enumerate(item_rows, start=1):
                connection.execute(
                    """
                    INSERT INTO order_items (
                        order_id, line_number, product_key, product_name,
                        quantity, unit
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order_id,
                        line_number,
                        row["product_key"],
                        row["product_name"],
                        row["quantity"],
                        row["unit"],
                    ),
                )

            connection.execute(
                """
                UPDATE carts
                SET status = 'confirmed', confirmed_at = ?
                WHERE id = ?
                """,
                (now, cart_id),
            )
            connection.execute("DELETE FROM cart_items WHERE cart_id = ?", (cart_id,))
            connection.execute(
                """
                INSERT INTO carts (user_id, status, created_at)
                VALUES (?, 'active', ?)
                """,
                (normalized_user_id, now),
            )

            return Order(
                id=order_id,
                user_id=normalized_user_id,
                ordered_at=_parse_datetime(now),
                items=tuple(
                    OrderItem(
                        product_name=row["product_name"],
                        quantity=row["quantity"],
                        unit=row["unit"],
                    )
                    for row in item_rows
                ),
            )

    def list_orders(self, user_id: str) -> Tuple[Order, ...]:
        """Return confirmed orders newest first without creating user state."""

        normalized_user_id = _normalize_user_id(user_id)
        with self.database.connect() as connection:
            order_rows = connection.execute(
                """
                SELECT id, ordered_at
                FROM orders
                WHERE user_id = ?
                ORDER BY ordered_at DESC, id DESC
                """,
                (normalized_user_id,),
            ).fetchall()
            return tuple(
                self._load_order(connection, row, normalized_user_id)
                for row in order_rows
            )

    def get_recent_orders(
        self, user_id: str, limit: int = 5
    ) -> Tuple[Order, ...]:
        """Return a bounded set of confirmed orders, newest first."""

        normalized_user_id = _normalize_user_id(user_id)
        normalized_limit = _normalize_limit(limit)
        with self.database.connect() as connection:
            order_rows = connection.execute(
                """
                SELECT id, ordered_at
                FROM orders
                WHERE user_id = ?
                ORDER BY ordered_at DESC, id DESC
                LIMIT ?
                """,
                (normalized_user_id, normalized_limit),
            ).fetchall()
            return tuple(
                self._load_order(connection, row, normalized_user_id)
                for row in order_rows
            )

    def get_order(self, user_id: str, order_id: int) -> Order:
        """Return one order only when it belongs to the requested user."""

        normalized_user_id = _normalize_user_id(user_id)
        normalized_order_id = _normalize_order_id(order_id)
        with self.database.connect() as connection:
            order_row = connection.execute(
                """
                SELECT id, ordered_at
                FROM orders
                WHERE id = ? AND user_id = ?
                """,
                (normalized_order_id, normalized_user_id),
            ).fetchone()
            if order_row is None:
                raise OrderNotFoundError(
                    f"Order {normalized_order_id!r} was not found for this user"
                )
            return self._load_order(connection, order_row, normalized_user_id)

    def copy_order_to_cart(
        self,
        user_id: str,
        order_id: int,
        replace_existing: bool = True,
    ) -> Cart:
        """Copy an owned order into the active cart in one transaction.

        Replacing is the safe default for requests such as "the same order as
        last time". Passing ``replace_existing=False`` explicitly merges the
        quantities into the current cart instead.
        """

        normalized_user_id = _normalize_user_id(user_id)
        normalized_order_id = _normalize_order_id(order_id)
        if not isinstance(replace_existing, bool):
            raise GroceryError("replace_existing must be a boolean")
        now = self._now()

        with self.database.connect() as connection, connection:
            order_row = connection.execute(
                """
                SELECT id
                FROM orders
                WHERE id = ? AND user_id = ?
                """,
                (normalized_order_id, normalized_user_id),
            ).fetchone()
            if order_row is None:
                raise OrderNotFoundError(
                    f"Order {normalized_order_id!r} was not found for this user"
                )
            item_rows = connection.execute(
                """
                SELECT product_key, product_name, quantity, unit
                FROM order_items
                WHERE order_id = ?
                ORDER BY line_number
                """,
                (normalized_order_id,),
            ).fetchall()

            cart_id = self._get_or_create_active_cart(
                connection, normalized_user_id, now
            )
            if replace_existing:
                connection.execute(
                    "DELETE FROM cart_items WHERE cart_id = ?", (cart_id,)
                )

            for row in item_rows:
                self._add_or_merge_cart_item(
                    connection,
                    cart_id,
                    row["product_key"],
                    row["product_name"],
                    row["quantity"],
                    row["unit"],
                    now,
                )
            return self._load_cart(connection, cart_id, normalized_user_id)

    def get_frequent_items(
        self, user_id: str, limit: int = 10
    ) -> Tuple[FrequentItem, ...]:
        """Derive frequent products and usual quantities from confirmed orders."""

        normalized_user_id = _normalize_user_id(user_id)
        normalized_limit = _normalize_limit(limit)
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    oi.product_key,
                    oi.product_name,
                    oi.quantity,
                    oi.unit,
                    o.ordered_at
                FROM orders AS o
                JOIN order_items AS oi ON oi.order_id = o.id
                WHERE o.user_id = ?
                ORDER BY o.ordered_at DESC, o.id DESC, oi.line_number
                """,
                (normalized_user_id,),
            ).fetchall()

        aggregates = {}
        for row in rows:
            key = (row["product_key"], row["unit"])
            aggregate = aggregates.get(key)
            if aggregate is None:
                aggregate = {
                    "product_name": row["product_name"],
                    "unit": row["unit"],
                    "order_count": 0,
                    "total_quantity": 0.0,
                    "last_ordered_at": _parse_datetime(row["ordered_at"]),
                }
                aggregates[key] = aggregate
            aggregate["order_count"] += 1
            aggregate["total_quantity"] += row["quantity"]

        frequent_items = [
            FrequentItem(
                product_name=aggregate["product_name"],
                unit=aggregate["unit"],
                order_count=aggregate["order_count"],
                total_quantity=aggregate["total_quantity"],
                average_quantity=(
                    aggregate["total_quantity"] / aggregate["order_count"]
                ),
                last_ordered_at=aggregate["last_ordered_at"],
            )
            for aggregate in aggregates.values()
        ]
        frequent_items.sort(key=lambda item: item.product_name.casefold())
        frequent_items.sort(key=lambda item: item.last_ordered_at, reverse=True)
        frequent_items.sort(key=lambda item: item.order_count, reverse=True)
        return tuple(frequent_items[:normalized_limit])

    def remember_preference(
        self,
        user_id: str,
        subject: str,
        attribute: str,
        value: str,
    ) -> Preference:
        """Create or replace one explicit structured preference."""

        normalized_user_id = _normalize_user_id(user_id)
        subject_key, display_subject = _preference_identity(subject, "subject")
        attribute_key, display_attribute = _preference_identity(
            attribute, "attribute"
        )
        _, display_value = _preference_identity(value, "value")
        now = self._now()

        with self.database.connect() as connection, connection:
            self._ensure_user(connection, normalized_user_id, now)
            connection.execute(
                """
                INSERT INTO preferences (
                    user_id, subject_key, subject, attribute_key, attribute,
                    value, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, subject_key, attribute_key) DO UPDATE SET
                    subject = excluded.subject,
                    attribute = excluded.attribute,
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_user_id,
                    subject_key,
                    display_subject,
                    attribute_key,
                    display_attribute,
                    display_value,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT user_id, subject, attribute, value, created_at, updated_at
                FROM preferences
                WHERE user_id = ? AND subject_key = ? AND attribute_key = ?
                """,
                (normalized_user_id, subject_key, attribute_key),
            ).fetchone()
            return self._load_preference(row)

    def get_preference(
        self, user_id: str, subject: str, attribute: str
    ) -> Optional[Preference]:
        """Return one preference, or ``None`` when it has not been remembered."""

        normalized_user_id = _normalize_user_id(user_id)
        subject_key, _ = _preference_identity(subject, "subject")
        attribute_key, _ = _preference_identity(attribute, "attribute")
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, subject, attribute, value, created_at, updated_at
                FROM preferences
                WHERE user_id = ? AND subject_key = ? AND attribute_key = ?
                """,
                (normalized_user_id, subject_key, attribute_key),
            ).fetchone()
            return None if row is None else self._load_preference(row)

    def list_preferences(
        self, user_id: str, subject: Optional[str] = None
    ) -> Tuple[Preference, ...]:
        """Return all preferences, optionally restricted to one subject."""

        normalized_user_id = _normalize_user_id(user_id)
        parameters = [normalized_user_id]
        subject_clause = ""
        if subject is not None:
            subject_key, _ = _preference_identity(subject, "subject")
            subject_clause = " AND subject_key = ?"
            parameters.append(subject_key)

        with self.database.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT user_id, subject, attribute, value, created_at, updated_at
                FROM preferences
                WHERE user_id = ?{subject_clause}
                ORDER BY subject_key, attribute_key
                """,
                parameters,
            ).fetchall()
            return tuple(self._load_preference(row) for row in rows)

    def _now(self) -> str:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise GroceryError("Clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _ensure_user(
        connection: sqlite3.Connection, user_id: str, created_at: str
    ) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)",
            (user_id, created_at),
        )

    @staticmethod
    def _get_or_create_active_cart(
        connection: sqlite3.Connection, user_id: str, created_at: str
    ) -> int:
        row = connection.execute(
            "SELECT id FROM carts WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
        cursor = connection.execute(
            """
            INSERT INTO carts (user_id, status, created_at)
            VALUES (?, 'active', ?)
            """,
            (user_id, created_at),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _add_or_merge_cart_item(
        connection: sqlite3.Connection,
        cart_id: int,
        product_key: str,
        product_name: str,
        quantity: float,
        unit: Optional[str],
        updated_at: str,
    ) -> None:
        existing = connection.execute(
            """
            SELECT quantity, unit
            FROM cart_items
            WHERE cart_id = ? AND product_key = ?
            """,
            (cart_id, product_key),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO cart_items (
                    cart_id, product_key, product_name, quantity, unit,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cart_id,
                    product_key,
                    product_name,
                    quantity,
                    unit,
                    updated_at,
                    updated_at,
                ),
            )
            return

        if existing["unit"] != unit:
            raise UnitConflictError(
                f"{existing['unit']!r} and {unit!r} are different units for "
                "the same product"
            )
        combined_quantity = _normalize_quantity(existing["quantity"] + quantity)
        connection.execute(
            """
            UPDATE cart_items
            SET quantity = ?, updated_at = ?
            WHERE cart_id = ? AND product_key = ?
            """,
            (combined_quantity, updated_at, cart_id, product_key),
        )

    @staticmethod
    def _load_cart(
        connection: sqlite3.Connection, cart_id: int, user_id: str
    ) -> Cart:
        cart_row = connection.execute(
            "SELECT created_at FROM carts WHERE id = ?", (cart_id,)
        ).fetchone()
        item_rows = connection.execute(
            """
            SELECT product_name, quantity, unit
            FROM cart_items
            WHERE cart_id = ?
            ORDER BY product_key
            """,
            (cart_id,),
        ).fetchall()
        return Cart(
            id=cart_id,
            user_id=user_id,
            created_at=_parse_datetime(cart_row["created_at"]),
            items=tuple(
                CartItem(
                    product_name=row["product_name"],
                    quantity=row["quantity"],
                    unit=row["unit"],
                )
                for row in item_rows
            ),
        )

    @staticmethod
    def _load_order(
        connection: sqlite3.Connection,
        order_row: sqlite3.Row,
        user_id: str,
    ) -> Order:
        item_rows = connection.execute(
            """
            SELECT product_name, quantity, unit
            FROM order_items
            WHERE order_id = ?
            ORDER BY line_number
            """,
            (order_row["id"],),
        ).fetchall()
        return Order(
            id=order_row["id"],
            user_id=user_id,
            ordered_at=_parse_datetime(order_row["ordered_at"]),
            items=tuple(
                OrderItem(
                    product_name=row["product_name"],
                    quantity=row["quantity"],
                    unit=row["unit"],
                )
                for row in item_rows
            ),
        )

    @staticmethod
    def _load_preference(row: sqlite3.Row) -> Preference:
        return Preference(
            user_id=row["user_id"],
            subject=row["subject"],
            attribute=row["attribute"],
            value=row["value"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
        )
