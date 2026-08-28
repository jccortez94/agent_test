"""Immutable values returned by the grocery application service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


@dataclass(frozen=True)
class CartItem:
    product_name: str
    quantity: float
    unit: Optional[str] = None


@dataclass(frozen=True)
class Cart:
    id: int
    user_id: str
    created_at: datetime
    items: Tuple[CartItem, ...]


@dataclass(frozen=True)
class OrderItem:
    product_name: str
    quantity: float
    unit: Optional[str] = None


@dataclass(frozen=True)
class Order:
    id: int
    user_id: str
    ordered_at: datetime
    items: Tuple[OrderItem, ...]


@dataclass(frozen=True)
class FrequentItem:
    product_name: str
    unit: Optional[str]
    order_count: int
    total_quantity: float
    average_quantity: float
    last_ordered_at: datetime


@dataclass(frozen=True)
class Preference:
    user_id: str
    subject: str
    attribute: str
    value: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Conversation:
    id: int
    user_id: str
    created_at: datetime


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: datetime
