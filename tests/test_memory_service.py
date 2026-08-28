from datetime import datetime, timedelta, timezone

import pytest

from grocery_agent import (
    Database,
    GroceryService,
    InvalidLimitError,
    InvalidPreferenceError,
    OrderNotFoundError,
    UnitConflictError,
)


class MutableClock:
    def __init__(self):
        self.current = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.current

    def advance(self, days=1):
        self.current += timedelta(days=days)


@pytest.fixture
def memory_service(tmp_path):
    database = Database(tmp_path / "memory.sqlite3")
    database.initialize()
    clock = MutableClock()
    return GroceryService(database, clock=clock), database, clock


def test_recent_orders_are_bounded_and_newest_first(memory_service):
    service, _, clock = memory_service
    service.add_item("demo-user", "apples", 4)
    first_order = service.confirm_order("demo-user")

    clock.advance()
    service.add_item("demo-user", "coffee", 1)
    second_order = service.confirm_order("demo-user")

    assert service.get_recent_orders("demo-user", limit=1) == (second_order,)
    assert service.get_recent_orders("demo-user", limit=2) == (
        second_order,
        first_order,
    )
    assert service.get_order("demo-user", first_order.id) == first_order


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", None])
def test_invalid_result_limits_are_rejected(memory_service, limit):
    service, _, _ = memory_service

    with pytest.raises(InvalidLimitError):
        service.get_recent_orders("demo-user", limit=limit)

    with pytest.raises(InvalidLimitError):
        service.get_frequent_items("demo-user", limit=limit)


def test_orders_cannot_be_read_or_copied_across_users(memory_service):
    service, _, _ = memory_service
    service.add_item("alice", "apples", 4)
    alice_order = service.confirm_order("alice")

    with pytest.raises(OrderNotFoundError):
        service.get_order("bob", alice_order.id)

    with pytest.raises(OrderNotFoundError):
        service.copy_order_to_cart("bob", alice_order.id)

    assert service.get_cart("bob").items == ()


def test_copy_order_replaces_the_active_cart_by_default(memory_service):
    service, _, _ = memory_service
    service.add_item("demo-user", "apples", 4)
    service.add_item("demo-user", "coffee", 1)
    order = service.confirm_order("demo-user")
    service.add_item("demo-user", "bananas", 6)

    restored_cart = service.copy_order_to_cart("demo-user", order.id)

    assert [item.product_name for item in restored_cart.items] == [
        "apples",
        "coffee",
    ]
    assert service.list_orders("demo-user") == (order,)


def test_copy_order_can_explicitly_merge_quantities(memory_service):
    service, _, _ = memory_service
    service.add_item("demo-user", "milk", 2, "liters")
    service.add_item("demo-user", "apples", 4)
    order = service.confirm_order("demo-user")
    service.add_item("demo-user", "milk", 1, "liters")
    service.add_item("demo-user", "bananas", 6)

    merged_cart = service.copy_order_to_cart(
        "demo-user", order.id, replace_existing=False
    )

    quantities = {item.product_name: item.quantity for item in merged_cart.items}
    assert quantities == {"apples": 4, "bananas": 6, "milk": 3}


def test_failed_order_merge_rolls_back_every_item(memory_service):
    service, _, _ = memory_service
    service.add_item("demo-user", "apples", 4)
    service.add_item("demo-user", "milk", 2, "liters")
    order = service.confirm_order("demo-user")
    service.add_item("demo-user", "milk", 1, "bottles")
    service.add_item("demo-user", "pears", 3)

    with pytest.raises(UnitConflictError):
        service.copy_order_to_cart(
            "demo-user", order.id, replace_existing=False
        )

    current_items = service.get_cart("demo-user").items
    assert [item.product_name for item in current_items] == ["milk", "pears"]
    assert current_items[0].unit == "bottles"


def test_frequent_items_are_derived_from_confirmed_orders(memory_service):
    service, _, clock = memory_service
    service.add_item("demo-user", "apples", 2)
    service.add_item("demo-user", "coffee", 1)
    service.confirm_order("demo-user")

    clock.advance()
    service.add_item("demo-user", "apples", 4)
    service.add_item("demo-user", "bananas", 6)
    service.confirm_order("demo-user")

    clock.advance()
    service.add_item("demo-user", "apples", 3)
    service.add_item("demo-user", "coffee", 3)
    service.confirm_order("demo-user")

    frequent_items = service.get_frequent_items("demo-user")

    assert [item.product_name for item in frequent_items] == [
        "apples",
        "coffee",
        "bananas",
    ]
    apples = frequent_items[0]
    assert apples.order_count == 3
    assert apples.total_quantity == 9
    assert apples.average_quantity == 3
    assert apples.last_ordered_at == clock.current
    assert service.get_frequent_items("demo-user", limit=1) == (apples,)
    assert service.get_frequent_items("new-user") == ()


def test_preference_updates_are_case_insensitive_and_preserve_creation_time(
    memory_service,
):
    service, _, clock = memory_service
    original = service.remember_preference(
        "demo-user", "Coffee", "Brand", "Lavazza"
    )

    clock.advance()
    updated = service.remember_preference(
        "demo-user", " coffee ", "BRAND", "Illy"
    )

    assert updated.value == "Illy"
    assert updated.created_at == original.created_at
    assert updated.updated_at == clock.current
    assert service.get_preference("demo-user", "COFFEE", "brand") == updated
    assert service.list_preferences("demo-user") == (updated,)


def test_preferences_can_be_listed_by_subject_and_are_isolated_by_user(
    memory_service,
):
    service, _, _ = memory_service
    brand = service.remember_preference(
        "alice", "coffee", "brand", "Lavazza"
    )
    roast = service.remember_preference("alice", "coffee", "roast", "dark")
    service.remember_preference("alice", "milk", "type", "oat")
    service.remember_preference("bob", "coffee", "brand", "Illy")

    assert service.list_preferences("alice", subject="COFFEE") == (brand, roast)
    assert len(service.list_preferences("alice")) == 3
    assert len(service.list_preferences("bob")) == 1
    assert service.get_preference("alice", "tea", "brand") is None


@pytest.mark.parametrize(
    "subject, attribute, value",
    [
        ("", "brand", "Lavazza"),
        ("coffee", "", "Lavazza"),
        ("coffee", "brand", ""),
        (None, "brand", "Lavazza"),
    ],
)
def test_empty_preference_components_are_rejected(
    memory_service, subject, attribute, value
):
    service, _, _ = memory_service

    with pytest.raises(InvalidPreferenceError):
        service.remember_preference(
            "demo-user", subject=subject, attribute=attribute, value=value
        )

    assert service.list_preferences("demo-user") == ()


def test_preferences_survive_reinitialization_and_service_restart(memory_service):
    service, database, clock = memory_service
    expected = service.remember_preference(
        "demo-user", "milk", "type", "oat"
    )

    database.initialize()
    restarted_service = GroceryService(database, clock=clock)

    assert restarted_service.list_preferences("demo-user") == (expected,)
