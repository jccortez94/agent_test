from datetime import datetime, timezone

import pytest

from grocery_agent import (
    Database,
    EmptyCartError,
    GroceryService,
    InvalidProductNameError,
    InvalidQuantityError,
    ItemNotFoundError,
    UnitConflictError,
)


FIXED_TIME = datetime(2026, 8, 28, 14, 30, tzinfo=timezone.utc)


@pytest.fixture
def database(tmp_path):
    database = Database(tmp_path / "grocery.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def service(database):
    return GroceryService(database, clock=lambda: FIXED_TIME)


def test_new_user_gets_an_empty_active_cart(service):
    cart = service.get_cart("demo-user")

    assert cart.user_id == "demo-user"
    assert cart.created_at == FIXED_TIME
    assert cart.items == ()
    assert service.get_cart("demo-user").id == cart.id


def test_add_merge_update_and_remove_items(service):
    cart = service.add_item(
        "demo-user", "  Leche   de AVENA ", quantity=2, unit="Litros"
    )
    assert len(cart.items) == 1
    assert cart.items[0].product_name == "Leche de AVENA"
    assert cart.items[0].quantity == 2
    assert cart.items[0].unit == "litros"

    cart = service.add_item(
        "demo-user", "leche de avena", quantity=1.5, unit="litros"
    )
    assert len(cart.items) == 1
    assert cart.items[0].quantity == 3.5

    cart = service.set_item_quantity("demo-user", "LECHE DE AVENA", 6)
    assert cart.items[0].quantity == 6

    cart = service.remove_item("demo-user", "leche de avena")
    assert cart.items == ()


@pytest.mark.parametrize(
    "quantity",
    [0, -1, float("nan"), float("inf"), True, "not-a-number", None],
)
def test_invalid_quantities_are_rejected_without_changing_the_cart(
    service, quantity
):
    service.add_item("demo-user", "tomatoes", 2)

    with pytest.raises(InvalidQuantityError):
        service.set_item_quantity("demo-user", "tomatoes", quantity)

    assert service.get_cart("demo-user").items[0].quantity == 2


@pytest.mark.parametrize("product_name", ["", "   ", None, 123])
def test_invalid_product_names_are_rejected(service, product_name):
    with pytest.raises(InvalidProductNameError):
        service.add_item("demo-user", product_name, 1)


def test_conflicting_units_are_rejected_and_original_item_is_unchanged(service):
    service.add_item("demo-user", "milk", 2, "liters")

    with pytest.raises(UnitConflictError):
        service.add_item("demo-user", "MILK", 1, "bottles")

    item = service.get_cart("demo-user").items[0]
    assert item.quantity == 2
    assert item.unit == "liters"


def test_missing_item_operations_are_explicit(service):
    with pytest.raises(ItemNotFoundError):
        service.set_item_quantity("demo-user", "coffee", 2)

    with pytest.raises(ItemNotFoundError):
        service.remove_item("demo-user", "coffee")


def test_confirm_order_snapshots_items_and_opens_a_new_cart(service):
    original_cart = service.add_item("demo-user", "bananas", 6)
    service.add_item("demo-user", "oat milk", 2, "liters")

    order = service.confirm_order("demo-user")

    assert order.user_id == "demo-user"
    assert order.ordered_at == FIXED_TIME
    assert [(item.product_name, item.quantity, item.unit) for item in order.items] == [
        ("bananas", 6, None),
        ("oat milk", 2, "liters"),
    ]

    new_cart = service.get_cart("demo-user")
    assert new_cart.id != original_cart.id
    assert new_cart.items == ()

    stored_orders = service.list_orders("demo-user")
    assert stored_orders == (order,)


def test_confirmed_order_is_immutable_as_the_next_cart_changes(service, database):
    service.add_item("demo-user", "apples", 4)
    confirmed_order = service.confirm_order("demo-user")

    service.add_item("demo-user", "pears", 2)

    assert service.list_orders("demo-user") == (confirmed_order,)
    assert service.get_cart("demo-user").items[0].product_name == "pears"

    with database.connect() as connection:
        active_cart_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM carts
            WHERE user_id = ? AND status = 'active'
            """,
            ("demo-user",),
        ).fetchone()[0]
    assert active_cart_count == 1


def test_confirming_an_empty_cart_does_not_create_an_order(service):
    with pytest.raises(EmptyCartError):
        service.confirm_order("demo-user")

    assert service.list_orders("demo-user") == ()
    assert service.get_cart("demo-user").items == ()


def test_orders_persist_across_service_instances(database):
    first_service = GroceryService(database, clock=lambda: FIXED_TIME)
    first_service.add_item("demo-user", "Lavazza coffee", 1)
    expected_order = first_service.confirm_order("demo-user")

    restarted_service = GroceryService(database, clock=lambda: FIXED_TIME)

    assert restarted_service.list_orders("demo-user") == (expected_order,)
    assert restarted_service.get_cart("demo-user").items == ()


def test_users_have_isolated_carts_and_order_histories(service):
    service.add_item("alice", "apples", 4)
    service.add_item("bob", "pears", 3)
    alice_order = service.confirm_order("alice")

    assert service.get_cart("alice").items == ()
    assert service.get_cart("bob").items[0].product_name == "pears"
    assert service.list_orders("alice") == (alice_order,)
    assert service.list_orders("bob") == ()
