import json

import pytest

from grocery_agent import Database, GroceryService
from grocery_agent.tools import GroceryToolDispatcher, TOOL_DEFINITIONS


@pytest.fixture
def tools(tmp_path):
    database = Database(tmp_path / "tools.sqlite3")
    database.initialize()
    service = GroceryService(database)
    return GroceryToolDispatcher(service, "demo-user"), service


def test_all_openai_tools_use_strict_closed_schemas():
    names = {tool["name"] for tool in TOOL_DEFINITIONS}
    assert names == {
        "add_cart_item",
        "remove_cart_item",
        "set_cart_item_quantity",
        "get_cart",
        "confirm_order",
        "get_recent_orders",
        "copy_order_to_cart",
        "get_frequent_items",
        "get_preferences",
        "remember_preference",
    }
    for tool in TOOL_DEFINITIONS:
        parameters = tool["parameters"]
        assert tool["type"] == "function"
        assert tool["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_dispatcher_exercises_cart_order_and_memory_tools(tools):
    dispatcher, _ = tools
    added = dispatcher.execute(
        "add_cart_item",
        json.dumps({"product_name": "oat milk", "quantity": 2, "unit": "liters"}),
    )
    assert added["ok"] is True
    assert added["result"]["items"][0]["quantity"] == 2

    changed = dispatcher.execute(
        "set_cart_item_quantity",
        json.dumps({"product_name": "oat milk", "quantity": 3}),
    )
    assert changed["result"]["items"][0]["quantity"] == 3

    confirmed = dispatcher.execute("confirm_order", "{}")
    order_id = confirmed["result"]["order_id"]
    assert confirmed["ok"] is True

    recent = dispatcher.execute("get_recent_orders", '{"limit": 5}')
    frequent = dispatcher.execute("get_frequent_items", '{"limit": 5}')
    assert recent["result"][0]["order_id"] == order_id
    assert frequent["result"][0]["average_quantity"] == 3

    restored = dispatcher.execute(
        "copy_order_to_cart",
        json.dumps({"order_id": order_id, "replace_existing": True}),
    )
    assert restored["result"]["items"][0]["product_name"] == "oat milk"

    remembered = dispatcher.execute(
        "remember_preference",
        json.dumps({"subject": "milk", "attribute": "type", "value": "oat"}),
    )
    preferences = dispatcher.execute("get_preferences", '{"subject": "milk"}')
    assert remembered["ok"] is True
    assert preferences["result"][0]["value"] == "oat"

    removed = dispatcher.execute(
        "remove_cart_item", '{"product_name": "oat milk"}'
    )
    cart = dispatcher.execute("get_cart", "{}")
    assert removed["ok"] is True
    assert cart["result"]["items"] == []


@pytest.mark.parametrize(
    "name, arguments, error_type",
    [
        ("not_allowed", "{}", "UnknownToolError"),
        ("get_cart", "not json", "ToolInputError"),
        ("get_cart", '[]', "ToolInputError"),
        ("get_cart", '{"extra": true}', "ToolInputError"),
        (
            "add_cart_item",
            '{"product_name": "milk", "quantity": 2}',
            "ToolInputError",
        ),
        (
            "add_cart_item",
            '{"product_name": "milk", "quantity": true, "unit": null}',
            "ToolInputError",
        ),
        ("get_recent_orders", '{"limit": 21}', "ToolInputError"),
    ],
)
def test_invalid_or_unknown_tool_calls_do_not_mutate_state(
    tools, name, arguments, error_type
):
    dispatcher, service = tools

    result = dispatcher.execute(name, arguments)

    assert result["ok"] is False
    assert result["error"]["type"] == error_type
    assert service.get_cart("demo-user").items == ()
