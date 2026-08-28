"""Spanish acceptance scenarios derived from docs/Task.txt.

The model boundary is scripted so these tests exercise the real agent loop,
tools, and SQLite state without making paid or nondeterministic API calls.
"""

from __future__ import annotations

import json

from grocery_agent import (
    ConversationService,
    Database,
    GroceryService,
    TextAgent,
)
from tests.fakes import (
    ScriptedOpenAIClient,
    ScriptedResponse as Response,
    function_call as _call,
)


USER_ID = "usuario-aceptacion"


def _application(tmp_path, responses):
    database = Database(tmp_path / "aceptacion.sqlite3")
    database.initialize()
    groceries = GroceryService(database)
    conversations = ConversationService(database)
    client = ScriptedOpenAIClient(responses)
    agent = TextAgent(
        client=client,
        model="modelo-de-prueba",
        grocery_service=groceries,
        conversation_service=conversations,
    )
    return agent, client, groceries, conversations


def _items_by_name(items):
    return {
        item.product_name: (item.quantity, item.unit)
        for item in items
    }


def _function_outputs(request):
    return {
        item["call_id"]: json.loads(item["output"])
        for item in request["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }


def test_gestiona_la_cesta_y_guarda_el_pedido_con_fecha_productos_y_cantidades(
    tmp_path,
):
    responses = [
        Response(
            [_call("add_cart_item", {
                "product_name": "leche",
                "quantity": 2,
                "unit": "litros",
            }, "añadir-leche")]
        ),
        Response([], "He añadido dos litros de leche."),
        Response(
            [
                _call("add_cart_item", {
                    "product_name": "tomates",
                    "quantity": 4,
                    "unit": None,
                }, "añadir-tomates"),
                _call("add_cart_item", {
                    "product_name": "yogures",
                    "quantity": 2,
                    "unit": None,
                }, "añadir-yogures"),
            ]
        ),
        Response([], "He añadido cuatro tomates y dos yogures."),
        Response(
            [
                _call(
                    "remove_cart_item",
                    {"product_name": "tomates"},
                    "quitar-tomates",
                ),
                _call("set_cart_item_quantity", {
                    "product_name": "yogures",
                    "quantity": 6,
                }, "cambiar-yogures"),
            ]
        ),
        Response([], "He quitado los tomates y ahora llevas seis yogures."),
        Response([_call("get_cart", {}, "consultar-cesta")]),
        Response([], "Llevas dos litros de leche y seis yogures."),
        Response([_call("confirm_order", {}, "confirmar-pedido")]),
        Response([], "Pedido confirmado."),
    ]
    agent, client, groceries, conversations = _application(tmp_path, responses)
    conversation = conversations.start_conversation(USER_ID)

    first_turn = agent.run_turn(
        USER_ID, conversation.id, "Añade dos litros de leche."
    )
    agent.run_turn(
        USER_ID, conversation.id, "Añade cuatro tomates y dos yogures."
    )
    changed_turn = agent.run_turn(
        USER_ID,
        conversation.id,
        "Quita los tomates. En vez de dos yogures quiero seis.",
    )
    cart_turn = agent.run_turn(
        USER_ID, conversation.id, "¿Qué llevo en la cesta?"
    )
    confirmed_turn = agent.run_turn(
        USER_ID, conversation.id, "Confirma el pedido."
    )

    assert first_turn.executed_tools == ("add_cart_item",)
    assert changed_turn.executed_tools == (
        "remove_cart_item",
        "set_cart_item_quantity",
    )
    assert cart_turn.executed_tools == ("get_cart",)
    assert confirmed_turn.executed_tools == ("confirm_order",)
    assert groceries.get_cart(USER_ID).items == ()

    orders = groceries.get_recent_orders(USER_ID)
    assert len(orders) == 1
    assert orders[0].ordered_at.utcoffset() is not None
    assert _items_by_name(orders[0].items) == {
        "leche": (2, "litros"),
        "yogures": (6, None),
    }
    messages = conversations.list_messages(USER_ID, conversation.id)
    assert messages[-1].content == "Pedido confirmado."
    assert len(messages) == 10
    client.responses.assert_finished()


def test_reutiliza_la_compra_anterior_sin_leche_y_con_aguacates(tmp_path):
    agent, client, groceries, conversations = _application(tmp_path, [])
    groceries.add_item(USER_ID, "leche", 2, "litros")
    groceries.add_item(USER_ID, "arroz", 1, "paquete")
    previous_order = groceries.confirm_order(USER_ID)
    client.responses.queue(
        Response([_call("get_recent_orders", {"limit": 5}, "pedidos")]),
        Response([], "Claro. ¿Quieres que use tu compra anterior?"),
        Response(
            [
                _call(
                    "copy_order_to_cart",
                    {
                        "order_id": previous_order.id,
                        "replace_existing": True,
                    },
                    "copiar-pedido",
                ),
                _call(
                    "remove_cart_item",
                    {"product_name": "leche"},
                    "quitar-leche",
                ),
                _call(
                    "add_cart_item",
                    {
                        "product_name": "aguacates",
                        "quantity": 4,
                        "unit": None,
                    },
                    "añadir-aguacates",
                ),
            ]
        ),
        Response(
            [],
            "He usado la compra anterior, quitado la leche y añadido "
            "cuatro aguacates.",
        ),
    )
    conversation = conversations.start_conversation(USER_ID)

    reference_turn = agent.run_turn(
        USER_ID, conversation.id, "Necesito hacer la compra para esta semana."
    )
    change_turn = agent.run_turn(
        USER_ID,
        conversation.id,
        "Sí, pero esta vez no necesito leche y añade aguacates.",
    )

    assert reference_turn.executed_tools == ("get_recent_orders",)
    assert change_turn.executed_tools == (
        "copy_order_to_cart",
        "remove_cart_item",
        "add_cart_item",
    )
    assert _items_by_name(groceries.get_cart(USER_ID).items) == {
        "aguacates": (4, None),
        "arroz": (1, "paquete"),
    }
    client.responses.assert_finished()


def test_necesito_cafe_consulta_memoria_y_espera_confirmacion(tmp_path):
    agent, client, groceries, conversations = _application(tmp_path, [])
    groceries.remember_preference(USER_ID, "café", "marca", "Lavazza")
    groceries.add_item(USER_ID, "café Lavazza", 1, "paquete")
    groceries.confirm_order(USER_ID)
    client.responses.queue(
        Response(
            [
                _call(
                    "get_preferences",
                    {"subject": "café"},
                    "preferencias-café",
                ),
                _call(
                    "get_frequent_items",
                    {"limit": 20},
                    "frecuencia-café",
                ),
            ]
        ),
        Response(
            [],
            "La última vez compraste un paquete de café Lavazza. "
            "¿Quieres el mismo?",
        ),
        Response(
            [
                _call(
                    "add_cart_item",
                    {
                        "product_name": "café Lavazza",
                        "quantity": 1,
                        "unit": "paquete",
                    },
                    "añadir-café",
                )
            ]
        ),
        Response([], "He añadido un paquete de café Lavazza."),
    )
    conversation = conversations.start_conversation(USER_ID)

    suggestion_turn = agent.run_turn(
        USER_ID, conversation.id, "Necesito café."
    )

    assert suggestion_turn.executed_tools == (
        "get_preferences",
        "get_frequent_items",
    )
    assert suggestion_turn.text.endswith("¿Quieres el mismo?")
    assert groceries.get_cart(USER_ID).items == ()
    memory_outputs = _function_outputs(client.responses.requests[1])
    assert memory_outputs["preferencias-café"]["result"][0]["value"] == (
        "Lavazza"
    )
    assert memory_outputs["frecuencia-café"]["result"][0][
        "product_name"
    ] == "café Lavazza"

    confirmation_turn = agent.run_turn(
        USER_ID, conversation.id, "Sí, añade el mismo."
    )

    assert confirmation_turn.executed_tools == ("add_cart_item",)
    assert _items_by_name(groceries.get_cart(USER_ID).items) == {
        "café Lavazza": (1, "paquete")
    }
    client.responses.assert_finished()


def test_recuerda_la_preferencia_de_leche_de_avena_entre_conversaciones(
    tmp_path,
):
    responses = [
        Response(
            [_call("remember_preference", {
                "subject": "leche",
                "attribute": "tipo",
                "value": "avena",
            }, "recordar-avena")]
        ),
        Response([], "Recordaré que prefieres leche de avena."),
        Response(
            [
                _call(
                    "get_preferences",
                    {"subject": "leche"},
                    "consultar-leche",
                ),
                _call(
                    "get_frequent_items",
                    {"limit": 20},
                    "consultar-frecuencia",
                ),
            ]
        ),
        Response(
            [],
            "Recuerdo que prefieres leche de avena. "
            "¿Añado dos litros de esa?",
        ),
        Response(
            [_call("add_cart_item", {
                "product_name": "leche de avena",
                "quantity": 2,
                "unit": "litros",
            }, "añadir-avena")]
        ),
        Response([], "He añadido dos litros de leche de avena."),
    ]
    agent, client, groceries, conversations = _application(tmp_path, responses)
    first_conversation = conversations.start_conversation(USER_ID)

    agent.run_turn(
        USER_ID,
        first_conversation.id,
        "La próxima vez recuerda que prefiero leche de avena.",
    )
    second_conversation = conversations.start_conversation(USER_ID)
    suggestion_turn = agent.run_turn(
        USER_ID, second_conversation.id, "Añade dos litros de leche."
    )

    assert suggestion_turn.executed_tools == (
        "get_preferences",
        "get_frequent_items",
    )
    assert groceries.get_cart(USER_ID).items == ()
    assert groceries.list_preferences(USER_ID, "leche")[0].value == "avena"

    agent.run_turn(
        USER_ID, second_conversation.id, "Sí, añade leche de avena."
    )

    assert _items_by_name(groceries.get_cart(USER_ID).items) == {
        "leche de avena": (2, "litros")
    }
    assert [
        message.content
        for message in conversations.list_messages(
            USER_ID, first_conversation.id
        )
    ] == [
        "La próxima vez recuerda que prefiero leche de avena.",
        "Recordaré que prefieres leche de avena.",
    ]
    client.responses.assert_finished()


def test_reconstruye_la_compra_habitual_desde_productos_frecuentes(tmp_path):
    agent, client, groceries, conversations = _application(tmp_path, [])
    groceries.add_item(USER_ID, "manzanas", 2)
    groceries.add_item(USER_ID, "arroz", 1, "paquete")
    groceries.confirm_order(USER_ID)
    groceries.add_item(USER_ID, "manzanas", 4)
    groceries.add_item(USER_ID, "arroz", 1, "paquete")
    groceries.confirm_order(USER_ID)
    client.responses.queue(
        Response(
            [_call("get_frequent_items", {"limit": 20}, "compra-frecuente")]
        ),
        Response(
            [
                _call("add_cart_item", {
                    "product_name": "manzanas",
                    "quantity": 3,
                    "unit": None,
                }, "añadir-manzanas"),
                _call("add_cart_item", {
                    "product_name": "arroz",
                    "quantity": 1,
                    "unit": "paquete",
                }, "añadir-arroz"),
            ]
        ),
        Response([], "He reconstruido tu compra habitual."),
    )
    conversation = conversations.start_conversation(USER_ID)

    turn = agent.run_turn(
        USER_ID, conversation.id, "Hazme la compra habitual."
    )

    assert turn.executed_tools == (
        "get_frequent_items",
        "add_cart_item",
        "add_cart_item",
    )
    assert _items_by_name(groceries.get_cart(USER_ID).items) == {
        "arroz": (1, "paquete"),
        "manzanas": (3, None),
    }
    frequency_output = _function_outputs(client.responses.requests[1])[
        "compra-frecuente"
    ]["result"]
    apples = next(
        item for item in frequency_output if item["product_name"] == "manzanas"
    )
    assert apples["order_count"] == 2
    assert apples["average_quantity"] == 3
    client.responses.assert_finished()
