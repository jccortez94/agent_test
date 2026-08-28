import json

import pytest

from grocery_agent import (
    AgentProtocolError,
    ConversationService,
    Database,
    GroceryService,
    TextAgent,
)
from grocery_agent.agent import SYSTEM_INSTRUCTIONS
from tests.fakes import (
    ScriptedFunctionCall as FakeFunctionCall,
    ScriptedOpenAIClient,
    ScriptedResponse as FakeResponse,
)


@pytest.fixture
def agent_state(tmp_path):
    database = Database(tmp_path / "agent.sqlite3")
    database.initialize()
    grocery_service = GroceryService(database)
    conversation_service = ConversationService(database)
    conversation = conversation_service.start_conversation("demo-user")
    return grocery_service, conversation_service, conversation


def _agent(agent_state, outcomes, **kwargs):
    grocery_service, conversation_service, _ = agent_state
    client = ScriptedOpenAIClient(outcomes)
    agent = TextAgent(
        client=client,
        model="test-model",
        grocery_service=grocery_service,
        conversation_service=conversation_service,
        **kwargs,
    )
    return agent, client


def test_agent_executes_function_call_and_persists_visible_messages(agent_state):
    grocery_service, conversation_service, conversation = agent_state
    call = FakeFunctionCall(
        name="add_cart_item",
        arguments=json.dumps(
            {"product_name": "oat milk", "quantity": 2, "unit": "liters"}
        ),
        call_id="call-1",
    )
    agent, client = _agent(
        agent_state,
        [FakeResponse([call]), FakeResponse([], "Added two liters of oat milk.")],
    )

    turn = agent.run_turn("demo-user", conversation.id, "Add two liters of oat milk")

    assert turn.text == "Added two liters of oat milk."
    assert turn.executed_tools == ("add_cart_item",)
    assert grocery_service.get_cart("demo-user").items[0].quantity == 2
    messages = conversation_service.list_messages("demo-user", conversation.id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == [
        "Add two liters of oat milk",
        "Added two liters of oat milk.",
    ]

    first_request, second_request = client.responses.requests
    assert first_request["store"] is False
    assert first_request["parallel_tool_calls"] is False
    assert first_request["input"] == [
        {"role": "user", "content": "Add two liters of oat milk"}
    ]
    function_output = next(
        item
        for item in second_request["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert function_output["call_id"] == "call-1"
    assert json.loads(function_output["output"])["ok"] is True


def test_agent_executes_multiple_calls_from_one_response_in_order(agent_state):
    grocery_service, _, conversation = agent_state
    calls = [
        FakeFunctionCall(
            "add_cart_item",
            '{"product_name": "apples", "quantity": 4, "unit": null}',
            "call-1",
        ),
        FakeFunctionCall(
            "add_cart_item",
            '{"product_name": "bananas", "quantity": 6, "unit": null}',
            "call-2",
        ),
    ]
    agent, _ = _agent(
        agent_state,
        [FakeResponse(calls), FakeResponse([], "Added apples and bananas.")],
    )

    turn = agent.run_turn("demo-user", conversation.id, "Add apples and bananas")

    assert turn.executed_tools == ("add_cart_item", "add_cart_item")
    assert [
        item.product_name for item in grocery_service.get_cart("demo-user").items
    ] == ["apples", "bananas"]


def test_agent_returns_dispatch_errors_to_model_without_mutating_state(agent_state):
    grocery_service, _, conversation = agent_state
    bad_call = FakeFunctionCall("delete_everything", "{}", "call-bad")
    agent, client = _agent(
        agent_state,
        [FakeResponse([bad_call]), FakeResponse([], "I could not perform that action.")],
    )

    turn = agent.run_turn("demo-user", conversation.id, "Delete everything")

    assert turn.text == "I could not perform that action."
    assert grocery_service.get_cart("demo-user").items == ()
    function_output = next(
        item
        for item in client.responses.requests[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    result = json.loads(function_output["output"])
    assert result["ok"] is False
    assert result["error"]["type"] == "UnknownToolError"


def test_api_failure_persists_user_message_but_does_not_change_cart(agent_state):
    grocery_service, conversation_service, conversation = agent_state
    agent, _ = _agent(agent_state, [RuntimeError("API unavailable")])

    with pytest.raises(RuntimeError, match="API unavailable"):
        agent.run_turn("demo-user", conversation.id, "Add milk")

    assert grocery_service.get_cart("demo-user").items == ()
    messages = conversation_service.list_messages("demo-user", conversation.id)
    assert [(message.role, message.content) for message in messages] == [
        ("user", "Add milk")
    ]


def test_api_failure_after_a_tool_keeps_the_committed_application_state(agent_state):
    grocery_service, conversation_service, conversation = agent_state
    call = FakeFunctionCall(
        "add_cart_item",
        '{"product_name": "milk", "quantity": 2, "unit": null}',
        "call-1",
    )
    agent, _ = _agent(
        agent_state,
        [FakeResponse([call]), RuntimeError("API unavailable after tool")],
    )

    with pytest.raises(RuntimeError, match="API unavailable after tool"):
        agent.run_turn("demo-user", conversation.id, "Add two milk")

    cart = grocery_service.get_cart("demo-user")
    assert [(item.product_name, item.quantity) for item in cart.items] == [
        ("milk", 2)
    ]
    assert [
        message.role
        for message in conversation_service.list_messages(
            "demo-user", conversation.id
        )
    ] == ["user"]


def test_each_turn_rebuilds_context_from_sqlite_history(agent_state):
    _, _, conversation = agent_state
    agent, client = _agent(
        agent_state,
        [
            FakeResponse([], "Hello."),
            FakeResponse([], "Your cart is empty."),
        ],
    )

    agent.run_turn("demo-user", conversation.id, "Hello")
    agent.run_turn("demo-user", conversation.id, "What is in my cart?")

    assert client.responses.requests[1]["input"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "What is in my cart?"},
    ]


def test_agent_request_contains_the_memory_first_product_policy(agent_state):
    _, _, conversation = agent_state
    agent, client = _agent(
        agent_state,
        [FakeResponse([], "What kind of coffee would you like?")],
    )

    agent.run_turn("demo-user", conversation.id, "I need coffee")

    instructions = client.responses.requests[0]["instructions"]
    assert instructions == SYSTEM_INSTRUCTIONS
    assert "Treat a product as underspecified" in instructions
    assert "call both get_preferences" in instructions
    assert "get_frequent_items with\nlimit 20" in instructions
    assert "Explicit preferences outrank purchase frequency" in instructions
    assert "Do not mutate the\ncart until the user confirms" in instructions


def test_model_response_without_text_or_tools_is_rejected(agent_state):
    _, conversation_service, conversation = agent_state
    agent, _ = _agent(agent_state, [FakeResponse([])])

    with pytest.raises(AgentProtocolError):
        agent.run_turn("demo-user", conversation.id, "Hello")

    assert [
        message.role
        for message in conversation_service.list_messages(
            "demo-user", conversation.id
        )
    ] == ["user"]
