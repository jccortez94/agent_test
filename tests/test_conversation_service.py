from datetime import datetime, timedelta, timezone

import pytest

from grocery_agent import (
    ConversationNotFoundError,
    ConversationService,
    Database,
    InvalidMessageError,
)


class MutableClock:
    def __init__(self):
        self.current = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.current

    def advance(self):
        self.current += timedelta(minutes=1)


@pytest.fixture
def conversation_state(tmp_path):
    database = Database(tmp_path / "conversations.sqlite3")
    database.initialize()
    clock = MutableClock()
    return ConversationService(database, clock=clock), database, clock


def test_messages_are_persisted_in_order_and_can_be_bounded(conversation_state):
    service, _, clock = conversation_state
    conversation = service.start_conversation("demo-user")
    user_message = service.append_message(
        "demo-user", conversation.id, "user", "  Add milk  "
    )
    clock.advance()
    assistant_message = service.append_message(
        "demo-user", conversation.id, "assistant", "Added."
    )

    assert service.get_conversation("demo-user", conversation.id) == conversation
    assert service.list_messages("demo-user", conversation.id) == (
        user_message,
        assistant_message,
    )
    assert user_message.content == "Add milk"
    assert service.list_messages("demo-user", conversation.id, limit=1) == (
        assistant_message,
    )


def test_conversation_access_is_isolated_by_user(conversation_state):
    service, _, _ = conversation_state
    conversation = service.start_conversation("alice")

    with pytest.raises(ConversationNotFoundError):
        service.get_conversation("bob", conversation.id)

    with pytest.raises(ConversationNotFoundError):
        service.append_message("bob", conversation.id, "user", "Hello")


@pytest.mark.parametrize(
    "role, content",
    [("system", "hello"), ("tool", "hello"), ("user", ""), ("user", "   ")],
)
def test_invalid_visible_messages_are_rejected(conversation_state, role, content):
    service, _, _ = conversation_state
    conversation = service.start_conversation("demo-user")

    with pytest.raises(InvalidMessageError):
        service.append_message(
            "demo-user", conversation.id, role=role, content=content
        )

    assert service.list_messages("demo-user", conversation.id) == ()


def test_history_survives_service_restart(conversation_state):
    service, database, clock = conversation_state
    conversation = service.start_conversation("demo-user")
    expected = service.append_message(
        "demo-user", conversation.id, "user", "Remember oat milk"
    )

    restarted = ConversationService(database, clock=clock)

    assert restarted.list_messages("demo-user", conversation.id) == (expected,)
