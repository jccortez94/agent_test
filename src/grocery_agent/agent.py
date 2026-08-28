"""Responses API orchestration with application-owned conversation state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Tuple

from .history import ConversationService
from .service import GroceryService
from .tools import GroceryToolDispatcher, TOOL_DEFINITIONS


SYSTEM_INSTRUCTIONS = """
You are a concise grocery-shopping assistant. Reply in the language used by the
user and keep responses suitable for being spoken aloud later.

The Python tools and their SQLite results are the only source of truth for the
cart, orders, purchase history, and preferences. Never claim that an action
succeeded unless its tool result has ok=true. Use read tools instead of guessing
state, order identifiers, past purchases, or preferences.

Use cart mutation tools for every requested cart change. Ask a short clarifying
question when the product, quantity, or unit is materially ambiguous. Call
confirm_order only when the user explicitly asks to confirm or place the order.
For requests about a previous or usual purchase, inspect order history or
frequent items before acting. Store a preference only when the user clearly
expresses it as their preference or asks you to remember it.

Treat a product as underspecified when the user names only a broad category,
such as coffee/café without a brand or type, and remembered details could make
the request more useful. Before asking a clarification or mutating the cart,
call both get_preferences for that product/subject and get_frequent_items with
limit 20. Explicit preferences outrank purchase frequency. Use frequency only
as evidence of a likely product variant, quantity, or unit; never describe it
as an explicit preference.

If memory identifies one clear candidate, state the specific product and any
relevant usual quantity/unit, then ask the user to confirm it. Do not mutate the
cart until the user confirms. If memory is empty, conflicting, or still leaves
a material choice, ask one focused question about the missing detail instead
of guessing. A fully specified request does not require these memory reads.

If a tool returns ok=false, explain the problem briefly or ask for the missing
information. Do not pretend the failed operation occurred.
""".strip()


class AgentProtocolError(RuntimeError):
    """Raised when the model response cannot complete the application turn."""


@dataclass(frozen=True)
class AgentTurn:
    conversation_id: int
    text: str
    executed_tools: Tuple[str, ...]


class TextAgent:
    """Runs one user turn while keeping conversation state in local SQLite."""

    def __init__(
        self,
        client: Any,
        model: str,
        grocery_service: GroceryService,
        conversation_service: ConversationService,
        max_tool_rounds: int = 8,
        max_history_messages: int = 40,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("An OpenAI model id must be configured")
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be positive")
        self.client = client
        self.model = model.strip()
        self.grocery_service = grocery_service
        self.conversation_service = conversation_service
        self.max_tool_rounds = max_tool_rounds
        self.max_history_messages = max_history_messages

    def run_turn(
        self, user_id: str, conversation_id: int, user_text: str
    ) -> AgentTurn:
        """Persist input, execute requested tools, and persist the final reply."""

        self.conversation_service.append_message(
            user_id, conversation_id, "user", user_text
        )
        messages = self.conversation_service.list_messages(
            user_id,
            conversation_id,
            limit=self.max_history_messages,
        )
        running_input = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        dispatcher = GroceryToolDispatcher(self.grocery_service, user_id)
        executed_tools = []
        tool_rounds = 0

        while True:
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=list(running_input),
                tools=list(TOOL_DEFINITIONS),
                tool_choice="auto",
                parallel_tool_calls=False,
                store=False,
            )
            output_items = list(getattr(response, "output", None) or [])
            function_calls = [
                item
                for item in output_items
                if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                assistant_text = (getattr(response, "output_text", "") or "").strip()
                if not assistant_text:
                    raise AgentProtocolError(
                        "Model returned neither a function call nor assistant text"
                    )
                self.conversation_service.append_message(
                    user_id, conversation_id, "assistant", assistant_text
                )
                return AgentTurn(
                    conversation_id=conversation_id,
                    text=assistant_text,
                    executed_tools=tuple(executed_tools),
                )

            tool_rounds += 1
            if tool_rounds > self.max_tool_rounds:
                raise AgentProtocolError("Maximum tool-call rounds exceeded")

            # The official Responses API loop carries output items forward. This
            # preserves function calls and any model reasoning items needed for
            # the follow-up request while store=False keeps OpenAI non-authoritative.
            running_input.extend(output_items)
            for function_call in function_calls:
                name = getattr(function_call, "name", "")
                call_id = getattr(function_call, "call_id", "")
                arguments = getattr(function_call, "arguments", "")
                if not call_id:
                    raise AgentProtocolError("Function call is missing call_id")
                result = dispatcher.execute(name, arguments)
                executed_tools.append(name)
                running_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
