"""Shared deterministic OpenAI Responses API test doubles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ScriptedFunctionCall:
    name: str
    arguments: str
    call_id: str
    type: str = "function_call"


@dataclass
class ScriptedResponse:
    output: list[Any]
    output_text: str = ""


class ScriptedResponsesAPI:
    def __init__(self, outcomes=()):
        self._outcomes = list(outcomes)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if not self._outcomes:
            raise AssertionError("No scripted model response remains")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def queue(self, *outcomes):
        self._outcomes.extend(outcomes)

    def assert_finished(self):
        assert self._outcomes == []


class ScriptedOpenAIClient:
    def __init__(self, outcomes=()):
        self.responses = ScriptedResponsesAPI(outcomes)


def function_call(name: str, arguments: object, call_id: str):
    raw_arguments = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )
    return ScriptedFunctionCall(name, raw_arguments, call_id)
