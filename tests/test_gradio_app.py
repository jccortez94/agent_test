from __future__ import annotations

import gradio as gr
import pytest

from grocery_agent.agent import AgentTurn
from grocery_agent.database import Database
from grocery_agent.gradio_app import (
    DASHBOARD_ORDER_LIMIT,
    GradioController,
    _new_interaction_view,
    _parser,
    _submit_voice_view,
    _submit_view,
    create_gradio_app,
)
from grocery_agent.history import ConversationService
from grocery_agent.service import GroceryService
from grocery_agent.speech import (
    DEFAULT_SPEECH_INSTRUCTIONS,
    DEFAULT_TRANSCRIPTION_LANGUAGE,
)


class ScriptedAgent:
    def __init__(self, groceries, conversations):
        self.groceries = groceries
        self.conversations = conversations
        self.calls = []

    def run_turn(self, user_id, conversation_id, user_text):
        self.calls.append((user_id, conversation_id, user_text))
        self.conversations.append_message(
            user_id, conversation_id, "user", user_text
        )
        self.groceries.add_item(user_id, "apples", 4)
        answer = "Added four apples."
        self.conversations.append_message(
            user_id, conversation_id, "assistant", answer
        )
        return AgentTurn(conversation_id, answer, ("add_cart_item",))


class FailingAgent:
    def __init__(self, groceries, conversations):
        self.groceries = groceries
        self.conversations = conversations

    def run_turn(self, user_id, conversation_id, user_text):
        self.conversations.append_message(
            user_id, conversation_id, "user", user_text
        )
        self.groceries.add_item(user_id, "milk", 2, "liters")
        raise RuntimeError("API unavailable after tool")


class ScriptedSpeechService:
    def __init__(
        self,
        transcription="Add four apples by voice",
        audio=b"assistant-wav",
        transcription_error=None,
        synthesis_error=None,
    ):
        self.transcription = transcription
        self.audio = audio
        self.transcription_error = transcription_error
        self.synthesis_error = synthesis_error
        self.transcription_calls = []
        self.synthesis_calls = []

    def transcribe(self, audio_path):
        self.transcription_calls.append(audio_path)
        if self.transcription_error:
            raise self.transcription_error
        return self.transcription

    def synthesize(self, text):
        self.synthesis_calls.append(text)
        if self.synthesis_error:
            raise self.synthesis_error
        return self.audio


@pytest.fixture
def ui_state(tmp_path):
    database = Database(tmp_path / "ui.sqlite3")
    database.initialize()
    groceries = GroceryService(database)
    conversations = ConversationService(database)
    agent = ScriptedAgent(groceries, conversations)
    controller = GradioController(
        agent=agent,
        grocery_service=groceries,
        conversation_service=conversations,
        user_id="demo-user",
    )
    return controller, agent, groceries, conversations


def test_new_conversation_returns_an_empty_sqlite_backed_view(ui_state):
    controller, _, _, _ = ui_state

    snapshot = controller.new_conversation()

    assert snapshot.conversation_id > 0
    assert snapshot.chat_messages == []
    assert snapshot.cart_rows == []
    assert snapshot.order_rows == []
    assert snapshot.preference_rows == []
    assert snapshot.status == f"Conversación {snapshot.conversation_id} lista."


def test_submit_reloads_transcript_and_cart_and_clears_textbox(ui_state):
    controller, agent, _, _ = ui_state
    conversation_id = controller.new_conversation().conversation_id

    values = _submit_view(controller, "Add four apples", conversation_id)

    assert values[0] == ""
    assert values[1:4] == (None, "", None)
    assert values[4] == conversation_id
    assert values[5] == [
        {"role": "user", "content": "Add four apples"},
        {"role": "assistant", "content": "Added four apples."},
    ]
    assert values[6] == [["apples", 4, ""]]
    assert values[9] == "Listo. Herramientas utilizadas: add_cart_item."
    assert agent.calls == [("demo-user", conversation_id, "Add four apples")]


def test_empty_submission_does_not_call_the_agent(ui_state):
    controller, agent, _, _ = ui_state
    conversation_id = controller.new_conversation().conversation_id

    snapshot = controller.submit_text("   ", conversation_id)

    assert agent.calls == []
    assert snapshot.status == "Escribe un mensaje antes de enviarlo."


def test_new_conversation_keeps_cart_but_not_old_transcript(ui_state):
    controller, _, _, _ = ui_state
    first_id = controller.new_conversation().conversation_id
    controller.submit_text("Add apples", first_id)

    snapshot = controller.new_conversation()

    assert snapshot.conversation_id != first_id
    assert snapshot.chat_messages == []
    assert snapshot.cart_rows == [["apples", 4, ""]]


def test_snapshot_includes_confirmed_orders_and_preferences(ui_state):
    controller, _, groceries, _ = ui_state
    conversation_id = controller.new_conversation().conversation_id
    groceries.add_item("demo-user", "oat milk", 2, "liters")
    order = groceries.confirm_order("demo-user")
    groceries.remember_preference("demo-user", "coffee", "brand", "Lavazza")

    snapshot = controller.refresh(conversation_id)

    assert snapshot.cart_rows == []
    assert snapshot.order_rows[0][0] == order.id
    assert snapshot.order_rows[0][2] == "2 liters oat milk"
    assert snapshot.preference_rows == [["coffee", "brand", "Lavazza"]]
    assert snapshot.status == (
        f"Conversación {conversation_id} actualizada desde SQLite."
    )


def test_failure_displays_committed_state_but_hides_technical_details(
    ui_state, caplog
):
    controller, _, groceries, conversations = ui_state
    controller.agent = FailingAgent(groceries, conversations)
    conversation_id = controller.new_conversation().conversation_id

    with caplog.at_level("ERROR", logger="grocery_agent.gradio_app"):
        snapshot = controller.submit_text(
            "Add two liters of milk", conversation_id
        )

    assert snapshot.chat_messages == [
        {"role": "user", "content": "Add two liters of milk"}
    ]
    assert snapshot.cart_rows == [["milk", 2, "liters"]]
    assert snapshot.status == (
        "No se pudo completar la solicitud. Revisa la cesta antes de "
        "intentarlo de nuevo."
    )
    assert "API unavailable after tool" not in snapshot.status
    assert "API unavailable after tool" in caplog.text


def test_missing_or_unowned_session_conversation_is_replaced(ui_state):
    controller, _, _, _ = ui_state

    snapshot = controller.refresh(999_999)

    assert snapshot.conversation_id != 999_999
    assert snapshot.chat_messages == []


def test_voice_submission_runs_stt_agent_and_tts_in_order(ui_state):
    controller, agent, _, _ = ui_state
    speech = ScriptedSpeechService()
    controller.speech_service = speech
    conversation_id = controller.new_conversation().conversation_id

    submission = controller.submit_voice("recording.webm", conversation_id)

    assert submission.transcription == "Add four apples by voice"
    assert submission.audio == b"assistant-wav"
    assert submission.snapshot.chat_messages == [
        {"role": "user", "content": "Add four apples by voice"},
        {"role": "assistant", "content": "Added four apples."},
    ]
    assert submission.snapshot.cart_rows == [["apples", 4, ""]]
    assert submission.snapshot.status == (
        "Respuesta de voz lista. Herramientas utilizadas: add_cart_item."
    )
    assert speech.transcription_calls == ["recording.webm"]
    assert agent.calls == [
        ("demo-user", conversation_id, "Add four apples by voice")
    ]
    assert speech.synthesis_calls == ["Added four apples."]


def test_voice_view_clears_recording_and_returns_all_panels(ui_state):
    controller, _, _, _ = ui_state
    controller.speech_service = ScriptedSpeechService(
        transcription="Add apples", audio=b"wav"
    )
    conversation_id = controller.new_conversation().conversation_id

    values = _submit_voice_view(controller, "recording.wav", conversation_id)

    assert values[0] == ""
    assert values[1] is None
    assert values[2] == "Add apples"
    assert values[3] == b"wav"
    assert values[4] == conversation_id
    assert values[6] == [["apples", 4, ""]]


def test_missing_voice_input_does_not_call_speech_or_agent(ui_state):
    controller, agent, _, _ = ui_state
    speech = ScriptedSpeechService()
    controller.speech_service = speech
    conversation_id = controller.new_conversation().conversation_id

    submission = controller.submit_voice(None, conversation_id)

    assert submission.transcription == ""
    assert submission.audio is None
    assert submission.snapshot.status == (
        "Graba o sube un audio antes de enviarlo."
    )
    assert speech.transcription_calls == []
    assert agent.calls == []


def test_voice_input_reports_when_speech_is_not_configured(ui_state):
    controller, agent, _, _ = ui_state
    conversation_id = controller.new_conversation().conversation_id

    submission = controller.submit_voice("recording.wav", conversation_id)

    assert submission.snapshot.status == (
        "La entrada de voz no está configurada."
    )
    assert agent.calls == []


def test_transcription_failure_does_not_mutate_conversation_or_cart(ui_state):
    controller, agent, _, _ = ui_state
    speech = ScriptedSpeechService(
        transcription_error=RuntimeError("STT unavailable")
    )
    controller.speech_service = speech
    conversation_id = controller.new_conversation().conversation_id

    submission = controller.submit_voice("recording.wav", conversation_id)

    assert submission.transcription == ""
    assert submission.audio is None
    assert submission.snapshot.chat_messages == []
    assert submission.snapshot.cart_rows == []
    assert submission.snapshot.status == (
        "No se pudo transcribir el audio. Inténtalo de nuevo."
    )
    assert "STT unavailable" not in submission.snapshot.status
    assert agent.calls == []
    assert speech.synthesis_calls == []


def test_tts_failure_preserves_the_completed_agent_turn(ui_state):
    controller, agent, _, _ = ui_state
    speech = ScriptedSpeechService(
        transcription="Add apples",
        synthesis_error=RuntimeError("TTS unavailable"),
    )
    controller.speech_service = speech
    conversation_id = controller.new_conversation().conversation_id

    submission = controller.submit_voice("recording.wav", conversation_id)

    assert submission.transcription == "Add apples"
    assert submission.audio is None
    assert submission.snapshot.chat_messages[-1] == {
        "role": "assistant",
        "content": "Added four apples.",
    }
    assert submission.snapshot.cart_rows == [["apples", 4, ""]]
    assert submission.snapshot.status == (
        "La respuesta se guardó, pero no se pudo generar la voz."
    )
    assert "TTS unavailable" not in submission.snapshot.status
    assert len(agent.calls) == 1


def test_new_conversation_clears_all_ephemeral_inputs(ui_state):
    controller, _, _, _ = ui_state

    values = _new_interaction_view(controller)

    assert values[:4] == ("", None, "", None)
    assert values[4] > 0


def test_dashboard_limits_order_history_to_recent_orders(ui_state):
    controller, _, groceries, _ = ui_state
    conversation_id = controller.new_conversation().conversation_id
    orders = []
    for index in range(DASHBOARD_ORDER_LIMIT + 1):
        groceries.add_item("demo-user", f"product {index}", 1)
        orders.append(groceries.confirm_order("demo-user"))

    snapshot = controller.refresh(conversation_id)

    assert len(snapshot.order_rows) == DASHBOARD_ORDER_LIMIT
    assert snapshot.order_rows[0][0] == orders[-1].id
    assert all(row[0] != orders[0].id for row in snapshot.order_rows)


def test_gradio_blocks_can_be_built_without_launching_or_calling_openai(ui_state):
    controller, agent, _, _ = ui_state

    app = create_gradio_app(controller)

    assert isinstance(app, gr.Blocks)
    assert agent.calls == []
    component_types = [component["type"] for component in app.config["components"]]
    assert component_types.count("audio") == 2

    component_props = [
        component.get("props", {}) for component in app.config["components"]
    ]
    labels = {
        props["label"] for props in component_props if "label" in props
    }
    values = {
        props["value"]
        for props in component_props
        if isinstance(props.get("value"), str)
    }
    headers = {
        tuple(props["headers"])
        for props in component_props
        if "headers" in props
    }

    assert app.config["title"] == "Asistente de supermercado"
    assert {
        "Conversación",
        "Voz",
        "Transcripción",
        "Texto",
        "Mensaje",
        "Cesta actual",
        "Historial de pedidos",
        "Preferencias recordadas",
    } <= labels
    assert {
        "Enviar audio",
        "Enviar",
        "Nueva conversación",
        "Actualizar datos",
        "Iniciando una conversación local...",
    } <= values
    assert ("Producto", "Cantidad", "Unidad") in headers
    assert ("Pedido", "Fecha", "Productos") in headers
    assert ("Producto o tema", "Atributo", "Valor") in headers


def test_audio_parser_defaults_to_spanish(monkeypatch):
    monkeypatch.delenv("OPENAI_STT_LANGUAGE", raising=False)
    monkeypatch.delenv("OPENAI_TTS_INSTRUCTIONS", raising=False)

    arguments = _parser().parse_args([])

    assert arguments.transcription_language == DEFAULT_TRANSCRIPTION_LANGUAGE
    assert arguments.speech_instructions == DEFAULT_SPEECH_INSTRUCTIONS


def test_audio_parser_accepts_environment_and_cli_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_STT_LANGUAGE", "pt")
    monkeypatch.setenv("OPENAI_TTS_INSTRUCTIONS", "Fale naturalmente.")

    environment_arguments = _parser().parse_args([])
    cli_arguments = _parser().parse_args(
        [
            "--transcription-language",
            "fr",
            "--speech-instructions",
            "Parlez naturellement.",
        ]
    )

    assert environment_arguments.transcription_language == "pt"
    assert environment_arguments.speech_instructions == "Fale naturalmente."
    assert cli_arguments.transcription_language == "fr"
    assert cli_arguments.speech_instructions == "Parlez naturellement."
