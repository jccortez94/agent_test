"""Spanish Gradio interface backed by the local application services."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

import gradio as gr
from openai import OpenAI

from .agent import AgentTurn, TextAgent
from .database import Database
from .history import ConversationError, ConversationService
from .service import GroceryService
from .speech import (
    DEFAULT_SPEECH_INSTRUCTIONS,
    DEFAULT_TRANSCRIPTION_LANGUAGE,
    OpenAISpeechService,
)


ChatMessage = dict[str, str]
TableRows = List[List[Any]]
DASHBOARD_ORDER_LIMIT = 20
LOGGER = logging.getLogger(__name__)


class AgentRunner(Protocol):
    """Small interface used by the UI and its deterministic test doubles."""

    def run_turn(
        self, user_id: str, conversation_id: int, user_text: str
    ) -> AgentTurn:
        ...


class SpeechRunner(Protocol):
    """Audio operations needed by one turn of the UI."""

    def transcribe(self, audio_path: object) -> str:
        ...

    def synthesize(self, text: str) -> bytes:
        ...


@dataclass(frozen=True)
class DashboardSnapshot:
    """Everything rendered by one refresh of the browser view."""

    conversation_id: int
    chat_messages: List[ChatMessage]
    cart_rows: TableRows
    order_rows: TableRows
    preference_rows: TableRows
    status: str


@dataclass(frozen=True)
class VoiceSubmission:
    """Voice-specific outputs plus the authoritative dashboard state."""

    transcription: str
    audio: Optional[bytes]
    snapshot: DashboardSnapshot


class GradioController:
    """Translate UI events into application calls and fresh SQLite reads."""

    def __init__(
        self,
        agent: AgentRunner,
        grocery_service: GroceryService,
        conversation_service: ConversationService,
        user_id: str,
        speech_service: Optional[SpeechRunner] = None,
    ) -> None:
        self.agent = agent
        self.grocery_service = grocery_service
        self.conversation_service = conversation_service
        self.user_id = user_id
        self.speech_service = speech_service

    def new_conversation(self) -> DashboardSnapshot:
        conversation = self.conversation_service.start_conversation(self.user_id)
        return self._snapshot(
            conversation.id, f"Conversación {conversation.id} lista."
        )

    def refresh(self, conversation_id: object) -> DashboardSnapshot:
        resolved_id = self._resolve_conversation(conversation_id)
        return self._snapshot(
            resolved_id,
            f"Conversación {resolved_id} actualizada desde SQLite.",
        )

    def submit_text(
        self, user_text: object, conversation_id: object
    ) -> DashboardSnapshot:
        resolved_id = self._resolve_conversation(conversation_id)
        normalized_text = user_text.strip() if isinstance(user_text, str) else ""
        if not normalized_text:
            return self._snapshot(
                resolved_id, "Escribe un mensaje antes de enviarlo."
            )

        _, status = self._execute_agent_turn(resolved_id, normalized_text)
        return self._snapshot(resolved_id, status)

    def submit_voice(
        self, audio_path: object, conversation_id: object
    ) -> VoiceSubmission:
        """Run one recorded STT → agent → TTS turn."""

        resolved_id = self._resolve_conversation(conversation_id)
        if audio_path is None or (
            isinstance(audio_path, str) and not audio_path.strip()
        ):
            return VoiceSubmission(
                transcription="",
                audio=None,
                snapshot=self._snapshot(
                    resolved_id, "Graba o sube un audio antes de enviarlo."
                ),
            )
        if self.speech_service is None:
            return VoiceSubmission(
                transcription="",
                audio=None,
                snapshot=self._snapshot(
                    resolved_id, "La entrada de voz no está configurada."
                ),
            )

        try:
            transcription = self.speech_service.transcribe(audio_path)
        except Exception as error:
            return VoiceSubmission(
                transcription="",
                audio=None,
                snapshot=self._snapshot(
                    resolved_id,
                    _error_status(
                        "No se pudo transcribir el audio. Inténtalo de nuevo.",
                        error,
                    ),
                ),
            )

        turn, status = self._execute_agent_turn(resolved_id, transcription)
        if turn is None:
            return VoiceSubmission(
                transcription=transcription,
                audio=None,
                snapshot=self._snapshot(resolved_id, status),
            )

        try:
            audio = self.speech_service.synthesize(turn.text)
        except Exception as error:
            return VoiceSubmission(
                transcription=transcription,
                audio=None,
                snapshot=self._snapshot(
                    resolved_id,
                    _error_status(
                        "La respuesta se guardó, pero no se pudo generar la voz.",
                        error,
                    ),
                ),
            )

        return VoiceSubmission(
            transcription=transcription,
            audio=audio,
            snapshot=self._snapshot(
                resolved_id, _success_status(turn, "Respuesta de voz lista.")
            ),
        )

    def _execute_agent_turn(
        self, conversation_id: int, user_text: str
    ) -> Tuple[Optional[AgentTurn], str]:
        try:
            turn = self.agent.run_turn(
                self.user_id, conversation_id, user_text
            )
        except Exception as error:
            # The agent may already have committed a message or tool action. The
            # caller always reloads SQLite so that committed work remains visible.
            return None, _error_status(
                "No se pudo completar la solicitud. Revisa la cesta antes de "
                "intentarlo de nuevo.",
                error,
            )

        if turn.conversation_id != conversation_id:
            LOGGER.error(
                "Agent changed conversation id from %s to %s",
                conversation_id,
                turn.conversation_id,
            )
            return (
                None,
                "No se pudo completar la solicitud porque cambió la "
                "conversación activa.",
            )
        return turn, _success_status(turn, "Listo.")

    def _resolve_conversation(self, conversation_id: object) -> int:
        try:
            conversation = self.conversation_service.get_conversation(
                self.user_id, conversation_id  # type: ignore[arg-type]
            )
        except ConversationError:
            conversation = self.conversation_service.start_conversation(self.user_id)
        return conversation.id

    def _snapshot(self, conversation_id: int, status: str) -> DashboardSnapshot:
        messages = self.conversation_service.list_messages(
            self.user_id, conversation_id
        )
        cart = self.grocery_service.get_cart(self.user_id)
        orders = self.grocery_service.get_recent_orders(
            self.user_id, limit=DASHBOARD_ORDER_LIMIT
        )
        preferences = self.grocery_service.list_preferences(self.user_id)

        return DashboardSnapshot(
            conversation_id=conversation_id,
            chat_messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            cart_rows=[
                [item.product_name, item.quantity, item.unit or ""]
                for item in cart.items
            ],
            order_rows=[
                [
                    order.id,
                    order.ordered_at.astimezone(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                    "; ".join(_describe_item(item) for item in order.items),
                ]
                for order in orders
            ],
            preference_rows=[
                [preference.subject, preference.attribute, preference.value]
                for preference in preferences
            ],
            status=status,
        )


def _describe_item(item: Any) -> str:
    quantity = f"{item.quantity:g}"
    unit = f" {item.unit}" if item.unit else ""
    return f"{quantity}{unit} {item.product_name}"


def _error_status(user_message: str, error: Exception) -> str:
    LOGGER.error(
        "%s (%s): %s",
        user_message,
        type(error).__name__,
        str(error).strip() or "No details",
        exc_info=True,
    )
    return user_message


def _success_status(turn: AgentTurn, prefix: str) -> str:
    if not turn.executed_tools:
        return prefix
    tools = ", ".join(turn.executed_tools)
    return f"{prefix} Herramientas utilizadas: {tools}."


def _snapshot_values(snapshot: DashboardSnapshot) -> Tuple[Any, ...]:
    return (
        snapshot.conversation_id,
        snapshot.chat_messages,
        snapshot.cart_rows,
        snapshot.order_rows,
        snapshot.preference_rows,
        snapshot.status,
    )


def _new_view(controller: GradioController) -> Tuple[Any, ...]:
    return _snapshot_values(controller.new_conversation())


def _refresh_view(
    controller: GradioController, conversation_id: object
) -> Tuple[Any, ...]:
    return _snapshot_values(controller.refresh(conversation_id))


def _submit_view(
    controller: GradioController, user_text: object, conversation_id: object
) -> Tuple[Any, ...]:
    return ("", None, "", None) + _snapshot_values(
        controller.submit_text(user_text, conversation_id)
    )


def _new_interaction_view(controller: GradioController) -> Tuple[Any, ...]:
    return ("", None, "", None) + _snapshot_values(
        controller.new_conversation()
    )


def _submit_voice_view(
    controller: GradioController, audio_path: object, conversation_id: object
) -> Tuple[Any, ...]:
    submission = controller.submit_voice(audio_path, conversation_id)
    return (
        "",
        None,
        submission.transcription,
        submission.audio,
    ) + _snapshot_values(submission.snapshot)


def create_gradio_app(controller: GradioController) -> gr.Blocks:
    """Build the interface without launching a server."""

    with gr.Blocks(title="Asistente de supermercado") as demo:
        conversation_state = gr.State()

        gr.Markdown(
            "# Asistente de supermercado\n"
            "Escribe o graba una solicitud para gestionar la cesta. SQLite "
            "sigue siendo la fuente de verdad."
        )
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    value=[],
                    label="Conversación",
                    height=420,
                    placeholder="Empieza pidiendo que añada un producto.",
                )
                with gr.Tabs():
                    with gr.Tab("Voz"):
                        voice_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Graba o sube una solicitud",
                            buttons=[],
                        )
                        voice_button = gr.Button(
                            "Enviar audio", variant="primary"
                        )
                        transcription_box = gr.Textbox(
                            label="Transcripción",
                            interactive=False,
                            placeholder="Aquí aparecerá la solicitud transcrita.",
                        )
                        voice_output = gr.Audio(
                            label="Respuesta de voz del asistente",
                            interactive=False,
                            autoplay=True,
                            format="wav",
                            buttons=["download"],
                        )
                        gr.Markdown(
                            "La voz de la respuesta del asistente ha sido "
                            "generada por IA."
                        )
                    with gr.Tab("Texto"):
                        message_box = gr.Textbox(
                            label="Mensaje",
                            placeholder="Por ejemplo: añade cuatro manzanas",
                            lines=2,
                        )
                        send_button = gr.Button("Enviar", variant="primary")
                with gr.Row():
                    new_button = gr.Button("Nueva conversación")
                    refresh_button = gr.Button("Actualizar datos")
            with gr.Column(scale=1):
                cart_table = gr.Dataframe(
                    value=[],
                    headers=["Producto", "Cantidad", "Unidad"],
                    datatype=["str", "number", "str"],
                    type="array",
                    label="Cesta actual",
                    interactive=False,
                    max_height=480,
                )

        with gr.Tabs():
            with gr.Tab("Historial de pedidos"):
                order_table = gr.Dataframe(
                    value=[],
                    headers=["Pedido", "Fecha", "Productos"],
                    datatype=["number", "str", "str"],
                    type="array",
                    interactive=False,
                )
            with gr.Tab("Preferencias recordadas"):
                preference_table = gr.Dataframe(
                    value=[],
                    headers=["Producto o tema", "Atributo", "Valor"],
                    datatype=["str", "str", "str"],
                    type="array",
                    interactive=False,
                )

        status = gr.Markdown("Iniciando una conversación local...")
        view_outputs = [
            conversation_state,
            chatbot,
            cart_table,
            order_table,
            preference_table,
            status,
        ]

        demo.load(
            fn=lambda: _new_view(controller),
            outputs=view_outputs,
            api_visibility="private",
        )
        interaction_outputs = [
            message_box,
            voice_input,
            transcription_box,
            voice_output,
        ] + view_outputs
        new_button.click(
            fn=lambda: _new_interaction_view(controller),
            outputs=interaction_outputs,
            api_visibility="private",
        )
        refresh_button.click(
            fn=lambda conversation_id: _refresh_view(controller, conversation_id),
            inputs=[conversation_state],
            outputs=view_outputs,
            api_visibility="private",
        )

        submit_callback = lambda text, conversation_id: _submit_view(
            controller, text, conversation_id
        )
        send_button.click(
            fn=submit_callback,
            inputs=[message_box, conversation_state],
            outputs=interaction_outputs,
            api_visibility="private",
            concurrency_id="grocery-agent-turn",
            concurrency_limit=1,
        )
        message_box.submit(
            fn=submit_callback,
            inputs=[message_box, conversation_state],
            outputs=interaction_outputs,
            api_visibility="private",
            concurrency_id="grocery-agent-turn",
            concurrency_limit=1,
        )
        voice_button.click(
            fn=lambda audio_path, conversation_id: _submit_voice_view(
                controller, audio_path, conversation_id
            ),
            inputs=[voice_input, conversation_state],
            outputs=interaction_outputs,
            api_visibility="private",
            concurrency_id="grocery-agent-turn",
            concurrency_limit=1,
        )

    return demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the grocery Gradio app")
    parser.add_argument(
        "--database",
        default=os.environ.get("GROCERY_DATABASE", "data/grocery.sqlite3"),
        help="SQLite database path",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("GROCERY_USER_ID", "demo-user"),
        help="Local demo user id",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_LLM_MODEL"),
        help="OpenAI model id (or set OPENAI_LLM_MODEL)",
    )
    parser.add_argument(
        "--transcription-model",
        default=os.environ.get("OPENAI_STT_MODEL", "gpt-transcribe"),
        help="OpenAI speech-to-text model (or set OPENAI_STT_MODEL)",
    )
    parser.add_argument(
        "--transcription-language",
        default=os.environ.get(
            "OPENAI_STT_LANGUAGE", DEFAULT_TRANSCRIPTION_LANGUAGE
        ),
        help=(
            "Speech-to-text language hint (or set OPENAI_STT_LANGUAGE; "
            "empty disables the hint)"
        ),
    )
    parser.add_argument(
        "--speech-model",
        default=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        help="OpenAI text-to-speech model (or set OPENAI_TTS_MODEL)",
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("OPENAI_TTS_VOICE", "alloy"),
        help="OpenAI text-to-speech voice (or set OPENAI_TTS_VOICE)",
    )
    parser.add_argument(
        "--speech-instructions",
        default=os.environ.get(
            "OPENAI_TTS_INSTRUCTIONS", DEFAULT_SPEECH_INSTRUCTIONS
        ),
        help=(
            "Text-to-speech delivery instructions (or set "
            "OPENAI_TTS_INSTRUCTIONS; empty omits the API field)"
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server interface (defaults to local access only)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local app in the default browser",
    )
    return parser


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if not arguments.model:
        parser.error("provide --model or set OPENAI_LLM_MODEL")

    database = Database(Path(arguments.database))
    database.initialize()
    grocery_service = GroceryService(database)
    conversation_service = ConversationService(database)
    client = OpenAI()
    agent = TextAgent(
        client=client,
        model=arguments.model,
        grocery_service=grocery_service,
        conversation_service=conversation_service,
    )
    speech_service = OpenAISpeechService(
        client=client,
        transcription_model=arguments.transcription_model,
        speech_model=arguments.speech_model,
        voice=arguments.voice,
        transcription_language=arguments.transcription_language,
        speech_instructions=arguments.speech_instructions,
    )
    controller = GradioController(
        agent=agent,
        grocery_service=grocery_service,
        conversation_service=conversation_service,
        user_id=arguments.user,
        speech_service=speech_service,
    )
    demo = create_gradio_app(controller)
    demo.launch(
        server_name=arguments.host,
        server_port=arguments.port,
        inbrowser=arguments.open_browser,
        share=False,
        show_error=False,
    )


if __name__ == "__main__":
    main()
