"""Small OpenAI speech-to-text and text-to-speech boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union


AudioPath = Union[str, Path]
DEFAULT_TRANSCRIPTION_LANGUAGE = "es"
DEFAULT_SPEECH_INSTRUCTIONS = (
    "Habla en español de forma natural, clara y concisa."
)


class SpeechError(RuntimeError):
    """Raised when local audio input or an audio response is unusable."""


class OpenAISpeechService:
    """Use OpenAI audio endpoints without owning conversation state."""

    def __init__(
        self,
        client: Any,
        transcription_model: str = "gpt-transcribe",
        speech_model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        transcription_language: Optional[str] = DEFAULT_TRANSCRIPTION_LANGUAGE,
        speech_instructions: Optional[str] = DEFAULT_SPEECH_INSTRUCTIONS,
    ) -> None:
        self.client = client
        self.transcription_model = _required_setting(
            transcription_model, "transcription model"
        )
        self.speech_model = _required_setting(speech_model, "speech model")
        self.voice = _required_setting(voice, "speech voice")
        self.transcription_language = _optional_setting(
            transcription_language, "transcription language"
        )
        self.speech_instructions = _optional_setting(
            speech_instructions, "speech instructions"
        )

    def transcribe(self, audio_path: AudioPath) -> str:
        """Return normalized text from one completed audio recording."""

        if not isinstance(audio_path, (str, Path)) or not str(audio_path).strip():
            raise SpeechError("Graba o sube un audio antes de enviarlo.")
        path = Path(audio_path)
        if not path.is_file():
            raise SpeechError("El archivo de audio ya no está disponible.")

        try:
            audio_file = path.open("rb")
        except OSError as error:
            raise SpeechError(
                f"No se pudo leer el audio grabado: {error}"
            ) from error
        with audio_file:
            request = {
                "model": self.transcription_model,
                "file": audio_file,
            }
            if self.transcription_language is not None:
                request["language"] = self.transcription_language
            response = self.client.audio.transcriptions.create(**request)

        raw_text = response if isinstance(response, str) else getattr(
            response, "text", ""
        )
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if not text:
            raise SpeechError("No se detectó voz en la grabación.")
        return text

    def synthesize(self, text: str) -> bytes:
        """Generate a complete WAV response suitable for Gradio playback."""

        if not isinstance(text, str) or not text.strip():
            raise SpeechError(
                "No se puede generar voz a partir de una respuesta vacía."
            )
        normalized_text = text.strip()
        if len(normalized_text) > 4096:
            raise SpeechError(
                "La respuesta es demasiado larga para generar la voz."
            )

        request = {
            "model": self.speech_model,
            "voice": self.voice,
            "input": normalized_text,
            "response_format": "wav",
        }
        if self.speech_instructions is not None:
            request["instructions"] = self.speech_instructions
        response = self.client.audio.speech.create(**request)
        content = getattr(response, "content", b"")
        if not isinstance(content, (bytes, bytearray)) or not content:
            raise SpeechError("La generación de voz no devolvió ningún audio.")
        return bytes(content)


def _required_setting(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"The {label} must not be empty")
    return value.strip()


def _optional_setting(value: object, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"The {label} must be text or None")
    return value.strip() or None
