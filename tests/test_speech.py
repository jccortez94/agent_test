from types import SimpleNamespace

import pytest

from grocery_agent.speech import (
    DEFAULT_SPEECH_INSTRUCTIONS,
    DEFAULT_TRANSCRIPTION_LANGUAGE,
    OpenAISpeechService,
    SpeechError,
)


class FakeTranscriptions:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **request):
        request["file_name"] = request["file"].name
        request["file_bytes"] = request["file"].read()
        del request["file"]
        self.requests.append(request)
        return self.response


class FakeSpeech:
    def __init__(self, content=b"generated-wav"):
        self.content = content
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=self.content)


class FakeClient:
    def __init__(self, transcription_response=None, speech_content=b"generated-wav"):
        self.audio = SimpleNamespace(
            transcriptions=FakeTranscriptions(
                transcription_response or SimpleNamespace(text="Add four apples")
            ),
            speech=FakeSpeech(speech_content),
        )


def test_transcription_sends_the_file_and_returns_normalized_text(tmp_path):
    audio_path = tmp_path / "request.webm"
    audio_path.write_bytes(b"recorded-audio")
    client = FakeClient(SimpleNamespace(text="  Add oat milk.  "))
    service = OpenAISpeechService(client, transcription_model="test-stt")

    text = service.transcribe(audio_path)

    assert text == "Add oat milk."
    assert client.audio.transcriptions.requests == [
        {
            "model": "test-stt",
            "language": DEFAULT_TRANSCRIPTION_LANGUAGE,
            "file_name": str(audio_path),
            "file_bytes": b"recorded-audio",
        }
    ]


def test_transcription_accepts_a_plain_text_api_response(tmp_path):
    audio_path = tmp_path / "request.wav"
    audio_path.write_bytes(b"audio")
    service = OpenAISpeechService(FakeClient("six bananas"))

    assert service.transcribe(str(audio_path)) == "six bananas"


@pytest.mark.parametrize("audio_path", [None, "", 123])
def test_transcription_requires_an_audio_path(audio_path):
    service = OpenAISpeechService(FakeClient())

    with pytest.raises(SpeechError, match="Graba o sube"):
        service.transcribe(audio_path)


def test_transcription_rejects_a_missing_file(tmp_path):
    service = OpenAISpeechService(FakeClient())

    with pytest.raises(SpeechError, match="ya no está disponible"):
        service.transcribe(tmp_path / "missing.wav")


def test_transcription_rejects_an_empty_result(tmp_path):
    audio_path = tmp_path / "silence.wav"
    audio_path.write_bytes(b"audio")
    service = OpenAISpeechService(FakeClient(SimpleNamespace(text="  ")))

    with pytest.raises(SpeechError, match="No se detectó voz"):
        service.transcribe(audio_path)


def test_synthesis_requests_wav_and_returns_bytes():
    client = FakeClient(speech_content=bytearray(b"wav-bytes"))
    service = OpenAISpeechService(
        client, speech_model="test-tts", voice="coral"
    )

    audio = service.synthesize("  Order updated.  ")

    assert audio == b"wav-bytes"
    assert client.audio.speech.requests == [
        {
            "model": "test-tts",
            "voice": "coral",
            "input": "Order updated.",
            "response_format": "wav",
            "instructions": DEFAULT_SPEECH_INSTRUCTIONS,
        }
    ]


def test_audio_language_and_instructions_can_be_overridden(tmp_path):
    audio_path = tmp_path / "request.wav"
    audio_path.write_bytes(b"audio")
    client = FakeClient()
    service = OpenAISpeechService(
        client,
        transcription_language="pt",
        speech_instructions="Fale em português com naturalidade.",
    )

    service.transcribe(audio_path)
    service.synthesize("Pedido atualizado.")

    assert client.audio.transcriptions.requests[0]["language"] == "pt"
    assert (
        client.audio.speech.requests[0]["instructions"]
        == "Fale em português com naturalidade."
    )


@pytest.mark.parametrize("disabled_value", [None, "", "   "])
def test_optional_audio_configuration_is_omitted(disabled_value, tmp_path):
    audio_path = tmp_path / "request.wav"
    audio_path.write_bytes(b"audio")
    client = FakeClient()
    service = OpenAISpeechService(
        client,
        transcription_language=disabled_value,
        speech_instructions=disabled_value,
    )

    service.transcribe(audio_path)
    service.synthesize("Pedido actualizado.")

    assert "language" not in client.audio.transcriptions.requests[0]
    assert "instructions" not in client.audio.speech.requests[0]


@pytest.mark.parametrize("text", [None, "", "   "])
def test_synthesis_rejects_empty_text(text):
    service = OpenAISpeechService(FakeClient())

    with pytest.raises(SpeechError, match="respuesta vacía"):
        service.synthesize(text)


def test_synthesis_rejects_text_over_the_endpoint_limit():
    service = OpenAISpeechService(FakeClient())

    with pytest.raises(SpeechError, match="demasiado larga"):
        service.synthesize("x" * 4097)


def test_synthesis_rejects_an_empty_audio_response():
    service = OpenAISpeechService(FakeClient(speech_content=b""))

    with pytest.raises(SpeechError, match="no devolvió ningún audio"):
        service.synthesize("Hello")


@pytest.mark.parametrize(
    ("keyword", "value", "label"),
    [
        ("transcription_model", "", "transcription model"),
        ("speech_model", " ", "speech model"),
        ("voice", None, "speech voice"),
    ],
)
def test_configuration_values_must_not_be_empty(keyword, value, label):
    arguments = {keyword: value}

    with pytest.raises(ValueError, match=label):
        OpenAISpeechService(FakeClient(), **arguments)


@pytest.mark.parametrize(
    ("keyword", "label"),
    [
        ("transcription_language", "transcription language"),
        ("speech_instructions", "speech instructions"),
    ],
)
def test_optional_configuration_values_must_be_text_or_none(keyword, label):
    with pytest.raises(ValueError, match=label):
        OpenAISpeechService(FakeClient(), **{keyword: 123})
