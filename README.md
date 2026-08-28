# Grocery voice agent

A functional Spanish voice-to-voice grocery-ordering prototype built for the
requirements in [`docs/Task.txt`](docs/Task.txt). The application uses OpenAI
for speech recognition, conversational reasoning, and speech generation, while
Python and SQLite remain authoritative for carts, orders, conversation history,
and persistent user memory.

## What the prototype supports

- Push-to-talk microphone input and uploaded audio.
- Spanish transcription and spoken responses by default.
- Adding, removing, updating, and reviewing cart items.
- Confirming a cart as an immutable order.
- Reusing previous orders and deriving usual products and quantities.
- Remembering explicit preferences across conversations.
- A Spanish Gradio interface showing the transcript, current cart, recent
  orders, stored preferences, and agent status.

## Quick start

Use Python 3.10 or newer. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Configure credentials in the environment rather than in a repository file.
The conversational model must support OpenAI function calling:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_LLM_MODEL = "your-model-id"
```

Start the local interface:

```powershell
.venv\Scripts\python -m grocery_agent.gradio_app --open-browser
```

Without `--open-browser`, visit <http://127.0.0.1:7860>. Press `Ctrl+C` in the
terminal to stop the server. Use `--port 7861` when the default port is busy.

The default database is `data/grocery.sqlite3` and the default local user is
`demo-user`. Both are configurable for an isolated demo run:

```powershell
.venv\Scripts\python -m grocery_agent.gradio_app `
  --database data/demo-candidate.sqlite3 `
  --user candidate-demo `
  --port 7861 `
  --open-browser
```

The app binds to `127.0.0.1`, disables Gradio sharing, and provides no login.
Database files, virtual environments, caches, and package build artifacts are
excluded by `.gitignore`.

## Audio configuration

These optional environment variables show the application defaults:

```powershell
$env:OPENAI_STT_MODEL = "gpt-transcribe"
$env:OPENAI_STT_LANGUAGE = "es"
$env:OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
$env:OPENAI_TTS_VOICE = "alloy"
$env:OPENAI_TTS_INSTRUCTIONS = "Habla en español de forma natural, clara y concisa."
```

Equivalent command-line flags are `--transcription-model`,
`--transcription-language`, `--speech-model`, `--voice`, and
`--speech-instructions`. An empty language or instruction value omits that
optional field from the API request. Models such as `tts-1` and `tts-1-hd` do
not accept speech instructions, so omit them when selecting those models:

```powershell
.venv\Scripts\python -m grocery_agent.gradio_app `
  --speech-model tts-1 `
  --speech-instructions ""
```

The audio parameters follow the official OpenAI
[transcription](https://developers.openai.com/api/reference/python/resources/audio/subresources/transcriptions/methods/create)
and [speech generation](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create)
references.

## Architecture

One completed voice turn follows this path:

```text
Microphone / audio file
        |
        v
OpenAI speech-to-text
        |
        v
TextAgent <----> OpenAI Responses API
    |
    +---- native function calls ----> GroceryToolDispatcher
                                         |
                                         v
                                  GroceryService / SQLite
    |
    v
Assistant text ----> OpenAI text-to-speech ----> Gradio audio player
```

The Gradio browser state contains only the current conversation identifier and
ephemeral input/output values. Every callback reloads the visible transcript,
cart, recent orders, and preferences from SQLite.

### Source-of-truth boundaries

- `TextAgent` lets the model interpret requests and select strict tools.
- `GroceryToolDispatcher` allow-lists tool names, validates arguments, and binds
  calls to the locally configured user; the model never supplies a user ID.
- `GroceryService` owns cart, order, and structured-memory mutations.
- `ConversationService` owns user-visible conversation history.
- OpenAI-hosted conversation storage is disabled with `store=False`.

The model cannot directly write application tables or declare a mutation
successful. Tool results use explicit `{ "ok": true/false }` envelopes, and the
instructions require `ok=true` before reporting success.

## Persistence and memory

SQLite stores:

- One active cart per user.
- Immutable order snapshots with UTC timestamps, products, quantities, and
  units.
- Explicit preferences keyed by user, subject, and attribute.
- Conversations and visible user/assistant messages.

Frequently purchased items and usual quantities are derived from confirmed
orders instead of stored as independently mutable facts. For an underspecified
product, the agent checks explicit preferences and purchase frequency before
proposing a specific item. Explicit preferences take priority, and the cart is
not changed until the user confirms the proposal.

## Important design decisions

### Completed-turn audio instead of realtime streaming

The voice loop is speech-to-text, agent, then text-to-speech. It is easier to
inspect, test, and explain than a WebRTC or WebSocket session. The tradeoff is
push-to-talk interaction and one complete round trip per turn.

### Structured SQLite memory instead of a vector database

The assignment's remembered facts—orders, quantities, brands, and preferences—
are structured and require exact retrieval and mutation. SQLite keeps these
facts explicit, inspectable, and deterministic. Semantic product matching is a
known limitation rather than hidden inside an embedding store.

### Short database transactions

Each successful tool call commits independently. No SQLite transaction remains
open while waiting for a network response. If a later API call fails, committed
state remains visible and the interface asks the user to review the cart before
retrying.

### Sequential tool execution

`parallel_tool_calls=False` keeps multi-tool behavior ordered and easier to
reason about. The agent has a bounded number of tool-call rounds.

## Testing

Run the complete deterministic suite:

```powershell
.venv\Scripts\python -m pytest
```

The suite uses temporary SQLite databases and scripted OpenAI boundaries. It
does not require credentials and does not make paid OpenAI calls. It covers:

- Domain validation, atomic order confirmation, and user isolation.
- Order reuse, derived purchase frequency, and preference persistence.
- Strict tool schemas, dispatch errors, and Responses API orchestration.
- Conversation persistence and bounded model context.
- Speech request construction and failure handling.
- Gradio state refresh, reset behavior, localized copy, and bounded order
  display.

Run only the assignment-derived Spanish acceptance scenarios with:

```powershell
.venv\Scripts\python -m pytest tests\test_acceptance_es.py -q
```

These scripted scenarios validate the application loop and persistent state,
not a particular model's Spanish interpretation. Use the separate
[live microphone checklist](docs/acceptance_scenarios_es.md) to validate the
real speech-to-text, model, and text-to-speech boundary.

## Known limitations

- This is a local single-user prototype with no authentication or authorization
  layer beyond user-scoped service queries.
- Products are identified by normalized free text; there is no catalog, SKU
  resolution, synonym handling, pricing, inventory, payment, or fulfillment.
- Purchase frequency is derived from all confirmed order history rather than a
  configurable recency window or minimum purchase threshold.
- Tool calls within a multi-action turn commit separately, so a later failure
  can leave earlier actions applied. The UI reloads SQLite and asks the user to
  inspect the current cart.
- Conversation and order data are local SQLite records. The dashboard renders
  only the 20 most recent orders, although all confirmed orders remain stored.
- The automated tests use a scripted model boundary; live model behavior is
  verified manually.

## Project layout

```text
src/grocery_agent/
  agent.py          Responses API and native function-calling loop
  database.py       SQLite schema and connection management
  gradio_app.py     Spanish controller, interface, and application entry point
  history.py        Persistent conversation service
  models.py         Immutable returned application values
  service.py        Cart, order, and structured-memory use cases
  speech.py         OpenAI speech-to-text and text-to-speech boundary
  tools.py          Strict tool definitions and allow-listed dispatcher
tests/               Deterministic unit, integration, UI, and acceptance tests
docs/                Original assignment and live Spanish acceptance checklist
```
