# 🧙 WizardAI SDK — Complete Documentation

[![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)](https://pypi.org/project/wizardai-sdk/)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-orange?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/badge/pip_install-wizardai--sdk-purple?style=flat-square)](https://pypi.org/project/wizardai-sdk/)

> **A powerful, all-in-one Python SDK for AI integration** — combining conversational AI, computer vision, speech I/O, memory management, and a flexible plugin system into a single, easy-to-use module. Works with OpenAI, Anthropic, Hugging Face, local models (Ollama, LM Studio), and any OpenAI-compatible endpoint.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Reference](#4-module-reference)
   - 4.1 [WizardAI Core (`core.py`)](#41-wizardai-core)
   - 4.2 [AIClient (`ai_client.py`)](#42-aiclient)
   - 4.3 [ConversationAgent (`conversation.py`)](#43-conversationagent)
   - 4.4 [MemoryManager (`memory.py`)](#44-memorymanager)
   - 4.5 [VisionModule (`vision.py`)](#45-visionmodule)
   - 4.6 [SpeechModule (`speech.py`)](#46-speechmodule)
   - 4.7 [Plugin System (`plugins.py`)](#47-plugin-system)
   - 4.8 [Exceptions (`exceptions.py`)](#48-exceptions)
   - 4.9 [Utilities (`utils.py`)](#49-utilities)
5. [Configuration Reference](#5-configuration-reference)
6. [Environment Variables](#6-environment-variables)
7. [AI Backend Guide](#7-ai-backend-guide)
8. [Error Handling](#8-error-handling)
9. [Advanced Usage](#9-advanced-usage)
10. [Publishing to PyPI](#10-publishing-to-pypi)
11. [Project Structure](#11-project-structure)
12. [Contributing](#12-contributing)
13. [License](#13-license)

---

## 1. Installation

### Minimal install (core + AI backends only)

```bash
pip install wizardai-sdk
```

### Feature-specific installs

```bash
# Computer vision (OpenCV)
pip install "wizardai-sdk[vision]"

# Speech recognition + TTS
pip install "wizardai-sdk[speech]"

# High-quality offline speech recognition (OpenAI Whisper)
pip install "wizardai-sdk[whisper]"

# Everything at once
pip install "wizardai-sdk[full]"

# Developer tools (testing, linting, formatting)
pip install "wizardai-sdk[dev]"
```

### From source

```bash
git clone https://github.com/VIkasPatel22/wizardai-sdk.git
cd wizardai-sdk
pip install -e ".[full]"
```

### System dependencies for speech

Some speech features require OS-level libraries:

```bash
# Linux
sudo apt-get install portaudio19-dev python3-dev

# macOS
brew install portaudio

# Windows — use the prebuilt wheel
pip install pipwin && pipwin install pyaudio
```

---

## 2. Quick Start

### Minimal example

```python
import wizardai

wiz = wizardai.WizardAI(openai_api_key="sk-...")
wiz.start()

# Rule-based chat (no API call)
wiz.agent.add_pattern("hello", "Hello from WizardAI!")
print(wiz.chat("hello"))          # → "Hello from WizardAI!"

# LLM call
print(wiz.ask("What is the speed of light?"))

wiz.stop()
```

### Context manager (recommended)

```python
with wizardai.WizardAI(openai_api_key="sk-...") as wiz:
    print(wiz.ask("Tell me a joke."))
```

### Full features example

```python
wiz = wizardai.WizardAI(
    openai_api_key="sk-...",
    enable_vision=True,
    enable_speech=True,
    stt_backend="google",
    tts_backend="pyttsx3",
    memory_path="session.json",
    system_prompt="You are a helpful assistant.",
)
wiz.start()

# Multimodal: capture + describe an image
frame   = wiz.capture()
b64     = wiz.vision.encode_to_base64(frame)
caption = wiz.ask("Describe this image.", image_b64=b64)
wiz.say(caption)

# Streaming response
for chunk in wiz.ai.chat_stream([{"role": "user", "content": "Write a poem"}]):
    print(chunk, end="", flush=True)

# Long-term memory
wiz.remember("user_name", "Alice")
print(wiz.recall("user_name"))    # → "Alice"

wiz.stop()
```

### Interactive terminal REPL

```python
wiz = wizardai.WizardAI(openai_api_key="sk-...")
wiz.start()
wiz.run_repl()                    # text mode
wiz.run_repl(voice_mode=True)     # voice I/O mode
```

---

## 3. Architecture Overview

```
wizardai/
├── core.py            ← WizardAI (top-level orchestrator)
│     ├── ai_client.py       ← AIClient  (multi-backend LLM)
│     ├── conversation.py    ← ConversationAgent (pattern matching)
│     ├── memory.py          ← MemoryManager (history + long-term)
│     ├── vision.py          ← VisionModule (camera / OpenCV)
│     ├── speech.py          ← SpeechModule (STT + TTS)
│     ├── plugins.py         ← PluginBase + PluginManager
│     ├── exceptions.py      ← Custom exception hierarchy
│     └── utils.py           ← Logger, FileHelper, DataSerializer
```

**Chat pipeline priority** (inside `wiz.chat()`):

```
User input
    │
    ├─ 1. Plugin dispatch   → if a plugin handles it, return response
    ├─ 2. Pattern matching  → if an agent rule matches, return response
    └─ 3. LLM fallback      → call AI backend if no pattern matched
```

---

## 4. Module Reference

---

### 4.1 WizardAI Core

**File:** `wizardai/core.py`  
**Class:** `WizardAI`

The top-level orchestrator. Wires all sub-modules together and exposes convenience shortcuts for the most common operations.

#### Constructor

```python
WizardAI(
    # AI backend
    ai_backend="openai",              # str | AIBackend
    openai_api_key=None,              # str — or set OPENAI_API_KEY
    anthropic_api_key=None,           # str — or set ANTHROPIC_API_KEY
    huggingface_api_key=None,         # str — or set HUGGINGFACE_API_KEY
    custom_endpoint=None,             # str — URL for self-hosted models
    default_model=None,               # str — overrides backend default
    max_tokens=1024,                  # int
    temperature=0.7,                  # float

    # Vision
    enable_vision=False,              # bool
    camera_device=0,                  # int — OpenCV device index
    camera_width=640,                 # int
    camera_height=480,                # int

    # Speech
    enable_speech=False,              # bool
    stt_backend="google",             # "google" | "sphinx" | "whisper"
    tts_backend="pyttsx3",            # "pyttsx3" | "gtts" | "elevenlabs"
    language="en-US",                 # BCP-47 language code

    # Conversation agent
    agent_name="WizardBot",           # str
    fallback_response="I'm not sure how to respond to that.",

    # Memory
    max_history=50,                   # int — sliding window size
    memory_path=None,                 # str | Path — persistent storage

    # System
    system_prompt=None,               # str — default LLM system prompt
    log_level="INFO",                 # "DEBUG"|"INFO"|"WARNING"|"ERROR"
    log_file=None,                    # str — optional log file path
    data_dir="./wizardai_data",       # str — working directory
)
```

#### Session lifecycle

```python
wiz.start()   # opens camera, initialises speech, calls plugin.on_start()
wiz.stop()    # stops all modules, saves memory, calls plugin.on_stop()

# Or use as context manager:
with WizardAI(...) as wiz:
    ...
```

#### Chat methods

| Method | Description |
|--------|-------------|
| `wiz.chat(text)` | Full pipeline: plugins → patterns → LLM fallback |
| `wiz.ask(prompt, **kwargs)` | Direct LLM call, bypasses pattern matching |
| `wiz.ask_raw(prompt)` | Like `ask()` but returns full `AIResponse` object |

```python
# chat — goes through full pipeline
reply = wiz.chat("hello world")

# ask — always hits the LLM
reply = wiz.ask("What is quantum entanglement?")

# ask with options
reply = wiz.ask(
    "Summarise this.",
    model="gpt-4o",
    max_tokens=500,
    temperature=0.3,
    system_prompt="You are a concise summariser.",
    include_history=False,    # don't send conversation history
    image_b64=b64_string,     # for multimodal models
)
```

#### Speech shortcuts

```python
text = wiz.listen(timeout=5.0)          # STT: microphone → text
wiz.say("Hello!")                        # TTS: text → audio
reply = wiz.voice_chat(timeout=5.0)     # listen + chat + say, returns reply
```

#### Vision shortcuts

```python
frame = wiz.capture()                   # returns numpy ndarray
path  = wiz.snapshot("photo.jpg")       # capture and save to disk
```

#### Memory shortcuts

```python
wiz.remember("key", value)              # store in long-term memory
value = wiz.recall("key", default=None) # retrieve from long-term memory
history = wiz.get_history(n=10)         # last n turns as list of dicts
```

#### Plugin shortcuts

```python
wiz.add_plugin(MyPlugin, config={"key": "val"})
wiz.load_plugins_from_dir("./my_plugins/")
```

#### Configuration helpers

```python
wiz.set_system_prompt("You are a pirate.")
wiz.set_model("gpt-4o")
wiz.set_api_key("sk-new-key")
```

#### Interactive REPL

```python
wiz.run_repl()                          # keyboard input
wiz.run_repl(voice_mode=True)           # microphone input
wiz.run_repl(
    prompt_str="Me: ",
    quit_commands=["quit", "exit", "bye"],
)
```

---

### 4.2 AIClient

**File:** `wizardai/ai_client.py`  
**Classes:** `AIBackend`, `AIResponse`, `AIClient`

Unified interface for OpenAI, Anthropic, Hugging Face, and any custom OpenAI-compatible endpoint. Handles retries, rate limiting, and streaming.

#### `AIBackend` enum

```python
from wizardai import AIBackend

AIBackend.OPENAI        # "openai"
AIBackend.ANTHROPIC     # "anthropic"
AIBackend.HUGGINGFACE   # "huggingface"
AIBackend.CUSTOM        # "custom"
```

#### `AIResponse` dataclass

```python
response.text           # str — generated text
response.model          # str — model used
response.backend        # AIBackend
response.usage          # dict — token usage stats
response.raw            # dict — raw API response
response.latency_ms     # float — round-trip time in ms
str(response)           # same as response.text
```

#### Constructor

```python
from wizardai import AIClient

client = AIClient(
    backend="openai",           # str | AIBackend
    api_key="sk-...",           # falls back to env var
    model="gpt-4o-mini",        # default model
    endpoint=None,              # custom endpoint URL
    max_retries=3,              # retry attempts on transient errors
    retry_delay=1.0,            # initial delay (doubles each attempt)
    timeout=30.0,               # HTTP timeout in seconds
    rate_limit_calls=60,        # max calls per window
    rate_limit_period=60.0,     # window in seconds
)
```

#### Methods

```python
# Single-turn completion
response = client.complete("Write a haiku.")
print(response.text)
print(response.usage)         # {"prompt_tokens": 12, "completion_tokens": 17}
print(response.latency_ms)    # e.g. 834.2

# Multi-turn chat
messages = [
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user",      "content": "What is my name?"},
]
response = client.chat(messages, system_prompt="You are helpful.")
print(response.text)          # "Your name is Alice."

# Streaming chat
for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)

# Streaming completion
for chunk in client.complete_stream("Tell me a story."):
    print(chunk, end="", flush=True)

# Runtime updates
client.set_model("gpt-4o")
client.set_api_key("sk-new...")
client.set_endpoint("http://localhost:11434/v1/chat/completions")
```

#### Default models per backend

| Backend | Default Model |
|---------|--------------|
| `openai` | `gpt-4o-mini` |
| `anthropic` | `claude-3-5-haiku-20241022` |
| `huggingface` | `mistralai/Mistral-7B-Instruct-v0.2` |
| `custom` | `default` |

#### Supported endpoints

| Provider | Endpoint URL | Notes |
|----------|-------------|-------|
| OpenAI | `https://api.openai.com/v1/chat/completions` | Requires `OPENAI_API_KEY` |
| Anthropic | Use `backend="anthropic"` | Uses native Anthropic API |
| Ollama (local) | `http://localhost:11434/v1/chat/completions` | No key needed |
| LM Studio | `http://localhost:1234/v1/chat/completions` | No key needed |
| Groq | `https://api.groq.com/openai/v1/chat/completions` | Requires API key |
| Together AI | `https://api.together.xyz/v1/chat/completions` | Requires API key |
| Azure OpenAI | Your Azure deployment URL | Requires Azure key |

---

### 4.3 ConversationAgent

**File:** `wizardai/conversation.py`  
**Classes:** `Pattern`, `ConversationAgent`

AIML-style rule-based chat engine with wildcard patterns, priorities, context awareness, callable templates, and memory integration.

#### `Pattern` dataclass

```python
from wizardai import Pattern

Pattern(
    pattern="my name is *",     # input pattern (supports wildcards)
    template="Hi {wildcard}!",  # response: str, callable, or list of str
    priority=0,                 # higher wins when patterns overlap
    context=None,               # optional context key this rule requires
    tags=[],                    # arbitrary labels for grouping
)
```

**Wildcard syntax:**

| Wildcard | Matches | Example |
|----------|---------|---------|
| `*` | One or more words | `"tell me about *"` |
| `?` | Exactly one word | `"is ? available"` |
| `{name}` | Named capture group | `"weather in {city}"` |

#### Constructor

```python
from wizardai import ConversationAgent, MemoryManager

agent = ConversationAgent(
    name="WizardBot",
    fallback="I'm not sure how to respond to that.",
    memory=MemoryManager(),       # optional — creates new one by default
    case_sensitive=False,
)
```

#### Adding patterns

```python
# Simple string response
agent.add_pattern("hello", "Hello there!")

# Wildcard with substitution
agent.add_pattern("my name is *", "Nice to meet you, {wildcard}!")

# Named capture group
agent.add_pattern("weather in {city}", "Checking weather for {city}...")

# Callable template (dynamic response)
import random
agent.add_pattern(
    "tell me a joke",
    lambda text, ctx: random.choice([
        "Why do Python devs wear glasses? Because they can't C!",
        "I told a UDP joke. You might not get it.",
    ])
)

# List of alternatives (randomly selected)
agent.add_pattern("how are you", [
    "Doing great, thanks!",
    "I'm running at 100% efficiency.",
    "Excellent — ready to help!",
])

# With priority (higher wins when patterns overlap)
agent.add_pattern("hello world", "Special greeting!", priority=10)

# Context-aware rule (only triggers when context is active)
agent.add_pattern("yes", "Great, proceeding!", context="confirm_action")
agent.add_pattern("no",  "OK, cancelling.",    context="confirm_action")
agent.set_context("confirm_action")   # activates the context

# Using Pattern object directly (full control)
agent.add_pattern_obj(Pattern(
    pattern="translate * to {lang}",
    template="Translating for you...",
    priority=5,
    tags=["translate", "language"],
))
```

#### Responding

```python
reply = agent.respond("hello")                    # → "Hello there!"
reply = agent.respond("my name is Bob")           # → "Nice to meet you, Bob!"
reply = agent.respond("tell me a joke")           # → random joke
```

#### Management

```python
agent.list_patterns()             # returns list of Pattern objects
agent.remove_pattern("hello")     # remove by pattern string
agent.clear_patterns()            # remove all patterns
agent.set_context("my_context")   # activate a context key
agent.clear_context()             # deactivate current context
```

#### Pre/post processors

```python
# Preprocess input before matching
agent.add_preprocessor(lambda text: text.strip().lower())

# Postprocess output before returning
agent.add_postprocessor(lambda text: text + " 😊")
```

---

### 4.4 MemoryManager

**File:** `wizardai/memory.py`  
**Classes:** `Message`, `MemoryManager`

Provides short-term conversation history (sliding window) and long-term key-value storage, with optional JSON disk persistence.

#### `Message` object

```python
msg.role        # str — "user" | "assistant" | "system"
msg.content     # str — message text
msg.timestamp   # float — Unix timestamp
msg.metadata    # dict — arbitrary extra data

msg.to_dict()                   # serialize to dict
Message.from_dict(data)         # deserialize from dict
```

#### Constructor

```python
from wizardai import MemoryManager

mem = MemoryManager(
    max_history=50,             # sliding window size
    persist_path="mem.json",    # auto-save on every write (optional)
)
```

#### Short-term memory (conversation history)

```python
# Add messages
msg = mem.add_message("user",      "What's the capital of France?")
msg = mem.add_message("assistant", "Paris!")
msg = mem.add_message("system",    "You are helpful.", metadata={"source": "init"})

# Retrieve history
msgs = mem.get_history()                        # all messages (Message objects)
msgs = mem.get_history(n=5)                     # last 5 messages
msgs = mem.get_history(role_filter="user")      # only user messages

# As dicts (for serialization)
dicts = mem.get_history_as_dicts(n=10)

# As API-ready payload (for OpenAI / Anthropic)
api_msgs = mem.get_messages_for_api()           # [{"role": ..., "content": ...}]
api_msgs = mem.get_messages_for_api(n=5, include_system=False)

# Other utilities
last = mem.last_message()                       # most recent Message
last = mem.last_message(role="user")            # most recent user message
results = mem.search_history("France", top_k=3) # [(Message, score), ...]
mem.clear_history()                             # wipe all messages
```

#### Long-term memory (key-value store)

```python
# Store
mem.remember("user_name", "Alice")
mem.remember("preferences", {"theme": "dark", "lang": "en"})
mem.remember("visit_count", 42)

# Retrieve
name  = mem.recall("user_name")               # → "Alice"
prefs = mem.recall("preferences")             # → {"theme": "dark", ...}
val   = mem.recall("missing_key", default=0)  # → 0

# Delete
was_deleted = mem.forget("user_name")         # → True
keys = mem.list_memories()                    # → ["preferences", "visit_count"]
```

#### Ephemeral context (not persisted)

```python
mem.set_context("current_topic", "weather")
topic = mem.get_context("current_topic")      # → "weather"
mem.clear_context()
```

#### Persistence

```python
mem.save()                        # save to persist_path
mem.save("backup.json")           # save to specific path
mem.load()                        # load from persist_path
mem.load("backup.json")           # load from specific path
```

---

### 4.5 VisionModule

**File:** `wizardai/vision.py`  
**Class:** `VisionModule`

Real-time camera access and image processing using OpenCV. All OpenCV imports are deferred, so the rest of WizardAI works even without `opencv-python` installed.

**Requires:** `pip install "wizardai-sdk[vision]"`

#### Constructor

```python
from wizardai import VisionModule

cam = VisionModule(
    device_id=0,      # OpenCV camera index (0 = default webcam)
    width=1280,       # capture width in pixels
    height=720,       # capture height in pixels
    fps=30,           # requested frames per second
)
```

#### Camera lifecycle

```python
cam.open()           # opens camera device (raises CameraNotFoundError if unavailable)
cam.close()          # releases camera
cam.is_open()        # → True/False
```

#### Frame capture

```python
frame = cam.capture_frame()                   # → numpy.ndarray (BGR)
path  = cam.save_frame(frame, "output.jpg")   # save to disk → Path
b64   = cam.encode_to_base64(frame)           # → base64 string (JPEG)
```

#### Face detection

```python
# Returns list of dicts: [{"bbox": (x, y, w, h), "confidence": 0.9}, ...]
faces = cam.detect_faces(frame)
print(f"Found {len(faces)} face(s)")

for face in faces:
    x, y, w, h = face["bbox"]
    print(f"Face at ({x}, {y}), size {w}x{h}")
```

#### Frame streaming

```python
def on_frame(frame):
    faces = cam.detect_faces(frame)
    if faces:
        print(f"Detected {len(faces)} face(s)")

# Start streaming (non-blocking — runs in background thread)
cam.start_stream(callback=on_frame, show_preview=True)

import time
time.sleep(10)        # stream for 10 seconds

cam.stop_stream()
cam.close()
```

#### Complete example

```python
cam = VisionModule(device_id=0, width=1280, height=720)
cam.open()

frame = cam.capture_frame()
cam.save_frame(frame, "snapshot.jpg")

b64 = cam.encode_to_base64(frame)
# Now pass b64 to an LLM: wiz.ask("Describe this image.", image_b64=b64)

cam.close()
```

---

### 4.6 SpeechModule

**File:** `wizardai/speech.py`  
**Class:** `SpeechModule`

Speech recognition (STT) and text-to-speech (TTS) with multiple backend options.

**Requires:** `pip install "wizardai-sdk[speech]"` (and optionally `[whisper]`)

#### STT backends

| Backend | Type | Package | Notes |
|---------|------|---------|-------|
| `google` | Online | `SpeechRecognition` | Free, requires internet |
| `sphinx` | Offline | `pocketsphinx` | Lower accuracy |
| `whisper` | Offline | `openai-whisper` | High accuracy, slow on CPU |

#### TTS backends

| Backend | Type | Package | Notes |
|---------|------|---------|-------|
| `pyttsx3` | Offline | `pyttsx3` | Uses OS voices |
| `gtts` | Online | `gtts`, `pygame` | Google TTS, MP3 output |
| `elevenlabs` | Online | (requests) | High quality, requires API key |

#### Constructor

```python
from wizardai import SpeechModule

speech = SpeechModule(
    stt_backend="google",             # "google" | "sphinx" | "whisper"
    tts_backend="pyttsx3",            # "pyttsx3" | "gtts" | "elevenlabs"
    language="en-US",                 # BCP-47 language code
    tts_rate=150,                     # pyttsx3 words-per-minute
    tts_volume=1.0,                   # pyttsx3 volume (0.0–1.0)
    elevenlabs_api_key=None,          # or set ELEVENLABS_API_KEY
    elevenlabs_voice_id=None,         # ElevenLabs voice ID
)
```

#### Speech-to-Text (STT)

```python
# Listen from microphone
text = speech.listen(
    timeout=5.0,              # seconds to wait for speech start
    phrase_time_limit=15.0,   # max seconds to record per phrase
    adjust_noise=True,        # calibrate for ambient noise first
    device_index=None,        # microphone device index (None = default)
)
print("You said:", text)

# Transcribe from file
text = speech.transcribe_file("audio.wav")
text = speech.transcribe_file("audio.mp3")
```

#### Text-to-Speech (TTS)

```python
# Speak synchronously (blocks until done)
speech.say("Hello! I am WizardAI.")

# Non-blocking speech
speech.say("Processing your request...", blocking=False)

# Streaming word-by-word TTS
for word in speech.stream_say("Generating response token by token..."):
    print(word, end=" ", flush=True)
```

#### Continuous listening

```python
def on_speech(text):
    print("Heard:", text)
    reply = wiz.chat(text)
    speech.say(reply)

# Start background listener
speech.start_continuous_listening(callback=on_speech, phrase_time_limit=10)

# ... do other work ...

speech.stop_continuous_listening()
```

#### Whisper (offline, high accuracy)

```python
speech = SpeechModule(stt_backend="whisper")
# First call downloads the model (~140 MB for "base")
text = speech.listen()
```

---

### 4.7 Plugin System

**Files:** `wizardai/plugins.py`  
**Classes:** `PluginBase`, `PluginManager`

Extend WizardAI with custom skills by subclassing `PluginBase` and registering with `PluginManager`.

#### Creating a plugin

```python
from wizardai import PluginBase
from typing import Optional

class WeatherPlugin(PluginBase):
    # Required class attributes
    name        = "weather"
    description = "Returns weather data for any city."
    version     = "1.0.0"
    author      = "Your Name"
    triggers    = ["weather in *", "what's the weather in *"]

    def setup(self):
        """Called once after __init__. Use to initialise resources."""
        self.api_key = self.config.get("api_key", "")

    def teardown(self):
        """Called when the plugin is unregistered. Use to clean up."""
        pass

    def on_message(self, text: str, context: dict) -> Optional[str]:
        """
        Process user input. Return a string response, or None to pass through.
        Only called when plugin is enabled.
        """
        city = text.split("in", 1)[-1].strip()
        return f"The weather in {city} is sunny, 25°C."

    def on_start(self):
        """Called when the WizardAI session starts."""
        self.logger.info(f"WeatherPlugin ready.")

    def on_stop(self):
        """Called when the WizardAI session ends."""
        pass
```

#### Registering plugins

```python
from wizardai import PluginManager

manager = PluginManager()

# Register by class
plugin = manager.register(WeatherPlugin, config={"api_key": "abc123"})

# Register with name override
plugin = manager.register(WeatherPlugin, name_override="my_weather")

# Load from a single Python file
plugin = manager.load_from_file("./plugins/joke_plugin.py")

# Load all plugins from a directory
plugins = manager.load_from_directory("./plugins/")
```

#### Dispatching

```python
# Returns first non-None response (stops at first match)
response = manager.dispatch("weather in Paris", context={})
print(response)      # → "The weather in Paris is sunny, 25°C."

# Returns responses from ALL matching plugins
results = manager.dispatch_all("hello", context={})
# → [("plugin_name", "response"), ...]
```

#### Management

```python
# Introspection
all_plugins     = manager.list_plugins()
enabled_plugins = manager.list_plugins(enabled_only=True)
plugin          = manager.get("weather")

# Enable / disable
plugin.enable()
plugin.disable()
print(plugin.is_enabled)

# Unregister
manager.unregister("weather")

# Session lifecycle
manager.start_all()   # calls on_start() on all enabled plugins
manager.stop_all()    # calls on_stop() on all plugins

print(len(manager))   # → number of registered plugins
```

#### Using plugins with WizardAI core

```python
wiz = WizardAI(openai_api_key="sk-...")
wiz.add_plugin(WeatherPlugin, config={"api_key": "..."})

# Now wiz.chat() will try plugins first
reply = wiz.chat("weather in Tokyo")   # → "The weather in Tokyo is..."
```

---

### 4.8 Exceptions

**File:** `wizardai/exceptions.py`

All WizardAI exceptions inherit from `WizardAIError`.

#### Exception hierarchy

```
WizardAIError
├── APIError
│   ├── RateLimitError
│   └── AuthenticationError
├── VisionError
│   └── CameraNotFoundError
├── SpeechError
│   └── MicrophoneNotFoundError
├── ConversationError
├── PluginError
└── ConfigurationError
```

#### Usage

```python
from wizardai import WizardAIError, APIError, VisionError, SpeechError
from wizardai.exceptions import (
    RateLimitError,
    AuthenticationError,
    CameraNotFoundError,
    MicrophoneNotFoundError,
    ConversationError,
    PluginError,
    ConfigurationError,
)

try:
    reply = wiz.ask("Hello!")
except AuthenticationError as e:
    print(f"Bad API key for backend: {e.backend}")
    print(f"HTTP code: {e.code}")

except RateLimitError as e:
    print(f"Rate limit hit. Retry after: {e.retry_after}s")

except APIError as e:
    print(f"API error {e.code}: {e.message}")
    print(f"Backend: {e.backend}")

except WizardAIError as e:
    print(f"General error: {e.message}")

# Vision errors
try:
    cam = VisionModule(device_id=99)
    cam.open()
except CameraNotFoundError as e:
    print(f"Camera {e.device_id} not found!")

# Speech errors
try:
    text = speech.listen()
except MicrophoneNotFoundError:
    print("No microphone detected!")

# Plugin errors
try:
    manager.register(BadPlugin)
except PluginError as e:
    print(f"Plugin '{e.plugin_name}' failed: {e.message}")
```

#### Exception attributes

| Exception | Extra Attributes |
|-----------|-----------------|
| `WizardAIError` | `message`, `code` |
| `APIError` | `message`, `code`, `backend` |
| `RateLimitError` | `retry_after` |
| `AuthenticationError` | `backend` |
| `CameraNotFoundError` | `device_id` |
| `PluginError` | `plugin_name` |

---

### 4.9 Utilities

**File:** `wizardai/utils.py`  
**Classes:** `Logger`, `FileHelper`, `DataSerializer`, `RateLimiter`

#### Logger

```python
from wizardai.utils import Logger

log = Logger(
    name="my_app",
    level="DEBUG",          # "DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL"
    log_file="app.log",     # optional file output
    coloured=True,          # ANSI colour codes in terminal
)

log.debug("Detailed trace info")
log.info("Session started")
log.warning("Low memory")
log.error("Connection failed")
log.critical("Fatal error!")
```

#### FileHelper

```python
from wizardai.utils import FileHelper

files = FileHelper(base_dir="./my_data")

path = files.ensure_dir("images")          # creates dir, returns Path
path = files.save_text("note.txt", "hello")
text = files.read_text("note.txt")
path = files.save_bytes("photo.jpg", data)
data = files.read_bytes("photo.jpg")
files.delete("old_file.txt")
listing = files.list_files(pattern="*.json")
```

#### DataSerializer

```python
from wizardai.utils import DataSerializer

s = DataSerializer()

# JSON
s.save({"key": "val"}, "data.json")
data = s.load("data.json")

# Pickle
s.save(any_python_object, "data.pkl")
data = s.load("data.pkl")

# Gzip JSON
s.save(large_dict, "data.json.gz")
data = s.load("data.json.gz")
```

#### RateLimiter

```python
from wizardai.utils import RateLimiter

limiter = RateLimiter(max_calls=10, period=60.0)  # 10 calls per minute

with limiter:
    # code that should be rate-limited
    make_api_call()
```

---

## 5. Configuration Reference

### WizardAI constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ai_backend` | `str\|AIBackend` | `"openai"` | AI backend to use |
| `openai_api_key` | `str` | `None` | OpenAI API key |
| `anthropic_api_key` | `str` | `None` | Anthropic API key |
| `huggingface_api_key` | `str` | `None` | Hugging Face API key |
| `custom_endpoint` | `str` | `None` | Custom REST endpoint URL |
| `default_model` | `str` | Backend default | Model identifier |
| `max_tokens` | `int` | `1024` | Max tokens per LLM response |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `enable_vision` | `bool` | `False` | Open webcam on `start()` |
| `camera_device` | `int` | `0` | OpenCV camera index |
| `camera_width` | `int` | `640` | Capture width (px) |
| `camera_height` | `int` | `480` | Capture height (px) |
| `enable_speech` | `bool` | `False` | Init STT/TTS on `start()` |
| `stt_backend` | `str` | `"google"` | STT engine |
| `tts_backend` | `str` | `"pyttsx3"` | TTS engine |
| `language` | `str` | `"en-US"` | BCP-47 language code |
| `agent_name` | `str` | `"WizardBot"` | Agent display name |
| `fallback_response` | `str` | `"I'm not sure..."` | Fallback when no pattern matches |
| `max_history` | `int` | `50` | Conversation history window |
| `memory_path` | `str` | `None` | Path for memory persistence |
| `system_prompt` | `str` | `None` | Default LLM system prompt |
| `log_level` | `str` | `"INFO"` | `DEBUG\|INFO\|WARNING\|ERROR` |
| `log_file` | `str` | `None` | Optional log file path |
| `data_dir` | `str` | `"./wizardai_data"` | Working data directory |

### AIClient constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str\|AIBackend` | `"openai"` | Backend identifier |
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `model` | `str` | Backend default | Model identifier |
| `endpoint` | `str` | `None` | Custom endpoint URL |
| `max_retries` | `int` | `3` | Retry attempts on errors |
| `retry_delay` | `float` | `1.0` | Initial retry delay (doubles each attempt) |
| `timeout` | `float` | `30.0` | HTTP timeout in seconds |
| `rate_limit_calls` | `int` | `60` | Max calls per rate window |
| `rate_limit_period` | `float` | `60.0` | Rate-limit window in seconds |

---

## 6. Environment Variables

WizardAI automatically reads these environment variables if the corresponding API key is not passed explicitly:

| Variable | Used By | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `AIClient` (openai backend) | OpenAI API key |
| `ANTHROPIC_API_KEY` | `AIClient` (anthropic backend) | Anthropic API key |
| `HUGGINGFACE_API_KEY` | `AIClient` (huggingface backend) | Hugging Face token |
| `WIZARDAI_CUSTOM_API_KEY` | `AIClient` (custom backend) | Custom endpoint API key |
| `ELEVENLABS_API_KEY` | `SpeechModule` | ElevenLabs TTS API key |

### Setting environment variables

```bash
# .env file (use python-dotenv to load)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=...
```

```python
# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Now keys are read automatically
wiz = WizardAI()
```

---

## 7. AI Backend Guide

### OpenAI

```python
wiz = WizardAI(
    ai_backend="openai",
    openai_api_key="sk-...",
    default_model="gpt-4o",         # gpt-4o, gpt-4o-mini, gpt-3.5-turbo
)
```

### Anthropic (Claude)

```python
wiz = WizardAI(
    ai_backend="anthropic",
    anthropic_api_key="sk-ant-...",
    default_model="claude-3-5-sonnet-20241022",
)
```

### Hugging Face

```python
wiz = WizardAI(
    ai_backend="huggingface",
    huggingface_api_key="hf_...",
    default_model="mistralai/Mistral-7B-Instruct-v0.2",
)
```

### Ollama (local, no API key)

```python
wiz = WizardAI(
    ai_backend="custom",
    custom_endpoint="http://localhost:11434/v1/chat/completions",
    default_model="llama3",
)
```

### LM Studio (local, no API key)

```python
wiz = WizardAI(
    ai_backend="custom",
    custom_endpoint="http://localhost:1234/v1/chat/completions",
    default_model="local-model",
)
```

### Groq (fast inference)

```python
wiz = WizardAI(
    ai_backend="custom",
    custom_endpoint="https://api.groq.com/openai/v1/chat/completions",
    openai_api_key="gsk_...",
    default_model="llama-3.1-70b-versatile",
)
```

---

## 8. Error Handling

### Recommended pattern

```python
import wizardai
from wizardai.exceptions import (
    AuthenticationError, RateLimitError, APIError,
    CameraNotFoundError, MicrophoneNotFoundError,
    WizardAIError,
)
import time

wiz = wizardai.WizardAI(openai_api_key="sk-...")
wiz.start()

try:
    reply = wiz.ask("Hello!")
    print(reply)

except AuthenticationError:
    print("Invalid API key. Check your credentials.")

except RateLimitError as e:
    print(f"Rate limited. Waiting {e.retry_after or 60}s...")
    time.sleep(e.retry_after or 60)

except APIError as e:
    print(f"API error {e.code}: {e.message}")

except WizardAIError as e:
    print(f"WizardAI error: {e}")

finally:
    wiz.stop()
```

### Handling retries manually

```python
import time

def ask_with_retry(wiz, prompt, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return wiz.ask(prompt)
        except RateLimitError as e:
            wait = e.retry_after or (2 ** attempt)
            print(f"Rate limited, waiting {wait}s (attempt {attempt+1})")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")
```

---

## 9. Advanced Usage

### Multimodal (image + text)

```python
with WizardAI(openai_api_key="sk-...", enable_vision=True) as wiz:
    frame = wiz.capture()
    b64   = wiz.vision.encode_to_base64(frame)
    reply = wiz.ask(
        "What objects can you see in this image?",
        image_b64=b64,
        model="gpt-4o",           # use a vision-capable model
    )
    print(reply)
```

### Voice assistant loop

```python
with WizardAI(
    openai_api_key="sk-...",
    enable_speech=True,
    stt_backend="whisper",
    tts_backend="pyttsx3",
) as wiz:
    wiz.say("Hello! I'm WizardAI. How can I help?")
    while True:
        reply = wiz.voice_chat(timeout=8)
        if reply and "goodbye" in reply.lower():
            break
```

### Custom streaming response

```python
client = AIClient(backend="openai", api_key="sk-...")

print("Bot: ", end="", flush=True)
for chunk in client.chat_stream([{"role": "user", "content": "Tell me a story"}]):
    print(chunk, end="", flush=True)
print()
```

### Persistent session with memory

```python
wiz = WizardAI(
    openai_api_key="sk-...",
    memory_path="./session.json",    # auto-saves after every message
    max_history=100,
)
wiz.start()

# Session persists across restarts — memory loaded from disk automatically
wiz.remember("user_name", "Alice")
wiz.remember("preferences", {"lang": "en", "theme": "dark"})

# On next run, memory is restored automatically
name = wiz.recall("user_name")     # → "Alice" even after restart
```

### Multiple AI backends in one app

```python
fast_client  = AIClient(backend="openai",    model="gpt-4o-mini")   # quick responses
smart_client = AIClient(backend="anthropic", model="claude-3-5-sonnet-20241022")  # complex tasks

# Route by complexity
def smart_route(question: str) -> str:
    if len(question.split()) < 10:
        response = fast_client.complete(question)
    else:
        response = smart_client.complete(question)
    return response.text
```

### Plugin with API integration

```python
import requests
from wizardai import PluginBase

class CryptoPlugin(PluginBase):
    name        = "crypto"
    description = "Live cryptocurrency prices."
    triggers    = ["price of *", "* price", "how much is *"]

    def on_message(self, text: str, context: dict):
        coin = text.split()[-1].upper()
        try:
            r = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price"
                f"?ids={coin.lower()}&vs_currencies=usd",
                timeout=5
            )
            data = r.json()
            price = data.get(coin.lower(), {}).get("usd", "unknown")
            return f"{coin} is currently ${price} USD."
        except Exception:
            return f"Could not fetch price for {coin}."
```

### Loading plugins from a directory

Structure your `plugins/` folder as individual Python files:

```
plugins/
├── weather_plugin.py
├── crypto_plugin.py
└── joke_plugin.py
```

Each file must contain exactly one `PluginBase` subclass:

```python
# plugins/joke_plugin.py
from wizardai import PluginBase
import random

class JokePlugin(PluginBase):
    name = "jokes"
    def on_message(self, text, ctx):
        if "joke" in text.lower():
            return random.choice([
                "Why do programmers prefer dark mode? Light attracts bugs!",
                "A SQL query walks into a bar, sees two tables and asks: 'Can I join you?'"
            ])
        return None
```

```python
wiz = WizardAI(openai_api_key="sk-...")
wiz.load_plugins_from_dir("./plugins/")
wiz.start()
```

---

## 10. Publishing to PyPI

### Step 1: Configure `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "wizardai-sdk"
version = "1.0.1"
authors = [{ name = "WizardAI Contributors" }]
description = "A powerful AI SDK with vision, speech, and multi-backend support"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.9"
dependencies = ["requests>=2.28.0", "openai>=1.0.0", "anthropic>=0.20.0"]

[project.optional-dependencies]
vision  = ["opencv-python>=4.7.0", "numpy>=1.24.0"]
speech  = ["SpeechRecognition>=3.10.0", "pyttsx3>=2.90", "gtts>=2.3.0", "pygame>=2.4.0"]
whisper = ["openai-whisper>=20230918"]
full    = ["opencv-python>=4.7.0", "numpy>=1.24.0", "SpeechRecognition>=3.10.0",
           "pyttsx3>=2.90", "gtts>=2.3.0", "pygame>=2.4.0", "openai-whisper>=20230918"]
dev     = ["pytest>=7.0", "black>=23.0", "ruff>=0.1.0", "build>=0.10", "twine>=4.0"]

[tool.setuptools.packages.find]
where   = ["."]
include = ["wizardai*"]
```

### Step 2: Configure `setup.py`

```python
from pathlib import Path
from setuptools import setup, find_packages

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Safe version extraction (no exec())
version = "1.0.1"
init_path = Path(__file__).parent / "wizardai" / "__init__.py"
if init_path.exists():
    for line in init_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="wizardai-sdk",
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
)
```

### Step 3: GitHub Actions workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write    # required for trusted publishing

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.x"

      - name: Install build tools
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

### Step 4: Configure PyPI Trusted Publisher

Go to `https://pypi.org/manage/account/publishing/` and fill in:

| Field | Value |
|-------|-------|
| PyPI Project Name | `wizardai-sdk` |
| Owner | `VIkasPatel22` |
| Repository name | `wizardai-sdk` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

### Step 5: Create a release

Go to `https://github.com/VIkasPatel22/wizardai-sdk/releases/new`, create tag `v1.0.1`, and publish. The workflow runs automatically.

### Step 6: Install from PyPI

```bash
pip install wizardai-sdk
pip install "wizardai-sdk[full]"
```

---

## 11. Project Structure

```
wizardai-sdk/
│
├── wizardai/                    ← Main package
│   ├── __init__.py              ← Public API surface, __version__, __all__
│   ├── core.py                  ← WizardAI orchestrator
│   ├── ai_client.py             ← AIClient, AIBackend, AIResponse
│   ├── conversation.py          ← ConversationAgent, Pattern
│   ├── memory.py                ← MemoryManager, Message
│   ├── vision.py                ← VisionModule
│   ├── speech.py                ← SpeechModule
│   ├── plugins.py               ← PluginBase, PluginManager
│   ├── utils.py                 ← Logger, FileHelper, DataSerializer
│   └── exceptions.py            ← Exception hierarchy
│
├── examples/
│   └── full_demo.py             ← Complete usage demo
│
├── plugins/
│   └── sample_plugin.py         ← Example plugin
│
├── tests/
│   ├── __init__.py
│   ├── test_ai_client.py
│   ├── test_conversation.py
│   ├── test_memory.py
│   ├── test_plugins.py
│   └── test_core.py
│
├── .github/
│   └── workflows/
│       └── publish.yml          ← PyPI publish workflow
│
├── LICENSE
├── README.md
├── DOCUMENTATION.md             ← This file
├── pyproject.toml
├── setup.py
├── requirements.txt             ← Core dependencies
└── requirements-full.txt        ← All optional dependencies
```

---

## 12. Contributing

```bash
# 1. Fork the repo and clone it
git clone https://github.com/VIkasPatel22/wizardai-sdk.git
cd wizardai-sdk

# 2. Install in dev mode with all extras
pip install -e ".[dev,full]"

# 3. Create a feature branch
git checkout -b feature/my-awesome-feature

# 4. Make your changes, then run tests
pytest tests/ -v

# 5. Format and lint
black .
isort .
ruff check .

# 6. Open a Pull Request
```

### Running tests

```bash
pytest tests/                            # run all tests
pytest tests/test_memory.py -v           # single file
pytest tests/ --cov=wizardai             # with coverage report
```

---

## 13. License

MIT © WizardAI Contributors

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

See [LICENSE](LICENSE) for the full text.

---

*Documentation version: 1.0.0 | Last updated: 2026*
