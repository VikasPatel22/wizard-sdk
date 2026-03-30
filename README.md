# 🧙 WizardEnv — Complete Documentation

[![Version](https://img.shields.io/badge/version-1.1.0-blue?style=flat-square)](https://pypi.org/project/wizardai-sdk/)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-orange?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/badge/pip_install-wizardai--sdk-purple?style=flat-square)](https://pypi.org/project/wizardai-sdk/)

> **A powerful, all-in-one Python SDK for AI integration** — conversational AI, computer vision, speech I/O, memory management, and a plugin system in a single file. Powered by the **Sagittarius Labs API** at [https://sagittarius-labs.pages.dev/](https://sagittarius-labs.pages.dev/).

---

## Table of Contents

1. [Installation](#1-installation)
2. [Getting Your API Key](#2-getting-your-api-key)
3. [Quick Start](#3-quick-start)
4. [Architecture Overview](#4-architecture-overview)
5. [WizardAI Core](#5-wizardai-core)
6. [AIClient](#6-aiclient)
7. [ConversationAgent](#7-conversationagent)
8. [MemoryManager](#8-memorymanager)
9. [VisionModule](#9-visionmodule)
10. [SpeechModule](#10-speechmodule)
11. [Plugin System](#11-plugin-system)
12. [Utilities](#12-utilities)
13. [Exceptions & Error Handling](#13-exceptions--error-handling)
14. [Environment Variables](#14-environment-variables)
15. [Configuration Reference](#15-configuration-reference)
16. [Advanced Usage](#16-advanced-usage)
17. [Publishing to PyPI](#17-publishing-to-pypi)
18. [Contributing](#18-contributing)
19. [License](#19-license)

---

## 1. Installation

### Minimal (core AI only)

```bash
pip install wizardai-sdk
```

The only required third-party dependency is `requests`.

### With optional features

```bash
# Computer vision (OpenCV)
pip install "wizardai-sdk[vision]"

# Speech recognition + TTS
pip install "wizardai-sdk[speech]"

# High-accuracy offline speech (OpenAI Whisper)
pip install "wizardai-sdk[whisper]"

# Everything at once
pip install "wizardai-sdk[full]"

# Developer tools
pip install "wizardai-sdk[dev]"
```

### From source

```bash
git clone https://github.com/YourUsername/wizardai-sdk.git
cd wizardai-sdk
pip install -e ".[full]"
```

### System dependencies for speech (Linux/macOS)

```bash
# Linux
sudo apt-get install portaudio19-dev python3-dev

# macOS
brew install portaudio
```

---

## 2. Getting Your API Key

WizardAI is powered by the **Sagittarius Labs API**.

1. Visit [https://sagittarius-labs.pages.dev/](https://sagittarius-labs.pages.dev/)
2. Create a free account and generate your API key
3. Use the key as shown below

> If your API key is wrong or missing, WizardAI will print a message directing you back to [https://sagittarius-labs.pages.dev/](https://sagittarius-labs.pages.dev/) to get or verify it.

### Setting the key via environment variable (recommended)

```bash
export WIZARDAI_API_KEY="your_key_here"
```

Or in a `.env` file with `python-dotenv`:

```
WIZARDAI_API_KEY=your_key_here
```

```python
from dotenv import load_dotenv
load_dotenv()

import wizardai
wiz = wizardai.WizardAI()   # key is read from env automatically
```

---

## 3. Quick Start

### Minimal example

```python
import wizardai

wiz = wizardai.WizardAI(api_key="YOUR_KEY")
wiz.start()

# Rule-based response (no API call)
wiz.agent.add_pattern("hello", "Hello from WizardAI!")
print(wiz.chat("hello"))              # → "Hello from WizardAI!"

# Direct LLM call
print(wiz.ask("What is the speed of light?"))

wiz.stop()
```

### Recommended: context manager

```python
with wizardai.WizardAI(api_key="YOUR_KEY") as wiz:
    print(wiz.ask("Tell me a joke."))
```

### Full-featured example

```python
wiz = wizardai.WizardAI(
    api_key="YOUR_KEY",
    enable_vision=True,
    enable_speech=True,
    stt_backend="google",
    tts_backend="pyttsx3",
    memory_path="session.json",
    system_prompt="You are a helpful assistant.",
)
wiz.start()

# Multimodal: capture + describe
frame   = wiz.capture()
b64     = wiz.vision.encode_to_base64(frame)
caption = wiz.ask("Describe this image.", image_b64=b64)
wiz.say(caption)

# Streaming response
for chunk in wiz.ai.chat_stream([{"role": "user", "content": "Write a poem"}]):
    print(chunk, end="", flush=True)

# Long-term memory
wiz.remember("user_name", "Alice")
print(wiz.recall("user_name"))        # → "Alice"

wiz.stop()
```

### Interactive terminal REPL

```python
wiz = wizardai.WizardAI(api_key="YOUR_KEY")
wiz.start()
wiz.run_repl()                        # keyboard input
wiz.run_repl(voice_mode=True)         # microphone input
```

---

## 4. Architecture Overview

```
wizardai.py  (single file)
│
├── WizardAI               ← Top-level orchestrator
│     ├── AIClient         ← Sagittarius Labs LLM calls
│     ├── ConversationAgent← AIML-style pattern matching
│     ├── MemoryManager    ← Short + long-term memory
│     ├── VisionModule     ← Camera / OpenCV (optional)
│     ├── SpeechModule     ← STT + TTS (optional)
│     ├── PluginManager    ← Extensible skill plugins
│     ├── FileHelper       ← File I/O utilities
│     └── DataSerializer   ← JSON/Pickle persistence
```

**Chat pipeline priority** (inside `wiz.chat()`):

```
User input
    │
    ├─ 1. Plugin dispatch   → first plugin that returns non-None
    ├─ 2. Pattern matching  → ConversationAgent rules
    └─ 3. LLM fallback      → Sagittarius Labs API
```

---

## 5. WizardAI Core

### Constructor

```python
WizardAI(
    api_key=None,              # str — or set WIZARDAI_API_KEY env var
                               #       Get one at https://sagittarius-labs.pages.dev/
    model=None,                # str — override default model
    max_tokens=1024,           # int
    temperature=0.7,           # float

    enable_vision=False,       # bool — open webcam on start()
    camera_device=0,           # int  — OpenCV device index
    camera_width=640,          # int
    camera_height=480,         # int

    enable_speech=False,       # bool
    stt_backend="google",      # "google" | "sphinx" | "whisper"
    tts_backend="pyttsx3",     # "pyttsx3" | "gtts" | "elevenlabs"
    language="en-US",          # BCP-47 language code

    agent_name="WizardBot",
    fallback_response="I'm not sure how to respond to that.",

    max_history=50,            # conversation window size
    memory_path=None,          # auto-save path (JSON)

    system_prompt=None,        # default LLM system prompt
    log_level="INFO",          # "DEBUG"|"INFO"|"WARNING"|"ERROR"
    log_file=None,             # optional log file path
    data_dir="./wizardai_data",# working directory
)
```

### Session lifecycle

```python
wiz.start()   # opens camera, init speech, calls plugin.on_start()
wiz.stop()    # stops all modules, saves memory, calls plugin.on_stop()

with WizardAI(api_key="...") as wiz:
    ...       # auto start/stop
```

### Chat methods

| Method | Description |
|--------|-------------|
| `wiz.chat(text)` | Full pipeline: plugins → patterns → LLM |
| `wiz.ask(prompt, **kwargs)` | Direct LLM call |
| `wiz.ask_raw(prompt)` | Returns full `AIResponse` object |

```python
# Basic chat
reply = wiz.chat("hello world")

# Ask with options
reply = wiz.ask(
    "Summarise this.",
    max_tokens=300,
    temperature=0.3,
    system_prompt="You are a concise summariser.",
    include_history=False,     # don't send conversation history
    image_b64=b64_string,      # for multimodal requests
)
```

### Speech shortcuts

```python
text  = wiz.listen(timeout=5.0)
wiz.say("Hello!")
reply = wiz.voice_chat(timeout=5.0)   # listen + chat + speak
```

### Vision shortcuts

```python
frame = wiz.capture()
path  = wiz.snapshot("photo.jpg")
```

### Memory shortcuts

```python
wiz.remember("key", value)
value   = wiz.recall("key", default=None)
history = wiz.get_history(n=10)
```

### Configuration helpers

```python
wiz.set_system_prompt("You are a pirate.")
wiz.set_model("sagittarius/deep-vl-r1-128b")
wiz.set_api_key("new_key")
```

---

## 6. AIClient

All AI calls go through the **Sagittarius Labs API** at
`https://sagittarius-labs.pages.dev/api/chat`.

If authentication fails, a helpful error message directs you to
[https://sagittarius-labs.pages.dev/](https://sagittarius-labs.pages.dev/) to get or verify your key.

### Constructor

```python
from wizardai import AIClient

client = AIClient(
    api_key="YOUR_KEY",       # or set WIZARDAI_API_KEY
    model="sagittarius/deep-vl-r1-128b",
    max_retries=3,
    retry_delay=1.0,
    timeout=60.0,
    rate_limit_calls=60,
    rate_limit_period=60.0,
)
```

### AIResponse

```python
response.text        # str  — generated text
response.model       # str  — model used
response.usage       # dict — token stats
response.raw         # dict — raw API response
response.latency_ms  # float — round-trip time in ms
str(response)        # same as response.text
```

### Methods

```python
# Non-streaming
response = client.complete("Write a haiku.")
print(response.text)

# Multi-turn chat
messages = [
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user",      "content": "What is my name?"},
]
response = client.chat(messages, system_prompt="You are helpful.")
print(response.text)   # "Your name is Alice."

# Streaming
for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)

# Single-turn streaming
for chunk in client.complete_stream("Tell me a story."):
    print(chunk, end="", flush=True)

# Runtime changes
client.set_api_key("new_key")
client.set_model("sagittarius/deep-vl-r1-128b")
```

---

## 7. ConversationAgent

AIML-style rule engine with wildcards, priorities, context, and callable templates.

### Wildcard syntax

| Wildcard | Matches | Example |
|----------|---------|---------|
| `*` | One or more words | `"tell me about *"` |
| `?` | Exactly one word | `"is ? available"` |
| `{name}` | Named capture group | `"weather in {city}"` |

### Adding patterns

```python
agent = wiz.agent   # or ConversationAgent()

# Simple string
agent.add_pattern("hello", "Hello there!")

# Wildcard substitution
agent.add_pattern("my name is *", "Nice to meet you, {wildcard}!")

# Named capture group
agent.add_pattern("weather in {city}", "Checking weather for {city}…")

# Callable (dynamic)
import time
agent.add_pattern("what time is it", lambda: f"It's {time.strftime('%H:%M')}.")

# Random choice from list
agent.add_pattern("how are you", [
    "Doing great!",
    "Running at 100%!",
    "Excellent — ready to help!",
])

# Priority (higher = tried first)
agent.add_pattern("hello world", "Special greeting!", priority=10)

# Context-aware
agent.add_pattern("yes", "Proceeding!", context="confirm")
agent.add_pattern("no",  "Cancelled.",  context="confirm")
agent.set_context("confirm")

# Pattern object (full control)
from wizardai import Pattern
agent.add_pattern_obj(Pattern(
    pattern="translate * to {lang}",
    template="Translating…",
    priority=5,
    tags=["language"],
))
```

### Using patterns

```python
print(agent.respond("hello"))             # → "Hello there!"
print(agent.respond("my name is Bob"))    # → "Nice to meet you, Bob!"

# Bulk load from dict
agent.load_patterns_from_dict({
    "ping": "pong",
    "what version": "WizardAI v1.1.0",
})

# Load from JSON file
agent.load_patterns_from_file("patterns.json")

# Manage
agent.remove_pattern("hello")
agent.clear_patterns()
agent.reset()                             # wipe history + context

# Inline plugins (!name args)
agent.register_plugin("joke", lambda args: "Why so serious?")
# User types: "!joke" → calls the lambda
```

### Pre/post processors

```python
agent.add_preprocessor(lambda t: t.strip().lower())
agent.add_postprocessor(lambda t: t + " 😊")
```

---

## 8. MemoryManager

Short-term sliding-window history and long-term key-value storage.

```python
from wizardai import MemoryManager

mem = MemoryManager(
    max_history=50,
    persist_path="session.json",   # auto-save on every write
)

# Short-term history
mem.add_message("user",      "Hello!")
mem.add_message("assistant", "Hi there!")

msgs = mem.get_history()               # all Message objects
msgs = mem.get_history(n=5)            # last 5
msgs = mem.get_history(role_filter="user")

api_msgs = mem.get_messages_for_api()  # [{"role": ..., "content": ...}]
dicts    = mem.get_history_as_dicts()

last_msg = mem.last_message()
last_usr = mem.last_message(role="user")
results  = mem.search_history("hello", top_k=3)  # [(Message, score)]

mem.clear_history()

# Long-term memory
mem.remember("user_name", "Alice")
name = mem.recall("user_name")         # → "Alice"
mem.forget("user_name")
keys = mem.list_memories()

# Ephemeral context (not saved to disk)
mem.set_context("topic", "weather")
topic = mem.get_context("topic")
mem.clear_context()

# Persistence
mem.save()                   # save to persist_path
mem.save("backup.json")
mem.load()
mem.load("backup.json")
```

---

## 9. VisionModule

Real-time camera access and image processing using OpenCV.

**Requires:** `pip install opencv-python`

```python
from wizardai import VisionModule

cam = VisionModule(device_id=0, width=1280, height=720, fps=30)
cam.open()

frame = cam.capture_frame()             # → numpy.ndarray (BGR)
cam.save_frame(frame, "photo.jpg")
b64   = cam.encode_to_base64(frame)     # for LLM image inputs

# Face detection
faces = cam.detect_faces(frame)         # [{"x":…, "y":…, "w":…, "h":…}]
annotated, faces = cam.annotate_faces(frame)  # draws bounding boxes

# Image processing
gray     = cam.to_grayscale(frame)
rgb      = cam.to_rgb(frame)
flipped  = cam.flip(frame)              # horizontal flip by default
resized  = cam.resize_frame(frame, 320, 240)

# Drawing
cam.draw_rectangle(frame, x=10, y=10, w=100, h=50, colour=(0,255,0))
cam.draw_text(frame, "Hello!", x=10, y=80)

# Load from disk
frame2 = cam.load_image("photo.jpg")

# Streaming
def on_frame(frame):
    faces = cam.detect_faces(frame)
    print(f"{len(faces)} face(s)")

cam.start_stream(callback=on_frame, show_preview=True)
import time; time.sleep(10)
cam.stop_stream()

cam.close()

# Context manager
with VisionModule() as cam:
    frame = cam.capture_frame()
```

---

## 10. SpeechModule

Speech recognition (STT) and text-to-speech (TTS).

**Requires:** `pip install SpeechRecognition pyttsx3`

### STT backends

| Backend | Type | Package |
|---------|------|---------|
| `google` | Online | `SpeechRecognition` |
| `sphinx` | Offline | `pocketsphinx` |
| `whisper` | Offline (GPU recommended) | `openai-whisper` |

### TTS backends

| Backend | Type | Package |
|---------|------|---------|
| `pyttsx3` | Offline | `pyttsx3` |
| `gtts` | Online | `gtts`, `pygame` |
| `elevenlabs` | Online | (requests, set `ELEVENLABS_API_KEY`) |

```python
from wizardai import SpeechModule

speech = SpeechModule(stt_backend="google", tts_backend="pyttsx3")

# STT
text = speech.listen(timeout=5.0, phrase_time_limit=15.0)
text = speech.transcribe_file("audio.wav")

# TTS
speech.say("Hello, world!")
speech.say("Processing…", blocking=False)
speech.synthesise_to_file("Hello!", "output.mp3")

# Voices (pyttsx3)
voices = speech.list_voices()
speech.set_tts_voice(voices[0]["id"])
speech.set_tts_rate(180)
speech.set_tts_volume(0.8)

# Microphones
mics = speech.list_microphones()
text = speech.listen(device_index=mics[0]["index"])

# Continuous listening
def on_speech(text):
    print("Heard:", text)

speech.start_continuous_listening(callback=on_speech)
import time; time.sleep(30)
speech.stop_continuous_listening()
```

---

## 11. Plugin System

Extend WizardAI with custom skills by subclassing `PluginBase`.

### Creating a plugin

```python
from wizardai import PluginBase
from typing import Optional

class WeatherPlugin(PluginBase):
    name        = "weather"
    description = "Returns weather for a city."
    version     = "1.1.0"
    author      = "You"
    triggers    = ["weather in *"]

    def setup(self):
        """Initialise resources once."""
        self.api_key = self.config.get("api_key", "")

    def teardown(self):
        """Clean up when unregistered."""
        pass

    def on_message(self, text: str, context: dict) -> Optional[str]:
        """Return a response string, or None to pass through."""
        city = text.split("in", 1)[-1].strip()
        return f"The weather in {city} is sunny, 25°C."

    def on_start(self): self.logger.info("WeatherPlugin ready.")
    def on_stop(self):  pass
```

### Registering plugins

```python
from wizardai import PluginManager

manager = PluginManager()

# By class
manager.register(WeatherPlugin, config={"api_key": "..."})

# With name override
manager.register(WeatherPlugin, name_override="weather_v2")

# From a Python file
manager.load_from_file("plugins/joke_plugin.py")

# From a directory (all *.py files)
manager.load_from_directory("./plugins/")
```

### Dispatching

```python
# First matching plugin
response = manager.dispatch("weather in Paris")

# All matching plugins
results = manager.dispatch_all("hello")
# → [("plugin_name", "response"), ...]
```

### Management

```python
plugin = manager.get("weather")
plugin.enable()
plugin.disable()
print(plugin.is_enabled)

all_plugins     = manager.list_plugins()
enabled_plugins = manager.list_plugins(enabled_only=True)

manager.unregister("weather")

manager.start_all()
manager.stop_all()

print(len(manager))   # number of plugins
```

### Using with WizardAI

```python
with wizardai.WizardAI(api_key="...") as wiz:
    wiz.add_plugin(WeatherPlugin, config={"api_key": "..."})
    print(wiz.chat("weather in Tokyo"))   # → plugin handles it
```

### Loading from a directory

```
plugins/
├── weather_plugin.py   # contains class WeatherPlugin(PluginBase)
├── joke_plugin.py      # contains class JokePlugin(PluginBase)
└── crypto_plugin.py
```

```python
wiz.load_plugins_from_dir("./plugins/")
```

---

## 12. Utilities

### Logger

```python
from wizardai import Logger

log = Logger("my_app", level="DEBUG", log_file="app.log", coloured=True)
log.debug("Trace")
log.info("Started")
log.warning("Low memory")
log.error("Failed")
log.critical("Fatal!")
log.set_level("WARNING")
```

### FileHelper

```python
from wizardai import FileHelper

fh = FileHelper(base_dir="./data")
fh.ensure_dir("models/cache")

fh.write_text("note.txt", "Hello!")
text = fh.read_text("note.txt")
lines = fh.read_lines("note.txt")

fh.write_json("config.json", {"key": "val"})
cfg = fh.read_json("config.json")

fh.write_csv("data.csv", [{"a": 1, "b": 2}])
rows = fh.read_csv("data.csv")

fh.copy("src.txt", "dst.txt")
fh.delete("old.txt")
files = fh.list_files(pattern="*.json", recursive=True)
name  = fh.timestamp_filename("log", ".txt")  # "log_20260101_120000.txt"
```

### DataSerializer

```python
from wizardai import DataSerializer

ds = DataSerializer()
ds.save({"key": "val"}, "data.json")
ds.save(large_object,   "data.pkl.gz", compress=True)
data = ds.load("data.json")

json_str = ds.to_json_string({"a": 1})
data     = ds.from_json_string(json_str)

for record in ds.iter_jsonl("records.jsonl"):
    print(record)

ds.write_jsonl("out.jsonl", [{"a": 1}, {"b": 2}])
```

### RateLimiter

```python
from wizardai import RateLimiter

limiter = RateLimiter(max_calls=10, period=60.0)

# Block until a token is available
limiter.wait()
make_api_call()

# Non-blocking check
if limiter.is_allowed():
    make_api_call()

# Context manager
with limiter:
    make_api_call()
```

---

## 13. Exceptions & Error Handling

### Exception hierarchy

```
WizardAIError
├── APIError
│   ├── AuthenticationError   ← invalid/missing API key
│   └── RateLimitError        ← 429 too many requests
├── VisionError
│   └── CameraNotFoundError
├── SpeechError
│   └── MicrophoneNotFoundError
├── ConversationError
├── PluginError
└── ConfigurationError
```

### Recommended pattern

```python
import wizardai
from wizardai import (
    AuthenticationError,
    RateLimitError,
    APIError,
    WizardAIError,
    CameraNotFoundError,
    MicrophoneNotFoundError,
)
import time

try:
    with wizardai.WizardAI(api_key="YOUR_KEY") as wiz:
        reply = wiz.ask("Hello!")
        print(reply)

except AuthenticationError as e:
    # Printed message includes a link to https://sagittarius-labs.pages.dev/
    print(e)

except RateLimitError as e:
    wait = e.retry_after or 60
    print(f"Rate limited. Waiting {wait}s…")
    time.sleep(wait)

except APIError as e:
    print(f"API error {e.code}: {e.message}")

except WizardAIError as e:
    print(f"WizardAI error: {e}")
```

### Authentication errors

When your API key is wrong or missing, you will see:

```
AuthenticationError: Authentication failed — your API key is missing or invalid.
  Detail  : HTTP 401 from https://sagittarius-labs.pages.dev/api/chat. Verify your key.
  Fix     : Get or verify your key at https://sagittarius-labs.pages.dev/
  Env var : export WIZARDAI_API_KEY=<your_key>
```

### Retry helper

```python
def ask_with_retry(wiz, prompt, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return wiz.ask(prompt)
        except RateLimitError as e:
            wait = e.retry_after or (2 ** attempt)
            print(f"Rate limited, retrying in {wait}s…")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")
```

---

## 14. Environment Variables

| Variable | Description |
|----------|-------------|
| `WIZARDAI_API_KEY` | Your Sagittarius Labs API key |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS API key (optional) |

```bash
export WIZARDAI_API_KEY="your_key_here"
export ELEVENLABS_API_KEY="your_elevenlabs_key"
```

---

## 15. Configuration Reference

### WizardAI constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | Sagittarius Labs API key |
| `model` | `str` | `sagittarius/deep-vl-r1-128b` | Model identifier |
| `max_tokens` | `int` | `1024` | Max tokens per response |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `enable_vision` | `bool` | `False` | Open webcam on start |
| `camera_device` | `int` | `0` | OpenCV camera index |
| `camera_width` | `int` | `640` | Capture width |
| `camera_height` | `int` | `480` | Capture height |
| `enable_speech` | `bool` | `False` | Init STT/TTS on start |
| `stt_backend` | `str` | `"google"` | STT engine |
| `tts_backend` | `str` | `"pyttsx3"` | TTS engine |
| `language` | `str` | `"en-US"` | BCP-47 language code |
| `agent_name` | `str` | `"WizardBot"` | Agent display name |
| `fallback_response` | `str` | `"I'm not sure…"` | Pattern-match fallback |
| `max_history` | `int` | `50` | Conversation window |
| `memory_path` | `str` | `None` | Persistent memory path |
| `system_prompt` | `str` | `None` | Default system prompt |
| `log_level` | `str` | `"INFO"` | Log verbosity |
| `log_file` | `str` | `None` | Log file path |
| `data_dir` | `str` | `"./wizardai_data"` | Working directory |

### AIClient constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key (env fallback) |
| `model` | `str` | `sagittarius/deep-vl-r1-128b` | Default model |
| `max_retries` | `int` | `3` | Retry attempts |
| `retry_delay` | `float` | `1.0` | Initial retry delay |
| `timeout` | `float` | `60.0` | HTTP timeout |
| `rate_limit_calls` | `int` | `60` | Calls per window |
| `rate_limit_period` | `float` | `60.0` | Window in seconds |

---

## 16. Advanced Usage

### Multimodal (image + text)

```python
with wizardai.WizardAI(api_key="...", enable_vision=True) as wiz:
    frame = wiz.capture()
    b64   = wiz.vision.encode_to_base64(frame)
    reply = wiz.ask("What do you see in this image?", image_b64=b64)
    print(reply)
```

### Voice assistant loop

```python
with wizardai.WizardAI(
    api_key="...",
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

### Streaming response

```python
client = wizardai.AIClient(api_key="...")
print("Bot: ", end="", flush=True)
for chunk in client.chat_stream([{"role": "user", "content": "Write a poem"}]):
    print(chunk, end="", flush=True)
print()
```

### Persistent session with memory

```python
wiz = wizardai.WizardAI(
    api_key="...",
    memory_path="session.json",   # auto-saves after every message
)
wiz.start()

wiz.remember("user_name", "Alice")

# Memory survives restarts — loaded from disk automatically next time
name = wiz.recall("user_name")    # → "Alice" even after restarting
```

### Custom plugin with live API

```python
import requests as _req
from wizardai import PluginBase

class CryptoPlugin(PluginBase):
    name    = "crypto"
    version = "1.1.0"
    triggers = ["price of *", "* price"]

    def on_message(self, text, context):
        coin = text.split()[-1].upper()
        try:
            r = _req.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin.lower(), "vs_currencies": "usd"},
                timeout=5,
            )
            price = r.json().get(coin.lower(), {}).get("usd", "unknown")
            return f"{coin} is currently ${price} USD."
        except Exception:
            return f"Could not fetch price for {coin}."

with wizardai.WizardAI(api_key="...") as wiz:
    wiz.add_plugin(CryptoPlugin)
    print(wiz.chat("bitcoin price"))
```

### Multiple agents in one app

```python
from wizardai import ConversationAgent, MemoryManager

support_agent = ConversationAgent(name="Support", memory=MemoryManager())
sales_agent   = ConversationAgent(name="Sales",   memory=MemoryManager())

support_agent.add_pattern("refund *", "I'll process your refund for {wildcard}.")
sales_agent.add_pattern("buy *", "Great choice! Here's how to purchase {wildcard}.")

def route(text):
    if "refund" in text.lower():
        return support_agent.respond(text)
    return sales_agent.respond(text)
```

---

## 17. Publishing to PyPI

### `pyproject.toml`

```toml
[build-system]
requires      = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name            = "wizardai-sdk"
version         = "1.1.0"
description     = "All-in-one AI SDK powered by Sagittarius Labs"
readme          = "README.md"
license         = { text = "MIT" }
requires-python = ">=3.9"
dependencies    = ["requests>=2.28.0"]

[project.optional-dependencies]
vision  = ["opencv-python>=4.7.0", "numpy>=1.24.0"]
speech  = ["SpeechRecognition>=3.10.0", "pyttsx3>=2.90", "gtts>=2.3.0", "pygame>=2.4.0"]
whisper = ["openai-whisper>=20230918", "numpy>=1.24.0"]
full    = [
    "opencv-python>=4.7.0", "numpy>=1.24.0",
    "SpeechRecognition>=3.10.0", "pyttsx3>=2.90",
    "gtts>=2.3.0", "pygame>=2.4.0",
    "openai-whisper>=20230918",
]
dev = ["pytest>=7.0", "black>=23.0", "ruff>=0.1.0", "build>=0.10", "twine>=4.0"]
```

### `MANIFEST.in`

```
include wizardai.py
include README.md
include LICENSE
```

### GitHub Actions — `.github/workflows/publish.yml`

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
      id-token: write

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

### Configure PyPI Trusted Publisher

Go to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/):

| Field | Value |
|-------|-------|
| PyPI Project Name | `wizardai-sdk` |
| Owner | `YourGitHubUsername` |
| Repository name | `wizardai-sdk` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

### Create a release and publish

1. Push a tag: `git tag v1.1.0 && git push --tags`
2. Create a GitHub Release from the tag
3. The workflow publishes to PyPI automatically

```bash
pip install wizardai-sdk
pip install "wizardai-sdk[full]"
```

---

## 18. Contributing

```bash
# 1. Fork and clone
git clone https://github.com/YourUsername/wizardai-sdk.git
cd wizardai-sdk

# 2. Install dev extras
pip install -e ".[dev,full]"

# 3. Feature branch
git checkout -b feature/my-feature

# 4. Run tests
pytest tests/ -v

# 5. Format
black .
ruff check .

# 6. Open a Pull Request
```

---

## 19. License

MIT © WizardAI Contributors

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND.
```

---

*WizardEnv v1.1.0 — Powered by [Sagittarius Labs](https://sagittarius-labs.pages.dev/)*
