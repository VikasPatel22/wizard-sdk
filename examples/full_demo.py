"""
WizardAI SDK — Full Demo
========================
A single runnable script that walks through every module in the WizardAI SDK.

Usage
-----
    # Minimal (pattern-matching only, no API key required)
    python full_demo.py

    # With OpenAI
    OPENAI_API_KEY=sk-... python full_demo.py

    # With Anthropic
    ANTHROPIC_API_KEY=sk-ant-... python full_demo.py --backend anthropic

    # With vision + speech (requires webcam & microphone)
    python full_demo.py --vision --speech

Requirements
------------
    pip install "wizardai[all]"
    # or for specific features:
    pip install "wizardai[openai,vision,speech]"
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------

import wizardai
from wizardai import (
    WizardAI,
    AIClient,
    AIBackend,
    ConversationAgent,
    Pattern,
    MemoryManager,
    VisionModule,
    SpeechModule,
    PluginBase,
    PluginManager,
    Logger,
)
from wizardai.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    VisionError,
    SpeechError,
    PluginError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "\n" + "=" * 60
logger = Logger("Demo", level="INFO")


def section(title: str):
    print(f"{SEPARATOR}\n  {title}\n{'=' * 60}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def info(msg: str):
    print(f"  ℹ  {msg}")


# ---------------------------------------------------------------------------
# 1. MEMORY MANAGER
# ---------------------------------------------------------------------------

def demo_memory():
    section("1. MemoryManager")

    mem = MemoryManager(max_history=10)

    # Short-term history
    mem.add_message("system", "You are a helpful assistant.")
    mem.add_message("user", "What is the capital of France?")
    mem.add_message("assistant", "The capital of France is Paris.")
    mem.add_message("user", "And Germany?")
    mem.add_message("assistant", "Berlin is the capital of Germany.")

    history = mem.get_history()
    ok(f"Short-term history has {len(history)} messages")

    last = mem.last_message()
    ok(f"Last message — role='{last.role}', content='{last.content[:40]}'")

    # Search
    results = mem.search_history("capital", top_k=3)
    ok(f"Search 'capital' returned {len(results)} result(s)")

    # API-ready format
    api_msgs = mem.get_messages_for_api(include_system=False)
    info(f"API-formatted messages (no system): {api_msgs}")

    # Long-term memory
    mem.remember("user_name", "Alice")
    mem.remember("preferences", {"theme": "dark", "lang": "en"})
    ok(f"Recalled user_name: {mem.recall('user_name')}")
    ok(f"Recalled preferences: {mem.recall('preferences')}")
    mem.forget("user_name")
    ok(f"After forget, user_name={mem.recall('user_name', default='(not found)')}")
    ok(f"Long-term keys: {mem.list_memories()}")

    # Ephemeral context
    mem.set_context("current_topic", "geography")
    ok(f"Ephemeral context: {mem.get_context('current_topic')}")
    mem.clear_context()
    ok(f"After clear_context: {mem.get_context('current_topic', default='(empty)')}")

    # Persistence
    save_path = "/tmp/wizardai_demo_mem.json"
    mem.save(save_path)
    ok(f"Memory saved to {save_path}")

    mem2 = MemoryManager(max_history=10)
    mem2.load(save_path)
    ok(f"Loaded memory: {len(mem2.get_history())} messages, "
       f"long_term_keys={mem2.list_memories()}")

    print(f"\n  repr: {mem!r}\n")


# ---------------------------------------------------------------------------
# 2. CONVERSATION AGENT
# ---------------------------------------------------------------------------

def demo_conversation():
    section("2. ConversationAgent")

    mem   = MemoryManager(max_history=20)
    agent = ConversationAgent(
        name="Aria",
        fallback="Sorry, I didn't understand that.",
        memory=mem,
    )

    # Basic patterns
    agent.add_pattern("hello",              "Hi there! How can I help you?")
    agent.add_pattern("what is your name",  "I'm Aria, your AI assistant.")
    agent.add_pattern("goodbye",            "Goodbye! Have a wonderful day.")

    # Wildcard patterns
    agent.add_pattern("my name is *",       "Nice to meet you, {wildcard}!")
    agent.add_pattern("i like *",           "That's cool — {wildcard} sounds interesting!")
    agent.add_pattern("what is ? plus ?",   "Math is hard, let me think…")

    # Higher priority (wins when multiple patterns could match)
    agent.add_pattern("hello world",        "Special hello-world response!", priority=10)

    # Callable template
    import random
    agent.add_pattern(
        "tell me a joke",
        lambda text, ctx: random.choice([
            "Why do programmers love dark mode? Because light attracts bugs!",
            "I told a UDP joke. Not sure if you got it.",
            "A SQL query walks into a bar, walks up to two tables and asks… 'Can I join you?'",
        ]),
    )

    # List of random responses
    agent.add_pattern(
        "flip a coin",
        ["Heads!", "Tails!"],
    )

    # Context-aware patterns
    agent.add_pattern("yes", "Great, proceeding!", context="confirm")
    agent.add_pattern("no",  "OK, cancelled.",      context="confirm")

    test_inputs = [
        "hello",
        "hello world",              # higher priority
        "what is your name",
        "my name is Bob",
        "i like skiing",
        "tell me a joke",
        "flip a coin",
        "flip a coin",              # should differ sometimes
        "goodbye",
        "this makes no sense",      # → fallback
    ]

    for text in test_inputs:
        reply = agent.respond(text)
        ok(f"  '{text}' → '{reply}'")

    # Context-aware test
    agent.set_context("confirm")
    ok(f"Context set to 'confirm' → 'yes' → '{agent.respond('yes')}'")
    agent.clear_context()

    # Pattern list / removal
    patterns = agent.list_patterns()
    ok(f"Total patterns registered: {len(patterns)}")

    agent.remove_pattern("goodbye")
    ok(f"After removing 'goodbye': {len(agent.list_patterns())} patterns")

    print(f"\n  repr: {agent!r}\n")


# ---------------------------------------------------------------------------
# 3. AI CLIENT
# ---------------------------------------------------------------------------

def demo_ai_client():
    section("3. AIClient")

    # --- Detect which key is available ---
    openai_key     = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key  = os.environ.get("ANTHROPIC_API_KEY", "")
    hf_key         = os.environ.get("HUGGINGFACE_API_KEY", "")

    if not any([openai_key, anthropic_key, hf_key]):
        info("No API keys found in environment. Skipping live LLM calls.")
        info("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or HUGGINGFACE_API_KEY to enable.")

        # Show client creation and repr without calling the API
        client = AIClient(backend="openai", api_key="sk-placeholder")
        ok(f"AIClient created: {client!r}")
        return

    # Choose backend
    if openai_key:
        client = AIClient(backend="openai", api_key=openai_key)
        info(f"Using OpenAI backend: {client!r}")
    elif anthropic_key:
        client = AIClient(backend="anthropic", api_key=anthropic_key)
        info(f"Using Anthropic backend: {client!r}")
    else:
        client = AIClient(
            backend="huggingface",
            api_key=hf_key,
            model="mistralai/Mistral-7B-Instruct-v0.2",
        )
        info(f"Using HuggingFace backend: {client!r}")

    # Single-turn completion
    try:
        resp = client.complete(
            "In one sentence, what is the Pythagorean theorem?",
            max_tokens=80,
        )
        ok(f"complete() → '{resp.text.strip()}'")
        ok(f"  model={resp.model}, latency={resp.latency_ms:.0f}ms, usage={resp.usage}")
    except (APIError, AuthenticationError, RateLimitError) as e:
        info(f"LLM call skipped: {e}")
        return

    # Multi-turn chat
    messages = [
        {"role": "user",      "content": "My favourite colour is blue."},
        {"role": "assistant", "content": "That's a calming colour!"},
        {"role": "user",      "content": "What is my favourite colour?"},
    ]
    try:
        resp = client.chat(messages, max_tokens=50)
        ok(f"chat() → '{resp.text.strip()}'")
    except APIError as e:
        info(f"chat() skipped: {e}")

    # Runtime model switch
    client.set_model(client.model)  # no-op but shows API
    ok(f"Model after set_model: {client.model}")

    print(f"\n  repr: {client!r}\n")


# ---------------------------------------------------------------------------
# 4. PLUGIN SYSTEM
# ---------------------------------------------------------------------------

def demo_plugins():
    section("4. Plugin System")

    # --- Define custom plugins ---

    class GreetPlugin(PluginBase):
        name        = "greeter"
        description = "Says hello."
        version     = "1.0.0"
        triggers    = ["hello *"]

        def on_message(self, text: str, context: dict) -> Optional[str]:
            name = text.replace("hello", "").strip() or "stranger"
            return f"Hello, {name.title()}! 👋"

    class ReversePlugin(PluginBase):
        name        = "reverse"
        description = "Reverses the input text."
        version     = "1.0.0"

        def on_message(self, text: str, context: dict) -> Optional[str]:
            if text.lower().startswith("reverse "):
                payload = text[8:]
                return payload[::-1]
            return None  # pass through

    class MathPlugin(PluginBase):
        name    = "math"
        version = "1.0.0"

        def setup(self):
            self.precision = self.config.get("precision", 2)

        def on_message(self, text: str, context: dict) -> Optional[str]:
            try:
                result = eval(text, {"__builtins__": {}})  # noqa: S307
                return f"Result: {round(float(result), self.precision)}"
            except Exception:
                return None

    # --- PluginManager ---
    manager = PluginManager()

    greeter = manager.register(GreetPlugin)
    reverser = manager.register(ReversePlugin)
    math_p  = manager.register(MathPlugin, config={"precision": 4})

    ok(f"Registered {len(manager)} plugins: {manager!r}")

    # Dispatch (first match wins)
    test_cases = [
        "hello alice",
        "reverse hello world",
        "2 + 2 * 10",
        "something unhandled",   # → None
    ]
    for text in test_cases:
        result = manager.dispatch(text)
        ok(f"  dispatch({text!r}) → {result!r}")

    # dispatch_all
    all_results = manager.dispatch_all("hello plugin test")
    ok(f"dispatch_all results: {all_results}")

    # Enable / disable
    greeter.disable()
    ok(f"After greeter.disable(): dispatch('hello bob') → "
       f"{manager.dispatch('hello bob')!r}")
    greeter.enable()
    ok(f"After greeter.enable() : dispatch('hello bob') → "
       f"{manager.dispatch('hello bob')!r}")

    # List plugins
    for p in manager.list_plugins():
        info(f"  {p!r}")

    # Enable-only list
    active = manager.list_plugins(enabled_only=True)
    ok(f"Enabled plugins: {len(active)}")

    # Session lifecycle hooks
    manager.start_all()
    manager.stop_all()
    ok("start_all() / stop_all() called")

    # Unregister
    removed = manager.unregister("greeter")
    ok(f"unregister('greeter') → {removed}, remaining={len(manager)}")

    # Load from file (write a temp plugin file)
    import tempfile, textwrap
    plugin_src = textwrap.dedent("""\
        from wizardai import PluginBase
        from typing import Optional

        class EchoPlugin(PluginBase):
            name    = "echo"
            version = "0.1.0"

            def on_message(self, text, context) -> Optional[str]:
                return f"Echo: {text}"
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="wizardai_echo_"
    ) as f:
        f.write(plugin_src)
        tmp_path = f.name

    try:
        echo_plugin = manager.load_from_file(tmp_path)
        ok(f"load_from_file → {echo_plugin!r}")
        ok(f"dispatch('ping') → {manager.dispatch('ping')!r}")
    except PluginError as e:
        info(f"load_from_file skipped: {e}")
    finally:
        os.unlink(tmp_path)

    print()


# ---------------------------------------------------------------------------
# 5. VISION MODULE
# ---------------------------------------------------------------------------

def demo_vision():
    section("5. VisionModule")

    try:
        import cv2  # noqa: F401
    except ImportError:
        info("opencv-python not installed. Install with: pip install 'wizardai[vision]'")
        return

    try:
        cam = VisionModule(device_id=0, width=640, height=480, fps=30)
        cam.open()
        ok(f"Camera opened: {cam!r}")
    except Exception as e:
        info(f"Camera unavailable: {e}  (Skipping vision demo)")
        return

    # Capture a single frame
    try:
        frame = cam.capture_frame()
        if frame is not None:
            ok(f"Frame captured: shape={frame.shape}, dtype={frame.dtype}")

            # Save
            save_path = "/tmp/wizardai_snapshot.jpg"
            saved = cam.save_frame(frame, save_path)
            ok(f"Frame saved to: {saved}")

            # Grayscale
            gray = cam.to_grayscale(frame)
            ok(f"Grayscale frame: shape={gray.shape}")

            # Resize
            small = cam.resize(frame, width=320)
            ok(f"Resized frame: shape={small.shape}")

            # Base64 encode (for vision-LLM calls)
            b64 = cam.encode_to_base64(frame)
            ok(f"Base64-encoded frame: {len(b64)} chars (first 40: {b64[:40]}…)")

            # Face detection
            faces = cam.detect_faces(frame)
            ok(f"Faces detected: {len(faces)}")

        # Streaming (short burst)
        frame_count = 0
        def on_frame(f):
            nonlocal frame_count
            frame_count += 1

        cam.start_stream(callback=on_frame, show_preview=False)
        time.sleep(2)
        cam.stop_stream()
        ok(f"Stream ran for ~2s, processed {frame_count} frame(s)")

    except VisionError as e:
        info(f"Vision error: {e}")
    finally:
        cam.close()
        ok("Camera closed")

    # Context manager usage
    try:
        with VisionModule() as cam2:
            frame = cam2.capture_frame()
            ok(f"Context-manager frame: {'captured' if frame is not None else 'None'}")
    except Exception as e:
        info(f"Context-manager cam: {e}")

    print()


# ---------------------------------------------------------------------------
# 6. SPEECH MODULE
# ---------------------------------------------------------------------------

def demo_speech():
    section("6. SpeechModule")

    try:
        import speech_recognition  # noqa: F401
    except ImportError:
        info("SpeechRecognition not installed. Install with: pip install 'wizardai[speech]'")
        return

    try:
        speech = SpeechModule(
            stt_backend="google",
            tts_backend="pyttsx3",
            language="en-US",
            tts_rate=150,
            tts_volume=1.0,
        )
        speech.init_tts()
        ok(f"SpeechModule created: {speech!r}")
    except Exception as e:
        info(f"SpeechModule init failed: {e}")
        return

    # TTS — non-blocking so demo doesn't hang
    try:
        speech.say("Hello from WizardAI!", blocking=False)
        ok("say() called (non-blocking)")
        time.sleep(2)
    except SpeechError as e:
        info(f"TTS error: {e}")

    # Save TTS audio to file
    try:
        audio_path = "/tmp/wizardai_greeting.mp3"
        speech.save_audio("Greetings from WizardAI.", audio_path)
        ok(f"Audio saved to {audio_path}")
    except Exception as e:
        info(f"save_audio skipped: {e}")

    # List microphones
    try:
        mics = speech.list_microphones()
        ok(f"Found {len(mics)} microphone(s)")
        for i, name in enumerate(mics[:3]):
            info(f"  [{i}] {name}")
    except Exception as e:
        info(f"list_microphones error: {e}")

    # STT — non-interactive: just show the API
    info("listen() demo skipped (no live mic input in batch mode).")
    info("To test: text = speech.listen(timeout=5)")

    # Transcribe from file (if a wav exists)
    wav = "/tmp/test.wav"
    if os.path.exists(wav):
        try:
            text = speech.transcribe_file(wav)
            ok(f"transcribe_file('{wav}') → '{text}'")
        except Exception as e:
            info(f"transcribe_file error: {e}")
    else:
        info(f"Place a .wav file at {wav} to test transcribe_file().")

    print()


# ---------------------------------------------------------------------------
# 7. WIZARDAI CORE (ties everything together)
# ---------------------------------------------------------------------------

def demo_core():
    section("7. WizardAI Core (full integration)")

    openai_key    = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Build config based on available keys
    wiz_kwargs = dict(
        agent_name="WizardBot",
        fallback_response="I don't know the answer to that.",
        max_history=20,
        log_level="WARNING",
        data_dir="/tmp/wizardai_demo_data",
    )

    if openai_key:
        wiz_kwargs.update(ai_backend="openai", openai_api_key=openai_key)
        info("Core demo using OpenAI backend")
    elif anthropic_key:
        wiz_kwargs.update(ai_backend="anthropic", anthropic_api_key=anthropic_key)
        info("Core demo using Anthropic backend")
    else:
        info("No API key found — LLM calls will be skipped; patterns will still work.")
        wiz_kwargs.update(
            ai_backend="openai",
            openai_api_key="sk-placeholder",
        )

    # Instantiate
    wiz = WizardAI(**wiz_kwargs)
    ok(f"WizardAI created: {wiz!r}")

    # --- Patterns ---
    wiz.agent.add_pattern("ping",          "pong!")
    wiz.agent.add_pattern("hello *",       "Hello, {wildcard}!")
    wiz.agent.add_pattern("what time is it",
                          lambda t, c: f"It's {time.strftime('%H:%M:%S')}.")
    wiz.agent.add_pattern("roll a die",    ["1","2","3","4","5","6"])

    # Start session
    wiz.start()
    ok("wiz.start() called")

    # Pattern-based chat (no API key needed)
    for q in ["ping", "hello WizardAI", "what time is it", "roll a die"]:
        reply = wiz.chat(q)
        ok(f"  chat({q!r}) → {reply!r}")

    # LLM call
    if openai_key or anthropic_key:
        try:
            reply = wiz.ask("In exactly one sentence, what is Python?")
            ok(f"  ask() → {reply.strip()!r}")
        except APIError as e:
            info(f"  ask() failed: {e}")
    else:
        info("  ask() skipped (no API key)")

    # Memory shortcuts
    wiz.remember("demo_key", {"run": True, "ts": time.time()})
    ok(f"  recall('demo_key') → {wiz.recall('demo_key')}")
    ok(f"  get_history(3) → {len(wiz.get_history(3))} entries")

    # Plugin via core
    class PingPlugin(PluginBase):
        name = "ping_plugin"
        def on_message(self, text, context):
            return "PLUGIN PONG" if text.strip() == "plugin ping" else None

    wiz.add_plugin(PingPlugin)
    ok(f"  Plugin dispatch → {wiz.plugins.dispatch('plugin ping')!r}")

    # System prompt
    wiz.set_system_prompt("You are a concise, helpful assistant.")
    ok("  set_system_prompt() called")

    # Stop
    wiz.stop()
    ok("wiz.stop() called")

    # Context-manager pattern
    with WizardAI(
        openai_api_key=openai_key or "sk-placeholder",
        log_level="ERROR",
    ) as wiz2:
        wiz2.agent.add_pattern("hello", "Hi from context manager!")
        ok(f"  Context-manager wiz2.chat('hello') → {wiz2.chat('hello')!r}")

    print(f"\n  repr: {wiz!r}\n")


# ---------------------------------------------------------------------------
# 8. EXCEPTIONS
# ---------------------------------------------------------------------------

def demo_exceptions():
    section("8. Exception Hierarchy")

    from wizardai.exceptions import (
        WizardAIError, APIError, RateLimitError, AuthenticationError,
        VisionError, CameraNotFoundError,
        SpeechError, MicrophoneNotFoundError,
        ConversationError, PluginError, ConfigurationError,
    )

    hierarchy = [
        WizardAIError("base error"),
        APIError("api failed", code=500, backend="openai"),
        RateLimitError(retry_after=30.0),
        AuthenticationError("anthropic"),
        VisionError("camera error"),
        CameraNotFoundError(device_id=2),
        SpeechError("tts failed"),
        MicrophoneNotFoundError(),
        ConversationError("pattern error"),
        PluginError("plugin failed", plugin_name="my_plugin"),
        ConfigurationError("bad config"),
    ]

    for exc in hierarchy:
        ok(f"  {exc!r}")

    # Show try-except pattern
    def fake_api_call(key: str):
        if not key or key == "sk-placeholder":
            raise AuthenticationError("openai")
        raise RateLimitError(retry_after=60.0)

    try:
        fake_api_call("sk-placeholder")
    except AuthenticationError as e:
        ok(f"Caught AuthenticationError: backend={e.backend}, code={e.code}")
    except RateLimitError as e:
        ok(f"Caught RateLimitError: retry_after={e.retry_after}")
    except WizardAIError as e:
        ok(f"Caught WizardAIError: {e.message}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n🧙  WizardAI SDK — Full Demo")
    print(f"    SDK version : {wizardai.__version__}")
    print(f"    Python      : {sys.version.split()[0]}")

    # Run all demos in order
    demo_memory()
    demo_conversation()
    demo_ai_client()
    demo_plugins()
    demo_vision()
    demo_speech()
    demo_core()
    demo_exceptions()

    print(SEPARATOR)
    print("  All demos complete! 🎉")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
