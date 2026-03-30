"""
WizardAI AI Client
------------------
Lightweight interface for any OpenAI-compatible REST endpoint.
Supports streaming, automatic retry with exponential back-off, and rate limiting.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

from .exceptions import APIError, AuthenticationError, RateLimitError
from .utils import Logger, RateLimiter


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AIResponse:
    """Structured response returned by :class:`AIClient`.

    Attributes:
        text:         The generated text content.
        model:        Model identifier used for generation.
        usage:        Token / resource usage statistics.
        raw:          The raw response dict from the API.
        latency_ms:   Round-trip latency in milliseconds.
    """
    text: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def __str__(self):
        return self.text


# ---------------------------------------------------------------------------
# AIClient
# ---------------------------------------------------------------------------

class AIClient:
    """Lightweight client for any OpenAI-compatible REST endpoint.

    Handles API-key management (env var or explicit arg), model selection,
    streaming, rate limiting, and retry logic.

    Example::

        # Basic usage
        client = AIClient(
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key="sk-...",
            model="gpt-4o",
        )
        response = client.chat([{"role": "user", "content": "Hello!"}])
        print(response.text)

        # Streaming
        for chunk in client.chat_stream([{"role": "user", "content": "Tell me a story"}]):
            print(chunk, end="", flush=True)

        # Anthropic endpoint (OpenAI-compatible via proxy, or direct)
        client = AIClient(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-...",
            model="claude-opus-4-5",
        )

        # Local / self-hosted (e.g. Ollama, LM Studio, vLLM)
        client = AIClient(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="llama3",
        )
    """

    # Environment variable to look up the API key when not supplied explicitly
    _ENV_KEY = "WIZARDAI_API_KEY"

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
        rate_limit_calls: int = 60,
        rate_limit_period: float = 60.0,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        """
        Args:
            endpoint:           Full URL of the chat completions endpoint.
                                Examples:
                                  - "https://api.openai.com/v1/chat/completions"
                                  - "http://localhost:11434/v1/chat/completions"
            api_key:            Bearer token / API key.  Falls back to the
                                ``WIZARDAI_API_KEY`` environment variable.
                                Can be omitted for unauthenticated local servers.
            model:              Model identifier to send with every request.
            max_retries:        Number of retry attempts on transient errors.
            retry_delay:        Initial delay (seconds) between retries
                                (doubles on each attempt).
            timeout:            HTTP request timeout in seconds.
            rate_limit_calls:   Max API calls per *rate_limit_period* seconds.
            rate_limit_period:  Rate-limit window in seconds.
            logger:             Optional :class:`~wizardai.utils.Logger` instance.
            **kwargs:           Extra keyword args forwarded to every request.
        """
        if not endpoint:
            raise ValueError("'endpoint' is required. Pass the full URL of your API.")

        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logger or Logger("AIClient")
        self._extra = kwargs

        # Resolve API key (arg → env → empty string for local servers)
        self.api_key = api_key or os.environ.get(self._ENV_KEY, "")
        if not self.api_key:
            self.logger.warning(
                "No API key provided. If your endpoint requires authentication, "
                f"set the {self._ENV_KEY} environment variable or pass api_key=."
            )

        # Rate limiter
        self._rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)

        self.logger.info(
            f"AIClient initialised: endpoint={self.endpoint}, model={self.model}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """Send a multi-turn chat request (non-streaming).

        Args:
            messages:      List of ``{"role": ..., "content": ...}`` dicts.
            model:         Override the default model for this call.
            max_tokens:    Maximum tokens to generate.
            temperature:   Sampling temperature (0 = deterministic).
            system_prompt: Prepend a system message to the conversation.
            **kwargs:      Extra parameters forwarded to the endpoint.

        Returns:
            An :class:`AIResponse` object.

        Raises:
            APIError: On non-retryable API failures.
            RateLimitError: When the rate limit is exceeded.
            AuthenticationError: When the API key is invalid.
        """
        _model = model or self.model
        _messages = self._build_messages(messages, system_prompt)

        return self._with_retry(
            self._call,
            messages=_messages,
            model=_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Send a multi-turn chat request and stream the response token by token.

        Yields text chunks as they arrive from the server-sent events (SSE) stream.

        Args:
            messages:      List of ``{"role": ..., "content": ...}`` dicts.
            model:         Override the default model for this call.
            max_tokens:    Maximum tokens to generate.
            temperature:   Sampling temperature.
            system_prompt: Prepend a system message.
            **kwargs:      Extra parameters forwarded to the endpoint.

        Yields:
            str: Incremental text chunks.

        Example::

            for chunk in client.chat_stream([{"role": "user", "content": "Hi!"}]):
                print(chunk, end="", flush=True)
        """
        _model = model or self.model
        _messages = self._build_messages(messages, system_prompt)

        self._rate_limiter.wait()
        yield from self._stream(
            messages=_messages,
            model=_model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AIResponse:
        """Single-turn text completion convenience wrapper.

        Wraps :meth:`chat` with a single user message.

        Args:
            prompt:     The input prompt string.
            model:      Override the default model.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            An :class:`AIResponse` object.
        """
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def complete_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Single-turn streaming completion convenience wrapper."""
        yield from self.chat_stream(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def set_model(self, model: str):
        """Change the default model used by this client."""
        self.model = model
        self.logger.info(f"Model changed to: {model}")

    def set_api_key(self, api_key: str):
        """Update the API key at runtime."""
        self.api_key = api_key
        self.logger.info("API key updated.")

    def set_endpoint(self, endpoint: str):
        """Update the endpoint URL at runtime."""
        self.endpoint = endpoint.rstrip("/")
        self.logger.info(f"Endpoint updated to: {self.endpoint}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + list(messages)
        return list(messages)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _with_retry(self, fn, **kwargs) -> AIResponse:
        """Call *fn* with retry / back-off on transient errors."""
        self._rate_limiter.wait()

        last_error = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 2):
            try:
                start = time.monotonic()
                response = fn(**kwargs)
                response.latency_ms = (time.monotonic() - start) * 1000
                return response
            except RateLimitError as exc:
                last_error = exc
                wait = exc.retry_after or delay
                self.logger.warning(
                    f"Rate limited. Waiting {wait:.1f}s before retry {attempt}…"
                )
                time.sleep(wait)
                delay *= 2
            except AuthenticationError:
                raise  # never retry auth failures
            except APIError as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    self.logger.warning(
                        f"API error (attempt {attempt}/{self.max_retries}): "
                        f"{exc.message}. Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
            except Exception as exc:
                last_error = APIError(str(exc))
                if attempt <= self.max_retries:
                    self.logger.warning(
                        f"Unexpected error (attempt {attempt}): {exc}. "
                        f"Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise last_error from exc

        raise last_error or APIError("Max retries exceeded")

    # ------------------------------------------------------------------
    # HTTP calls
    # ------------------------------------------------------------------

    def _call(self, messages, model, max_tokens, temperature, stream=False, **kwargs) -> AIResponse:
        """POST to the endpoint (non-streaming path)."""
        try:
            import requests as _req
        except ImportError:
            raise APIError("The 'requests' package is required: pip install requests")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **self._extra,
            **kwargs,
        }

        try:
            r = _req.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except _req.RequestException as exc:
            raise APIError(str(exc)) from exc

        if r.status_code == 401:
            raise AuthenticationError(self.endpoint)
        if r.status_code == 429:
            raise RateLimitError()
        if not r.ok:
            raise APIError(f"Endpoint error {r.status_code}: {r.text}")

        data = r.json()

        # Try OpenAI-compatible format first
        try:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            used_model = data.get("model", model)
        except (KeyError, IndexError):
            # Fallback: stringify the whole response
            text = str(data)
            usage = {}
            used_model = model

        return AIResponse(
            text=text or "",
            model=used_model,
            usage=usage,
            raw=data,
        )

    def _stream(self, messages, model, max_tokens, temperature, **kwargs) -> Generator[str, None, None]:
        """POST to the endpoint with streaming (SSE) and yield text chunks."""
        try:
            import requests as _req
        except ImportError:
            raise APIError("The 'requests' package is required: pip install requests")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **self._extra,
            **kwargs,
        }

        try:
            r = _req.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
        except _req.RequestException as exc:
            raise APIError(str(exc)) from exc

        if r.status_code == 401:
            raise AuthenticationError(self.endpoint)
        if r.status_code == 429:
            raise RateLimitError()
        if not r.ok:
            raise APIError(f"Endpoint error {r.status_code}: {r.text}")

        for raw_line in r.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

            # SSE lines start with "data: "
            if not line.startswith("data: "):
                continue

            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"AIClient(endpoint={self.endpoint!r}, model={self.model!r}, "
            f"key={'***' if self.api_key else 'None'})"
        )
