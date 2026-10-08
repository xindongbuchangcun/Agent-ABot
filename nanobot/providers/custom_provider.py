"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urlparse

import json_repair
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):

    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
        seed: int | None = 0,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.seed = seed
        # Loopback model servers must never inherit HTTP_PROXY. Without an
        # explicit NO_PROXY, httpx otherwise sends localhost traffic through
        # the proxy and surfaces its 502 as a model-server failure.
        hostname = (urlparse(api_base).hostname or "").lower()
        client_options: dict[str, Any] = {}
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            client_options["http_client"] = DefaultAsyncHttpxClient(
                trust_env=False
            )
        # Keep affinity stable for this provider instance to improve backend cache locality.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
            **client_options,
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None,
                   tool_choice: str | dict[str, Any] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            # AgentNav's tool loop executes one call, returns its geometric
            # result, and only then asks the model for the next decision.
            # Disabling parallel calls also prevents local guided decoding
            # from emitting an unbounded top-level array of repeated calls.
            kwargs.update(
                tools=tools,
                tool_choice=tool_choice or "auto",
                parallel_tool_calls=False,
            )
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in msg.tool_calls or []:
            arguments = (
                json_repair.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments
            )
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"tool arguments must decode to a JSON object: {tc.function.name}"
                )
            tool_calls.append(
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

