from nanobot.providers.custom_provider import CustomProvider


def test_loopback_provider_ignores_environment_proxy():
    provider = CustomProvider(api_base="http://localhost:8000/v1")
    assert provider._client._client.trust_env is False


def test_remote_provider_keeps_environment_proxy_behavior():
    provider = CustomProvider(api_base="https://example.com/v1")
    assert provider._client._client.trust_env is True


def test_custom_provider_disables_parallel_tool_calls():
    import asyncio
    from types import SimpleNamespace

    provider = CustomProvider(api_base="http://localhost:8000/v1")
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    provider._client.chat.completions.create = create
    asyncio.run(
        provider.chat(
            messages=[{"role": "user", "content": "test"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "QUERY_DEPTH",
                        "description": "query",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice="required",
        )
    )

    assert captured["parallel_tool_calls"] is False
    assert captured["seed"] == 0
