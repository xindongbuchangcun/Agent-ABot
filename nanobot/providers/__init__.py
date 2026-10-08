"""LLM provider abstractions with lazy optional-provider imports."""

from importlib import import_module

__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "OpenAICodexProvider", "AzureOpenAIProvider"]

_PUBLIC_IMPORTS = {
    "LLMProvider": ("nanobot.providers.base", "LLMProvider"),
    "LLMResponse": ("nanobot.providers.base", "LLMResponse"),
    "LiteLLMProvider": ("nanobot.providers.litellm_provider", "LiteLLMProvider"),
    "OpenAICodexProvider": ("nanobot.providers.openai_codex_provider", "OpenAICodexProvider"),
    "AzureOpenAIProvider": ("nanobot.providers.azure_openai_provider", "AzureOpenAIProvider"),
}


def __getattr__(name: str):
    if name not in _PUBLIC_IMPORTS:
        raise AttributeError(name)
    module_name, attribute = _PUBLIC_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
