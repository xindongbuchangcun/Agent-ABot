"""Agent core module with dependency-safe lazy public imports."""

from importlib import import_module

__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore", "SkillsLoader"]

_PUBLIC_IMPORTS = {
    "AgentLoop": ("nanobot.agent.loop", "AgentLoop"),
    "ContextBuilder": ("nanobot.agent.context", "ContextBuilder"),
    "MemoryStore": ("nanobot.agent.memory", "MemoryStore"),
    "SkillsLoader": ("nanobot.agent.skills", "SkillsLoader"),
}


def __getattr__(name: str):
    if name not in _PUBLIC_IMPORTS:
        raise AttributeError(name)
    module_name, attribute = _PUBLIC_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
