import importlib
import sys
import types

import pytest

# We'll patch google.genai.Client to avoid requiring the actual package
class DummyClient:
    def __init__(self, *args, **kwargs):
        pass


def reload_graph(monkeypatch):
    """Helper to reload agent.graph with patched google.genai.Client"""
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = DummyClient
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    if "agent.graph" in sys.modules:
        del sys.modules["agent.graph"]
    import agent.graph  # noqa: F401
    return importlib.reload(sys.modules["agent.graph"])


def test_import_fails_without_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    with pytest.raises(ValueError):
        reload_graph(monkeypatch)


def test_generate_query_uses_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "foo")
    monkeypatch.setenv("GEMINI_API_KEY", "bar")
    mod = reload_graph(monkeypatch)

    captured = {}

    class DummyLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema):
            class Runner:
                def invoke(self, prompt):
                    class Result:
                        query = ["test-query"]
                    return Result()
            return Runner()

    monkeypatch.setattr(mod, "ChatOpenAI", DummyLLM)

    from langchain_core.messages import HumanMessage
    state = {"messages": [HumanMessage(content="hello")]} 
    result = mod.generate_query(state, {})
    assert result["query_list"] == ["test-query"]
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "foo"
