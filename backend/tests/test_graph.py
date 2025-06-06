import os
import pytest
from unittest.mock import MagicMock, patch, call

from langchain_core.runnables import RunnableConfig

# Modules to test
from agent.configuration import Configuration
from agent.graph import (
    _get_llm_client,
    _perform_google_search,
    _perform_local_search,
    web_research
)
from agent.tools_and_schemas import LocalSearchTool, LocalSearchOutput, LocalSearchResult
from agent.state import WebSearchState, OverallState

# Mock LLM Clients that might be returned by _get_llm_client
MockChatGoogleGenerativeAI = MagicMock()
MockChatOpenAI = MagicMock()

# Actual LLM client classes (for type checking if needed, not for instantiation in tests)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


@pytest.fixture(autouse=True)
def clear_env_vars():
    """Fixture to clear relevant environment variables before each test and restore after."""
    original_environ = dict(os.environ)
    env_keys_to_clear = ["GEMINI_API_KEY", "LLM_API_KEY", "LANGCHAIN_API_KEY"]
    for key in env_keys_to_clear:
        if key in os.environ:
            del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(original_environ)


@patch('agent.graph.ChatGoogleGenerativeAI', new=MockChatGoogleGenerativeAI)
@patch('agent.graph.ChatOpenAI', new=MockChatOpenAI)
class TestGetLlmClient:

    def setup_method(self):
        MockChatGoogleGenerativeAI.reset_mock()
        MockChatOpenAI.reset_mock()
        os.environ["GEMINI_API_KEY"] = "dummy_gemini_key" # Needs to be present for Gemini fallback

    def test_get_gemini_client_default(self):
        config_data = Configuration(llm_provider="gemini", query_generator_model="gemini-test-model")
        llm = _get_llm_client(config_data, config_data.query_generator_model)
        MockChatGoogleGenerativeAI.assert_called_once_with(
            model="gemini-test-model",
            temperature=0.0, # Default in helper
            max_retries=2,   # Default in helper
            api_key="dummy_gemini_key"
        )
        assert llm == MockChatGoogleGenerativeAI.return_value

    def test_get_gemini_client_with_llm_api_key(self):
        config_data = Configuration(llm_provider="gemini", llm_api_key="override_gemini_key", query_generator_model="gemini-test-model")
        _get_llm_client(config_data, config_data.query_generator_model)
        MockChatGoogleGenerativeAI.assert_called_once_with(
            model="gemini-test-model",
            temperature=0.0,
            max_retries=2,
            api_key="override_gemini_key" # LLM_API_KEY takes precedence
        )

    def test_get_openrouter_client(self):
        config_data = Configuration(
            llm_provider="openrouter",
            llm_api_key="or_key",
            openrouter_model_name="or/model",
            query_generator_model="should_be_ignored_if_or_model_name_is_set" # Fallback if openrouter_model_name is None
        )
        llm = _get_llm_client(config_data, config_data.query_generator_model, temperature=0.5)
        MockChatOpenAI.assert_called_once_with(
            model_name="or/model",
            openai_api_key="or_key",
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.5, # Passed from args
            max_retries=2
        )
        assert llm == MockChatOpenAI.return_value

    def test_get_openrouter_client_uses_task_model_if_specific_not_set(self):
        config_data = Configuration(
            llm_provider="openrouter",
            llm_api_key="or_key",
            # openrouter_model_name is None
            query_generator_model="actual_or_slug/model"
        )
        _get_llm_client(config_data, config_data.query_generator_model)
        MockChatOpenAI.assert_called_once_with(
            model_name="actual_or_slug/model", # Falls back to task_model_name
            openai_api_key="or_key",
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_retries=2
        )

    def test_get_deepseek_client(self):
        config_data = Configuration(
            llm_provider="deepseek",
            llm_api_key="ds_key",
            deepseek_model_name="deepseek-chat-test",
            query_generator_model="ignored_model"
        )
        _get_llm_client(config_data, config_data.query_generator_model)
        MockChatOpenAI.assert_called_once_with(
            model_name="deepseek-chat-test",
            openai_api_key="ds_key",
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.0,
            max_retries=2
        )

    def test_unsupported_provider_raises_error(self):
        config_data = Configuration(llm_provider="unknown_provider", llm_api_key="some_key")
        with pytest.raises(ValueError, match="Unsupported LLM provider: unknown_provider"):
            _get_llm_client(config_data, "any_model")

    def test_missing_api_key_for_openrouter(self):
        config_data = Configuration(llm_provider="openrouter") # No llm_api_key
        with pytest.raises(ValueError, match="LLM_API_KEY must be set for OpenRouter provider."):
            _get_llm_client(config_data, "any_model")

    def test_missing_api_key_for_deepseek(self):
        config_data = Configuration(llm_provider="deepseek") # No llm_api_key
        with pytest.raises(ValueError, match="LLM_API_KEY must be set for DeepSeek provider."):
            _get_llm_client(config_data, "any_model")

    def test_missing_gemini_api_key_raises_error(self):
        # Temporarily remove GEMINI_API_KEY for this specific test
        original_gemini_key = os.environ.pop("GEMINI_API_KEY", None)

        config_data = Configuration(llm_provider="gemini", llm_api_key=None) # No specific key, and global one removed
        with pytest.raises(ValueError, match="GEMINI_API_KEY must be set for Gemini provider"):
            _get_llm_client(config_data, "gemini-model")

        # Restore if it was there
        if original_gemini_key is not None:
            os.environ["GEMINI_API_KEY"] = original_gemini_key


class TestPerformGoogleSearch:
    @patch('agent.graph.genai_client') # Mock the global genai_client used by _perform_google_search
    @patch('agent.graph.resolve_urls')
    @patch('agent.graph.get_citations')
    @patch('agent.graph.insert_citation_markers')
    def test_successful_google_search(self, mock_insert_markers, mock_get_citations, mock_resolve_urls, mock_genai_client_module):
        # Setup mocks
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].grounding_metadata.grounding_chunks = ["chunk1"]
        mock_response.text = "Raw text from Google search with Gemini."
        mock_genai_client_module.models.generate_content.return_value = mock_response

        mock_resolve_urls.return_value = [{"url": "http://resolved.com", "short_url": "res_short"}]
        mock_get_citations.return_value = [{"segments": [{"id": "seg1", "text": "segment text"}]}]
        mock_insert_markers.return_value = "Modified text with citations."

        state = WebSearchState(search_query="test query", id=1)
        configurable = Configuration(query_generator_model="gemini-for-search") # Model used by Google Search tool

        sources, results = _perform_google_search(state, configurable, mock_genai_client_module)

        mock_genai_client_module.models.generate_content.assert_called_once()
        mock_resolve_urls.assert_called_once()
        mock_get_citations.assert_called_once()
        mock_insert_markers.assert_called_once_with("Raw text from Google search with Gemini.", mock_get_citations.return_value)

        assert len(sources) == 1
        assert sources[0]["id"] == "seg1"
        assert results == ["Modified text with citations."]

    @patch('agent.graph.genai_client')
    def test_google_search_no_results(self, mock_genai_client_module):
        mock_response = MagicMock()
        mock_response.candidates = [] # No candidates
        mock_genai_client_module.models.generate_content.return_value = mock_response

        state = WebSearchState(search_query="query with no results", id=2)
        configurable = Configuration()

        sources, results = _perform_google_search(state, configurable, mock_genai_client_module)
        assert sources == []
        assert results == []

    @patch('agent.graph.genai_client')
    def test_google_search_api_error(self, mock_genai_client_module):
        mock_genai_client_module.models.generate_content.side_effect = Exception("API Error")

        state = WebSearchState(search_query="query causing error", id=3)
        configurable = Configuration()

        sources, results = _perform_google_search(state, configurable, mock_genai_client_module)
        assert sources == []
        assert results == []


class TestPerformLocalSearch:
    @patch('agent.graph.local_search_tool', spec=LocalSearchTool) # Mock the global local_search_tool
    def test_successful_local_search(self, mock_tool_instance):
        mock_tool_instance._run.return_value = LocalSearchOutput(results=[
            LocalSearchResult(url="http://local1.com", title="Local Page 1", snippet="Snippet 1 for test query"),
            LocalSearchResult(url="http://local2.com", title="Local Page 2", snippet="Another snippet")
        ])

        state = WebSearchState(search_query="test query", id=1)
        configurable = Configuration(enable_local_search=True, local_search_domains=["http://local1.com"])

        sources, results = _perform_local_search(state, configurable, mock_tool_instance)

        mock_tool_instance._run.assert_called_once_with(query="test query", local_domains=["http://local1.com"])
        assert len(sources) == 2
        assert sources[0]["value"] == "http://local1.com"
        assert sources[0]["title"] == "Local Page 1"
        assert sources[0]["source_type"] == "local"
        assert results[0] == "[LOCAL] Local Page 1: Snippet 1 for test query (Source: http://local1.com)"
        assert len(results) == 2

    @patch('agent.graph.local_search_tool', spec=LocalSearchTool)
    def test_local_search_disabled(self, mock_tool_instance):
        state = WebSearchState(search_query="test query", id=1)
        configurable = Configuration(enable_local_search=False, local_search_domains=["http://local1.com"])

        sources, results = _perform_local_search(state, configurable, mock_tool_instance)
        mock_tool_instance._run.assert_not_called()
        assert sources == []
        assert results == []

    @patch('agent.graph.local_search_tool', spec=LocalSearchTool)
    def test_local_search_no_domains(self, mock_tool_instance):
        state = WebSearchState(search_query="test query", id=1)
        configurable = Configuration(enable_local_search=True, local_search_domains=[]) # No domains

        sources, results = _perform_local_search(state, configurable, mock_tool_instance)
        mock_tool_instance._run.assert_not_called() # Should not run if no domains
        assert sources == []
        assert results == []

    @patch('agent.graph.local_search_tool', spec=LocalSearchTool)
    def test_local_search_tool_error(self, mock_tool_instance):
        mock_tool_instance._run.side_effect = Exception("Tool Error")
        state = WebSearchState(search_query="test query", id=1)
        configurable = Configuration(enable_local_search=True, local_search_domains=["http://err.com"])

        sources, results = _perform_local_search(state, configurable, mock_tool_instance)
        assert sources == []
        assert results == []


@patch('agent.graph._perform_local_search')
@patch('agent.graph._perform_google_search')
class TestWebResearchNode:

    def test_internet_only_mode(self, mock_google_search, mock_local_search):
        mock_google_search.return_value = (["gs_source"], ["gs_result"])
        mock_local_search.return_value = (["ls_source"], ["ls_result"])

        state = WebSearchState(search_query="test", id=1)
        runnable_config = RunnableConfig(configurable={"search_mode": "internet_only"})

        result_state = web_research(state, runnable_config)

        mock_google_search.assert_called_once()
        mock_local_search.assert_not_called()
        assert result_state["sources_gathered"] == ["gs_source"]
        assert result_state["web_research_result"] == ["gs_result"]

    def test_local_only_mode_enabled(self, mock_google_search, mock_local_search):
        mock_google_search.return_value = (["gs_source"], ["gs_result"])
        mock_local_search.return_value = (["ls_source"], ["ls_result"])

        state = WebSearchState(search_query="test", id=1)
        # enable_local_search and local_search_domains are True/non-empty by default in this Configuration for testing
        runnable_config = RunnableConfig(configurable={
            "search_mode": "local_only",
            "enable_local_search": True,
            "local_search_domains": ["http://a.com"]
        })

        result_state = web_research(state, runnable_config)

        mock_google_search.assert_not_called()
        mock_local_search.assert_called_once()
        assert result_state["sources_gathered"] == ["ls_source"]
        assert result_state["web_research_result"] == ["ls_result"]

    def test_local_only_mode_disabled_config(self, mock_google_search, mock_local_search):
        state = WebSearchState(search_query="test", id=1)
        runnable_config = RunnableConfig(configurable={
            "search_mode": "local_only",
            "enable_local_search": False # Local search disabled
        })

        result_state = web_research(state, runnable_config)

        mock_google_search.assert_not_called()
        mock_local_search.assert_not_called() # Not called because it's disabled in config
        assert "No local results found" in result_state["web_research_result"][0]

    def test_internet_then_local_mode(self, mock_google_search, mock_local_search):
        mock_google_search.return_value = (["gs_source"], ["gs_result"])
        mock_local_search.return_value = (["ls_source"], ["ls_result"])

        state = WebSearchState(search_query="test", id=1)
        runnable_config = RunnableConfig(configurable={
            "search_mode": "internet_then_local",
            "enable_local_search": True,
            "local_search_domains": ["http://a.com"]
        })
        result_state = web_research(state, runnable_config)

        mock_google_search.assert_called_once()
        mock_local_search.assert_called_once()
        assert result_state["sources_gathered"] == ["gs_source", "ls_source"]
        assert result_state["web_research_result"] == ["gs_result", "ls_result"]

    def test_local_then_internet_mode(self, mock_google_search, mock_local_search):
        mock_google_search.return_value = (["gs_source"], ["gs_result"])
        mock_local_search.return_value = (["ls_source"], ["ls_result"])

        state = WebSearchState(search_query="test", id=1)
        runnable_config = RunnableConfig(configurable={
            "search_mode": "local_then_internet",
            "enable_local_search": True,
            "local_search_domains": ["http://a.com"]
        })
        result_state = web_research(state, runnable_config)

        mock_google_search.assert_called_once()
        mock_local_search.assert_called_once()
        # Order of calls is implicitly tested by the setup of these mocks if needed,
        # but here we check combined results.
        assert result_state["sources_gathered"] == ["ls_source", "gs_source"]
        assert result_state["web_research_result"] == ["ls_result", "gs_result"]

    def test_unknown_mode_defaults_to_internet_only(self, mock_google_search, mock_local_search):
        mock_google_search.return_value = (["gs_source"], ["gs_result"])

        state = WebSearchState(search_query="test", id=1)
        runnable_config = RunnableConfig(configurable={"search_mode": "some_unknown_mode"})
        result_state = web_research(state, runnable_config)

        mock_google_search.assert_called_once()
        mock_local_search.assert_not_called()
        assert result_state["sources_gathered"] == ["gs_source"]
        assert result_state["web_research_result"] == ["gs_result"]

    def test_no_results_found_message(self, mock_google_search, mock_local_search):
        mock_google_search.return_value = ([], []) # No google results
        mock_local_search.return_value = ([], [])  # No local results either

        state = WebSearchState(search_query="nothing_found_query", id=1)
        runnable_config = RunnableConfig(configurable={
            "search_mode": "internet_then_local", # Try both
            "enable_local_search": True,
            "local_search_domains": ["http://a.com"]
        })
        result_state = web_research(state, runnable_config)

        assert result_state["sources_gathered"] == []
        assert len(result_state["web_research_result"]) == 1
        assert "No results found for query" in result_state["web_research_result"][0]

# Note: For a full test suite, you'd also want to test
# the `generate_query`, `reflection`, `finalize_answer` nodes,
# and the overall graph compilation and execution flow.
# These tests focus on the modified/new parts related to LLM clients and search modes.

```
