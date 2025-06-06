import os
import pytest
from pydantic import ValidationError

from agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig

class TestConfiguration:
    # Store original environment variables
    original_environ = None

    @classmethod
    def setup_class(cls):
        """Store original environment variables before any tests run."""
        cls.original_environ = dict(os.environ)

    def setup_method(self):
        """Clear relevant environment variables before each test."""
        env_keys_to_clear = [
            "LLM_PROVIDER", "LLM_API_KEY", "OPENROUTER_MODEL_NAME", "DEEPSEEK_MODEL_NAME",
            "QUERY_GENERATOR_MODEL", "REFLECTION_MODEL", "ANSWER_MODEL",
            "NUMBER_OF_INITIAL_QUERIES", "MAX_RESEARCH_LOOPS", "LANGSMITH_ENABLED",
            "ENABLE_LOCAL_SEARCH", "LOCAL_SEARCH_DOMAINS", "SEARCH_MODE"
        ]
        for key in env_keys_to_clear:
            if key in os.environ:
                del os.environ[key]

    @classmethod
    def teardown_class(cls):
        """Restore original environment variables after all tests."""
        os.environ.clear()
        os.environ.update(cls.original_environ)

    def test_default_values(self):
        """Test that Configuration instantiates with defaults."""
        config = Configuration()
        assert config.llm_provider == "gemini"
        assert config.llm_api_key is None
        assert config.openrouter_model_name is None
        assert config.deepseek_model_name is None
        assert config.query_generator_model == "gemini-1.5-flash"
        assert config.reflection_model == "gemini-1.5-flash"
        assert config.answer_model == "gemini-1.5-pro"
        assert config.number_of_initial_queries == 3
        assert config.max_research_loops == 2
        assert config.langsmith_enabled is True
        assert config.enable_local_search is False
        assert config.local_search_domains == []
        assert config.search_mode == "internet_only"

    def test_env_variable_loading(self):
        """Test loading configuration from environment variables."""
        os.environ["LLM_PROVIDER"] = "openrouter"
        os.environ["LLM_API_KEY"] = "test_api_key_env"
        os.environ["OPENROUTER_MODEL_NAME"] = "env_or_model"
        os.environ["LANGSMITH_ENABLED"] = "false"
        os.environ["LOCAL_SEARCH_DOMAINS"] = "http://site1.env, http://site2.env"
        os.environ["SEARCH_MODE"] = "local_only"
        os.environ["NUMBER_OF_INITIAL_QUERIES"] = "5"

        # For from_runnable_config, env vars are loaded if not in RunnableConfig
        config = Configuration.from_runnable_config(RunnableConfig(configurable={}))

        assert config.llm_provider == "openrouter"
        assert config.llm_api_key == "test_api_key_env"
        assert config.openrouter_model_name == "env_or_model"
        assert config.langsmith_enabled is False
        assert config.local_search_domains == ["http://site1.env", "http://site2.env"]
        assert config.search_mode == "local_only"
        assert config.number_of_initial_queries == 5

    def test_runnable_config_overrides_env(self):
        """Test that RunnableConfig values override environment variables."""
        os.environ["LLM_PROVIDER"] = "env_provider"
        os.environ["LLM_API_KEY"] = "env_key"

        run_config_values = {
            "llm_provider": "runnable_provider",
            "llm_api_key": "runnable_key",
            "langsmith_enabled": False,
        }
        config = Configuration.from_runnable_config(RunnableConfig(configurable=run_config_values))

        assert config.llm_provider == "runnable_provider"
        assert config.llm_api_key == "runnable_key"
        assert config.langsmith_enabled is False # Overrode default True

    def test_runnable_config_overrides_defaults(self):
        """Test that RunnableConfig values override defaults when no env var."""
        run_config_values = {
            "llm_provider": "runnable_provider_only",
            "number_of_initial_queries": 10,
        }
        config = Configuration.from_runnable_config(RunnableConfig(configurable=run_config_values))

        assert config.llm_provider == "runnable_provider_only"
        assert config.number_of_initial_queries == 10
        assert config.max_research_loops == 2 # Default

    def test_precedence_runnable_env_default(self):
        """Test RunnableConfig > Env Var > Default precedence for a field."""
        # Default is 3 for number_of_initial_queries
        os.environ["NUMBER_OF_INITIAL_QUERIES"] = "7" # Env var

        # 1. RunnableConfig has precedence
        run_config_values = {"number_of_initial_queries": 15}
        config = Configuration.from_runnable_config(RunnableConfig(configurable=run_config_values))
        assert config.number_of_initial_queries == 15

        # 2. Env var has precedence if not in RunnableConfig
        config_env = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config_env.number_of_initial_queries == 7

        # 3. Default is used if not in RunnableConfig or Env
        del os.environ["NUMBER_OF_INITIAL_QUERIES"]
        config_default = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config_default.number_of_initial_queries == 3


    def test_local_search_domains_parsing(self):
        """Test parsing of LOCAL_SEARCH_DOMAINS."""
        # Test with validator directly for focused test, or through Configuration load
        os.environ["LOCAL_SEARCH_DOMAINS"] = " http://domain1.com ,http://domain2.com  "
        config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config.local_search_domains == ["http://domain1.com", "http://domain2.com"]

        os.environ["LOCAL_SEARCH_DOMAINS"] = ""
        config_empty = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config_empty.local_search_domains == []

        os.environ["LOCAL_SEARCH_DOMAINS"] = "http://single.com"
        config_single = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config_single.local_search_domains == ["http://single.com"]

        del os.environ["LOCAL_SEARCH_DOMAINS"]
        config_none = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config_none.local_search_domains == [] # Default factory

    def test_boolean_parsing_from_env(self):
        """Test boolean fields are correctly parsed from string env vars."""
        os.environ["LANGSMITH_ENABLED"] = "false"
        os.environ["ENABLE_LOCAL_SEARCH"] = "true"
        config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config.langsmith_enabled is False
        assert config.enable_local_search is True

        os.environ["LANGSMITH_ENABLED"] = "0"
        os.environ["ENABLE_LOCAL_SEARCH"] = "1"
        config_numeric = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        assert config_numeric.langsmith_enabled is False
        assert config_numeric.enable_local_search is True

        # Pydantic generally handles "t", "f", "yes", "no" etc. too, but "true"/"false"/"0"/"1" are common.

    def test_instantiation_via_constructor_uses_env_and_defaults(self):
        """Test direct Configuration() instantiation primarily uses env vars and defaults."""
        os.environ["LLM_PROVIDER"] = "constructor_test_provider"
        os.environ["MAX_RESEARCH_LOOPS"] = "99"

        # Clear a variable that has a default to ensure default is picked if not in env
        if "SEARCH_MODE" in os.environ:
            del os.environ["SEARCH_MODE"]

        config = Configuration() # Not using from_runnable_config here

        assert config.llm_provider == "constructor_test_provider"
        assert config.max_research_loops == 99
        assert config.search_mode == "internet_only" # Default
        assert config.langsmith_enabled is True # Default, assuming LANGSMITH_ENABLED env var is not set by this test

    def test_runnable_config_only_partial(self):
        """Test when RunnableConfig provides only a subset of fields."""
        os.environ["LLM_API_KEY"] = "env_api_key_for_partial_test"
        # Default for langsmith_enabled is True

        run_config_values = {
            "llm_provider": "partial_provider_in_runnable",
            "langsmith_enabled": False # Override default and any env var
        }
        config = Configuration.from_runnable_config(RunnableConfig(configurable=run_config_values))

        assert config.llm_provider == "partial_provider_in_runnable"
        assert config.llm_api_key == "env_api_key_for_partial_test" # Picked from env
        assert config.langsmith_enabled is False # From RunnableConfig
        assert config.max_research_loops == 2 # Default


# To run these tests:
# Ensure pytest is installed: pip install pytest
# Navigate to the directory containing `test_configuration.py` (or its parent)
# Run: pytest
#
# Note: For tests involving environment variables, it's crucial to isolate them
# so that tests don't interfere with each other or the actual environment.
# The setup_method and teardown_class here handle this by clearing/restoring.
# Consider using pytest-env or monkeypatch for more robust env var manipulation in larger test suites.
