import os
from pydantic import BaseModel, Field, validator
from typing import Any, Optional, List

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    llm_provider: str = Field(
        default="gemini",
        metadata={
            "description": "The LLM provider to use (e.g., 'gemini', 'openrouter', 'deepseek'). Environment variable: LLM_PROVIDER"
        },
    )

    llm_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The API key for the selected LLM provider. Environment variable: LLM_API_KEY"
        },
    )

    openrouter_model_name: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The specific OpenRouter model string (e.g., 'anthropic/claude-3-haiku'). Environment variable: OPENROUTER_MODEL_NAME"
        },
    )

    deepseek_model_name: Optional[str] = Field(
        default=None,
        metadata={
            "description": "The specific DeepSeek model (e.g., 'deepseek-chat'). Environment variable: DEEPSEEK_MODEL_NAME"
        },
    )

    query_generator_model: str = Field(
        default="gemini-1.5-flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation. Interpreted based on llm_provider (e.g., 'gemini-1.5-flash' for Gemini, part of model string for OpenRouter). Environment variable: QUERY_GENERATOR_MODEL"
        },
    )

    reflection_model: str = Field(
        default="gemini-1.5-flash",
        metadata={
            "description": "The name of the language model to use for the agent's reflection. Interpreted based on llm_provider. Environment variable: REFLECTION_MODEL"
        },
    )

    answer_model: str = Field(
        default="gemini-1.5-pro",
        metadata={
            "description": "The name of the language model to use for the agent's answer. Interpreted based on llm_provider. Environment variable: ANSWER_MODEL"
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    langsmith_enabled: bool = Field(
        default=True,
        metadata={
            "description": "Controls LangSmith tracing. Set to false to disable. If true, ensure LANGCHAIN_API_KEY and other relevant LangSmith environment variables (LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT) are set. Environment variable: LANGSMITH_ENABLED"
        },
    )

    enable_local_search: bool = Field(
        default=False,
        metadata={
            "description": "Enable or disable local network search functionality. Environment variable: ENABLE_LOCAL_SEARCH"
        },
    )

    local_search_domains: List[str] = Field(
        default_factory=list, # Use default_factory for mutable types like list
        metadata={
            "description": "Comma-separated list of base URLs or domains for local network search (e.g., 'http://intranet.mycompany.com,http://docs.internal'). Environment variable: LOCAL_SEARCH_DOMAINS"
        },
    )

    search_mode: str = Field(
        default="internet_only",
        metadata={
            "description": "Search behavior: 'internet_only', 'local_only', 'internet_then_local', 'local_then_internet'. Environment variable: SEARCH_MODE"
        },
    )

    @validator("local_search_domains", pre=True, always=True)
    def parse_local_search_domains(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            if not v: # Handle empty string case
                return []
            return [domain.strip() for domain in v.split(',')]
        if v is None: # Handle None if default_factory is not triggered early enough by env var
            return []
        return v # Already a list or handled by Pydantic

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Define a helper to fetch values preferentially from environment, then config, then default
        def get_value(field_name: str, default_value: Any = None) -> Any:
            env_var_name = field_name.upper()
            # For model_fields that have metadata and description, we can try to get env var name from there
            # However, it's safer to rely on convention (field_name.upper())
            # or explicitly map them if names differ significantly.
            # For now, we'll stick to the convention.
            value = os.environ.get(env_var_name, configurable.get(field_name))
            if value is None:
                # Fallback to default if defined in Field
                field_info = cls.model_fields.get(field_name)
                if field_info and field_info.default is not None:
                    return field_info.default
                return default_value
            return value

        raw_values: dict[str, Any] = {
            name: get_value(name, cls.model_fields[name].default)
            for name in cls.model_fields.keys()
        }

        # Filter out None values for fields that are not explicitly Optional
        # and don't have a default value that is None.
        # Pydantic handles default values automatically, so this filtering might be redundant
        # if defaults are correctly set up in the model fields.
        # However, ensuring that we only pass values that are actually provided (env, config, or explicit default)
        # can prevent issues with Pydantic's validation if a field is not Optional but no value is found.

        values_to_pass = {}
        for name, field_info in cls.model_fields.items():
            val = raw_values.get(name)
            if val is not None:
                values_to_pass[name] = val
            # If val is None but the field has a default value (even if None),
            # Pydantic will handle it. If it's Optional, None is fine.
            # If it's required and None, Pydantic will raise an error, which is correct.

        return cls(**values_to_pass)
