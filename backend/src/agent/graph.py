import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # For OpenRouter and potentially DeepSeek if OpenAI compatible
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.tools_and_schemas import LocalSearchTool # Import the new tool

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# --- LangSmith Tracing Configuration ---
# Instantiate Configuration to read environment variables for global settings like LangSmith.
# Note: Configuration.from_runnable_config() is for node-specific configs within the graph.
global_config = Configuration()

if global_config.langsmith_enabled:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # LANGCHAIN_API_KEY, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT should be set by the user in their environment.
    # We can add a check here if LANGCHAIN_API_KEY is not set and log a warning.
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Warning: LangSmith is enabled, but LANGCHAIN_API_KEY is not set. Tracing will likely fail.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # Explicitly unset other LangSmith variables to prevent accidental tracing
    langsmith_vars_to_unset = ["LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]
    for var in langsmith_vars_to_unset:
        if var in os.environ:
            del os.environ[var]

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# Instantiate LocalSearchTool
local_search_tool = LocalSearchTool()

# Helper function to get LLM client based on configuration
def _get_llm_client(configurable: Configuration, task_model_name: str, temperature: float = 0.0, max_retries: int = 2) -> BaseChatModel:
    """
    Instantiates and returns an LLM client based on the provider specified in the configuration.

    Args:
        configurable: The Configuration object.
        task_model_name: The specific model name for the task (e.g., query_generator_model).
        temperature: The temperature for the LLM.
        max_retries: The maximum number of retries for API calls.

    Returns:
        An instance of a Langchain chat model.

    Raises:
        ValueError: If the LLM provider is unsupported or required keys/names are missing.
    """
    provider = configurable.llm_provider.lower()
    api_key = configurable.llm_api_key

    if provider == "gemini":
        gemini_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set for Gemini provider, either via LLM_API_KEY or GEMINI_API_KEY environment variable.")
        return ChatGoogleGenerativeAI(
            model=task_model_name,
            temperature=temperature,
            max_retries=max_retries,
            api_key=gemini_api_key,
        )
    elif provider == "openrouter":
        if not api_key:
            raise ValueError("LLM_API_KEY must be set for OpenRouter provider.")
        if not configurable.openrouter_model_name:
            # Using task_model_name as the full OpenRouter model string if openrouter_model_name is not set
            # This assumes task_model_name (e.g. query_generator_model) would contain "anthropic/claude-3-haiku"
            model_to_use = task_model_name
        else:
            # If openrouter_model_name is set, it's the primary model identifier.
            # Task-specific models might be appended or it might be a single model for all tasks.
            # For now, let's assume openrouter_model_name is the one to use if provided,
            # otherwise, the specific task_model_name acts as the full OpenRouter model string.
            model_to_use = configurable.openrouter_model_name

        return ChatOpenAI(
            model_name=model_to_use,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_retries=max_retries,
        )
    elif provider == "deepseek":
        if not api_key:
            raise ValueError("LLM_API_KEY must be set for DeepSeek provider.")
        # Assuming DeepSeek is OpenAI API compatible
        # Users should set configurable.deepseek_model_name to "deepseek-chat" or "deepseek-coder" etc.
        model_to_use = configurable.deepseek_model_name or task_model_name
        if not model_to_use:
             raise ValueError("deepseek_model_name or a task-specific model must be provided for DeepSeek.")

        return ChatOpenAI(
            model_name=model_to_use,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1", # Common DeepSeek API base
            temperature=temperature,
            max_retries=max_retries,
        )
    # Add other providers here as elif blocks
    # elif provider == "another_provider":
    #     return AnotherProviderChatModel(...)
    else:
        raise ValueError(f"Unsupported LLM provider: {configurable.llm_provider}")


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = _get_llm_client(configurable, configurable.query_generator_model, temperature=1.0)
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]

# --- Helper functions for web_research node ---

def _perform_google_search(state: WebSearchState, configurable: Configuration, current_genai_client: Client) -> tuple[list, list]:
    """Performs Google search and returns sources and results."""
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    try:
        response = current_genai_client.models.generate_content(
            model=configurable.query_generator_model, # This model is for the Google Search "agent"
            contents=formatted_prompt,
            config={
                "tools": [{"google_search": {}}], # Native Google Search tool
                "temperature": 0,
            },
        )
        if not response.candidates or not response.candidates[0].grounding_metadata:
            print(f"Google Search for '{state['search_query']}' returned no results or grounding metadata.")
            return [], []

        resolved_urls = resolve_urls(
            response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
        )
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources = [item for citation_group in citations for item in citation_group["segments"]]
        return sources, [modified_text]
    except Exception as e:
        print(f"Error during Google Search for query '{state['search_query']}': {e}")
        return [], []


def _perform_local_search(state: WebSearchState, configurable: Configuration, tool: LocalSearchTool) -> tuple[list, list]:
    """Performs local search and returns sources and results."""
    if not configurable.enable_local_search or not configurable.local_search_domains:
        return [], []

    search_query = state["search_query"]
    print(f"Performing local search for: {search_query} in domains: {configurable.local_search_domains}")
    try:
        local_results = tool._run(query=search_query, local_domains=configurable.local_search_domains)

        sources: list = []
        research_texts: list = []

        for idx, res in enumerate(local_results.results):
            source_id = f"local_{state['id']}_{idx}" # Create a unique enough ID
            source_dict = {
                "id": source_id,
                "value": res.url,
                "short_url": res.url, # For local, short_url is same as full url
                "title": res.title,
                "source_type": "local",
                 # Adapt snippet to fit the 'segments' structure if needed by downstream tasks,
                 # or ensure downstream tasks can handle this simpler structure.
                 # For now, keeping it simpler for finalize_answer compatibility:
                "segments": [{'segment_id': '0', 'text': res.snippet}]
            }
            sources.append(source_dict)
            research_texts.append(f"[LOCAL] {res.title}: {res.snippet} (Source: {res.url})")

        return sources, research_texts
    except Exception as e:
        print(f"Error during local search for query '{search_query}': {e}")
        return [], []

# --- End of helper functions ---


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    LangGraph node that performs web research based on the search_mode configuration.
    It can perform Google search, local network search, or a combination of both.
    """
    configurable = Configuration.from_runnable_config(config)
    search_query = state["search_query"] # Each invocation of this node gets one query.

    all_sources_gathered: list = []
    all_web_research_results: list = []

    search_mode = configurable.search_mode.lower()

    print(f"Web research for '{search_query}': Mode - {search_mode}, Local Search Enabled: {configurable.enable_local_search}")

    if search_mode == "internet_only":
        gs_sources, gs_results = _perform_google_search(state, configurable, genai_client)
        all_sources_gathered.extend(gs_sources)
        all_web_research_results.extend(gs_results)

    elif search_mode == "local_only":
        if configurable.enable_local_search and configurable.local_search_domains:
            ls_sources, ls_results = _perform_local_search(state, configurable, local_search_tool)
            all_sources_gathered.extend(ls_sources)
            all_web_research_results.extend(ls_results)
        else:
            print(f"Local search only mode, but local search is not enabled or no domains configured for query: {search_query}")
            all_web_research_results.append(f"No local results found for '{search_query}' as local search is not configured.")


    elif search_mode == "internet_then_local":
        gs_sources, gs_results = _perform_google_search(state, configurable, genai_client)
        all_sources_gathered.extend(gs_sources)
        all_web_research_results.extend(gs_results)
        if configurable.enable_local_search and configurable.local_search_domains:
            ls_sources, ls_results = _perform_local_search(state, configurable, local_search_tool)
            all_sources_gathered.extend(ls_sources)
            all_web_research_results.extend(ls_results)

    elif search_mode == "local_then_internet":
        if configurable.enable_local_search and configurable.local_search_domains:
            ls_sources, ls_results = _perform_local_search(state, configurable, local_search_tool)
            all_sources_gathered.extend(ls_sources)
            all_web_research_results.extend(ls_results)
        gs_sources, gs_results = _perform_google_search(state, configurable, genai_client)
        all_sources_gathered.extend(gs_sources)
        all_web_research_results.extend(gs_results)

    else: # Default to internet_only if mode is unknown
        print(f"Unknown search mode '{search_mode}', defaulting to internet_only for query: {search_query}")
        gs_sources, gs_results = _perform_google_search(state, configurable, genai_client)
        all_sources_gathered.extend(gs_sources)
        all_web_research_results.extend(gs_results)

    if not all_web_research_results: # Ensure there's always some text result
        all_web_research_results.append(f"No results found for query: '{search_query}' in mode '{search_mode}'.")

    return {
        "sources_gathered": all_sources_gathered,
        "search_query": [search_query], # Keep as list to match OverallState type
        "web_research_result": all_web_research_results,
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = _get_llm_client(configurable, configurable.reflection_model, temperature=1.0)
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    # The 'reasoning_model' from state is deprecated by specific model fields in Configuration
    # We now use configurable.answer_model for this node.

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = _get_llm_client(configurable, configurable.answer_model, temperature=0.0)
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
