import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

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
from langchain_community.chat_models.litellm import ChatLiteLLM
from firecrawl import FirecrawlApp
from agent.utils import (
    get_research_topic,
    # get_citations, # Commented out as per plan
    # insert_citation_markers, # Commented out as per plan
    # resolve_urls, # Commented out as per plan
)

load_dotenv()

# Environment variable checks
required_env_vars = {
    "ANTHROPIC_API_KEY": "Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.",
    "FIRECRAWL_API_KEY": "Firecrawl API key not found. Please set the FIRECRAWL_API_KEY environment variable.",
    "ANTHROPIC_API_BASE": "Anthropic API base URL (ANTHROPIC_API_BASE) not found. Set this if you're using a proxy or custom Anthropic API endpoint.",
}

missing_vars = [var for var, msg in required_env_vars.items() if not os.getenv(var)]
if missing_vars:
    error_messages = "\n".join([required_env_vars[var] for var in missing_vars])
    raise ValueError(f"Missing environment variables:\n{error_messages}")


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatLiteLLM(model=configurable.query_generator_model, temperature=1.0)
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


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Instantiate LLM for processing (remains the same)
    llm = ChatLiteLLM(model=configurable.query_generator_model, temperature=0)

    # Instantiate FirecrawlApp
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    # Execute Firecrawl search
    firecrawl_search_results = app.search(query=state["search_query"], params={'pageOptions': {'fetchPageContent': True}})

    # Process Firecrawl's output
    processed_text_parts = []
    sources_gathered_fc = []
    if firecrawl_search_results:
        for item in firecrawl_search_results:
            if item.get('markdown'):
                processed_text_parts.append(f"Source URL: {item.get('url', 'N/A')}\nContent:\n{item['markdown']}")
            if item.get('url'):
                # Ensure 'id' is present and unique if needed by downstream processes.
                # For now, using URL as a stand-in for a unique ID if state['id'] is not directly applicable here.
                # Or, we can construct an ID based on the index or a hash of the URL.
                # Let's try to keep it consistent with how 'id' was potentially used.
                # If state["id"] was per-query, and now we have multiple results for one query, this needs care.
                # For simplicity, let's make a generic source entry.
                sources_gathered_fc.append({"value": item['url'], "short_url": item['url'], "id": item.get('url')})
    else:
        processed_text_parts.append("No search results found.")

    search_results_content = "\n\n---\n\n".join(processed_text_parts)

    # Prepare prompt for processing search results
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    processing_prompt = f"{formatted_prompt}\n\nHere are the search results:\n{search_results_content}\n\nPlease summarize these results and extract key information related to the research topic."

    # Process search results with LLM
    processed_text_response = llm.invoke(processing_prompt)
    processed_text = processed_text_response.content

    # # Old sources_gathered logic is replaced by sources_gathered_fc
    # sources_gathered = [
    #     {"value": state["search_query"], "short_url": state["search_query"], "id": str(state["id"])}
    # ]

    return {
        "sources_gathered": sources_gathered_fc,
        "search_query": [state["search_query"]], # This likely remains the original query string
        "web_research_result": [processed_text],
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
    llm = ChatLiteLLM(model=reasoning_model, temperature=1.0)
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
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatLiteLLM(model=reasoning_model, temperature=0)
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
