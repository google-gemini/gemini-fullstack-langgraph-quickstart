import os
import json
import logging
import time
import random
from typing import List

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_aws import ChatBedrockConverse
from langchain_community.tools import BraveSearch
import boto3

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
from agent.utils import (
    get_research_topic,
    extract_text_from_content,
)

# Set up logging
logger = logging.getLogger(__name__)

load_dotenv()

if not all(
    k in os.environ
    for k in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "BRAVE_SEARCH_API_KEY",
    ]
):
    raise ValueError("Required environment variables are not properly set")


def get_bedrock_llm(model_id: str, temperature: float = 0.7) -> ChatBedrockConverse:
    """Initialize a Bedrock model with the specified configuration."""
    return ChatBedrockConverse(
        model=model_id,
        region_name=os.getenv("AWS_REGION"),
        temperature=temperature,
        max_tokens=2048,
    )


# Initialize BraveSearch without retry wrapper to avoid interface issues
search_tool = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_SEARCH_API_KEY"), search_kwargs={"count": 5}
)


def safe_search_with_retry(query: str, max_retries: int = 5) -> str:
    """Perform search with manual retry logic to avoid tool interface issues."""
    for attempt in range(max_retries):
        try:
            return search_tool.run(query)
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Search attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                raise e


def parse_structured_output(response_text, schema_type: str):
    """Safely parse structured output with fallbacks."""
    try:
        # First extract text content using our utility function
        text_content = extract_text_from_content(response_text)

        # Clean up the response text
        if "```json" in text_content:
            text_content = text_content.split("```json")[1].split("```")[0].strip()
        elif "```" in text_content:
            text_content = text_content.split("```")[1].split("```")[0].strip()

        return json.loads(text_content)
    except Exception as e:
        logger.warning(
            f"Failed to parse {schema_type} response: {str(e)} - Response type: {type(response_text)}"
        )
        return None


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses AWS Bedrock Claude to create optimized search queries for web research based on
    the User's question with structured output parsing.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    logger.info("Starting query generation")

    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    try:
        # Initialize the model
        llm = get_bedrock_llm(
            model_id=configurable.query_generator_model,
            temperature=1.0,
        )

        # Format the prompt
        current_date = get_current_date()
        research_topic = get_research_topic(state["messages"])

        formatted_prompt = query_writer_instructions.format(
            current_date=current_date,
            research_topic=research_topic,
            number_queries=state["initial_search_query_count"],
        )

        # Add structured output instructions to the prompt
        structured_prompt = f"""
{formatted_prompt}

Please respond with a JSON object containing:
- "query": array of search query strings
- "rationale": brief explanation of why these queries are relevant

Example format:
{{"query": ["query1", "query2"], "rationale": "explanation"}}
"""

        logger.info(f"Using model: {configurable.query_generator_model}")

        response = llm.invoke(structured_prompt)
        logger.info(
            f"Raw response type: {type(response.content)}, content: {str(response.content)[:200]}..."
        )
        parsed = parse_structured_output(response.content, "query generation")

        if parsed and "query" in parsed:
            logger.info(f"Generated queries: {parsed['query']}")
            return {"query_list": parsed["query"]}
        else:
            # Fallback
            logger.warning("Using fallback query generation")
            return {"query_list": [research_topic]}

    except Exception as e:
        logger.error(f"Error in generate_query: {str(e)}")
        # Fallback to a simple query
        research_topic = get_research_topic(state["messages"])
        return {"query_list": [research_topic]}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using Brave Search.

    Executes a web search using Brave Search API with built-in retry logic
    and processes results with Claude 3 Sonnet.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    logger.info(f"Starting web research for query: {state['search_query']}")

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Perform web search with manual retry logic
    try:
        search_results = safe_search_with_retry(state["search_query"])
        search_data = json.loads(search_results)
        logger.info(f"Found {len(search_data)} search results")
    except Exception as e:
        logger.error(f"Search failed for query '{state['search_query']}': {str(e)}")
        # Return minimal results if search fails completely
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [
                f"Search failed for query '{state['search_query']}': {str(e)}"
            ],
        }

    # Format search results for the model
    formatted_results = []
    sources_gathered = []

    for idx, result in enumerate(search_data):
        source = {
            "title": result["title"],
            "url": result["link"],
            "snippet": result["snippet"],
            "short_url": f"[{idx + 1}]",
        }
        sources_gathered.append(source)
        formatted_results.append(
            f"{source['short_url']}: {source['title']}\n{source['snippet']}\nURL: {source['url']}\n"
        )

    try:
        llm = get_bedrock_llm(
            model_id=configurable.query_generator_model,
            temperature=0.0,
        )

        formatted_prompt = web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=state["search_query"],
            search_results="\n\n".join(formatted_results),
        )

        response = llm.invoke(formatted_prompt)

        # Extract text content properly
        content = extract_text_from_content(response.content)

        return {
            "sources_gathered": sources_gathered,
            "search_query": [state["search_query"]],
            "web_research_result": [content],
        }

    except Exception as e:
        logger.error(f"Error processing web research results: {str(e)}")
        return {
            "sources_gathered": sources_gathered,
            "search_query": [state["search_query"]],
            "web_research_result": [f"Error processing results: {str(e)}"],
        }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries using structured output parsing.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    logger.info("Starting reflection")

    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reflection_model

    try:
        # Initialize the model
        llm = get_bedrock_llm(
            model_id=reasoning_model,
            temperature=1.0,
        )

        # Flatten web_research_result and ensure all items are strings
        summaries = []
        for item in state["web_research_result"]:
            if isinstance(item, list):
                # Recursively flatten nested lists
                for subitem in item:
                    if isinstance(subitem, str):
                        summaries.append(subitem)
                    elif isinstance(subitem, dict):
                        # Skip dictionaries (likely sources_gathered got mixed in)
                        logger.warning(
                            f"Skipping dict in web_research_result: {type(subitem)}"
                        )
                        continue
                    else:
                        summaries.append(str(subitem))
            elif isinstance(item, str):
                summaries.append(item)
            elif isinstance(item, dict):
                # Skip dictionaries (likely sources_gathered got mixed in)
                logger.warning(f"Skipping dict in web_research_result: {type(item)}")
                continue
            else:
                summaries.append(str(item))

        logger.info(f"Processed {len(summaries)} summaries for reflection")

        # Format the prompt
        current_date = get_current_date()
        base_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n\n---\n\n".join(summaries),
        )

        # Add structured output instructions
        structured_prompt = f"""
{base_prompt}

Please respond with a JSON object containing:
- "is_sufficient": boolean (true/false)
- "knowledge_gap": string describing missing information
- "follow_up_queries": array of follow-up query strings

Example format:
{{"is_sufficient": false, "knowledge_gap": "Missing recent data", "follow_up_queries": ["query1", "query2"]}}
"""

        response = llm.invoke(structured_prompt)
        logger.info(
            f"Raw reflection response type: {type(response.content)}, content: {str(response.content)[:200]}..."
        )
        parsed = parse_structured_output(response.content, "reflection")

        if parsed and "is_sufficient" in parsed:
            logger.info(
                f"Reflection result - sufficient: {parsed['is_sufficient']}, follow-up queries: {parsed.get('follow_up_queries', [])}"
            )
            return {
                "is_sufficient": parsed["is_sufficient"],
                "knowledge_gap": parsed.get("knowledge_gap", ""),
                "follow_up_queries": parsed.get("follow_up_queries", []),
                "research_loop_count": state["research_loop_count"],
                "number_of_ran_queries": len(state["search_query"]),
            }
        else:
            # Fallback
            logger.warning("Using fallback reflection")
            return {
                "is_sufficient": state["research_loop_count"] >= 2,
                "knowledge_gap": "Fallback: research completed",
                "follow_up_queries": [],
                "research_loop_count": state["research_loop_count"],
                "number_of_ran_queries": len(state["search_query"]),
            }

    except Exception as e:
        logger.error(f"Error in reflection: {str(e)}")
        # Fallback to simple values
        research_topic = get_research_topic(state["messages"])
        return {
            "is_sufficient": state["research_loop_count"]
            >= 2,  # Default to sufficient after 2 loops
            "knowledge_gap": "Error in reflection process.",
            "follow_up_queries": [research_topic]
            if state["research_loop_count"] < 2
            else [],
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

    logger.info(
        f"Evaluating research: sufficient={state['is_sufficient']}, loop={state['research_loop_count']}, max={max_research_loops}"
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
    logger.info("Finalizing answer")

    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Flatten web_research_result and ensure all items are strings
    summaries = []
    for item in state["web_research_result"]:
        if isinstance(item, list):
            # Recursively flatten nested lists
            for subitem in item:
                if isinstance(subitem, str):
                    summaries.append(subitem)
                elif isinstance(subitem, dict):
                    # Skip dictionaries (likely sources_gathered got mixed in)
                    logger.warning(
                        f"Skipping dict in web_research_result: {type(subitem)}"
                    )
                    continue
                else:
                    # Extract text content from structured format
                    extracted = extract_text_from_content(subitem)
                    summaries.append(extracted)
        elif isinstance(item, str):
            summaries.append(item)
        elif isinstance(item, dict):
            # Skip dictionaries (likely sources_gathered got mixed in)
            logger.warning(f"Skipping dict in web_research_result: {type(item)}")
            continue
        else:
            # Extract text content from structured format
            extracted = extract_text_from_content(item)
            summaries.append(extracted)

    logger.info(f"Processed {len(summaries)} summaries for final answer")
    logger.info(
        f"Sample summary content: {summaries[0][:100] if summaries else 'No summaries'}..."
    )

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(summaries),
    )

    try:
        # init Claude 3 Haiku
        llm = get_bedrock_llm(
            model_id=reasoning_model,
            temperature=0.0,
        )
        result = llm.invoke(formatted_prompt)

        # Extract text content properly
        content = extract_text_from_content(result.content)

        # Replace the short urls with the original urls and add all used urls to the sources_gathered
        unique_sources = []
        for source in state["sources_gathered"]:
            if source["short_url"] in content:
                content = content.replace(source["short_url"], source["url"])
                unique_sources.append(source)

        logger.info(f"Generated final answer with {len(unique_sources)} sources")

        return {
            "messages": [AIMessage(content=content)],
            "sources_gathered": unique_sources,
        }

    except Exception as e:
        logger.error(f"Error in finalize_answer: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Error generating final answer: {str(e)}")],
            "sources_gathered": [],
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

graph = builder.compile(name="aws-bedrock-search-agent")
