import os
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import time
from functools import wraps

from tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from google.api_core import exceptions as google_exceptions
from langchain_core.exceptions import LangChainException

from state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from configuration import Configuration
from prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ErrorType(Enum):
    """Enumeration of different error types that can occur in the research agent."""
    API_FAILURE = "api_failure"
    NETWORK_TIMEOUT = "network_timeout"
    INVALID_RESPONSE = "invalid_response"
    MALFORMED_OUTPUT = "malformed_output"
    CONFIGURATION_ERROR = "configuration_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class AgentError:
    """Structured error information for the research agent."""
    error_type: ErrorType
    message: str
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = True


@dataclass
class APIConfig:
    """Configuration for API settings with defaults."""
    # LLM Configuration
    query_generator_temperature: float = 1.0
    reasoning_temperature: float = 0.0
    max_retries: int = 3
    request_timeout: int = 30
    
    # Research Configuration
    max_research_loops: int = 3
    initial_search_query_count: int = 3
    
    # Rate limiting
    api_rate_limit_delay: float = 1.0
    max_concurrent_requests: int = 5
    
    # Fallback models
    fallback_query_model: str = "gemini-1.5-flash"
    fallback_reasoning_model: str = "gemini-1.5-pro"


def handle_api_errors(func):
    """Decorator to handle common API errors with retry logic."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = APIConfig().max_retries
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except google_exceptions.GoogleAPIError as e:
                error = AgentError(
                    error_type=ErrorType.API_FAILURE,
                    message=f"Google API error: {str(e)}",
                    details={"attempt": attempt + 1, "max_retries": max_retries}
                )
                logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"Max retries exceeded for {func.__name__}")
                    raise APIException(error)
                
                # Exponential backoff
                time.sleep(2 ** attempt)
                
            except Exception as e:
                error = AgentError(
                    error_type=ErrorType.API_FAILURE,
                    message=f"Unexpected error in {func.__name__}: {str(e)}",
                    details={"function": func.__name__}
                )
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise APIException(error)
    
    return wrapper


class APIException(Exception):
    """Custom exception for API-related errors."""
    def __init__(self, agent_error: AgentError):
        self.agent_error = agent_error
        super().__init__(agent_error.message)


class ValidationException(Exception):
    """Custom exception for validation errors."""
    def __init__(self, agent_error: AgentError):
        self.agent_error = agent_error
        super().__init__(agent_error.message)


def validate_environment():
    """Validate required environment variables and configuration."""
    if os.getenv("GEMINI_API_KEY") is None:
        raise ValidationException(AgentError(
            error_type=ErrorType.CONFIGURATION_ERROR,
            message="GEMINI_API_KEY environment variable is not set",
            recoverable=False
        ))


def validate_state_input(state: Dict[str, Any], required_keys: List[str], node_name: str):
    """Validate that required state keys are present and valid."""
    for key in required_keys:
        if not state.get(key):
            raise ValidationException(AgentError(
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Missing required state key '{key}' in node '{node_name}'",
                details={"node": node_name, "missing_key": key}
            ))


def create_llm_with_fallback(
    primary_model: str, 
    fallback_model: str, 
    temperature: float, 
    api_config: APIConfig
) -> ChatGoogleGenerativeAI:
    """Create LLM instance with fallback model support."""
    try:
        return ChatGoogleGenerativeAI(
            model=primary_model,
            temperature=temperature,
            max_retries=api_config.max_retries,
            timeout=api_config.request_timeout,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    except Exception as e:
        logger.warning(f"Failed to initialize primary model {primary_model}, falling back to {fallback_model}")
        return ChatGoogleGenerativeAI(
            model=fallback_model,
            temperature=temperature,
            max_retries=api_config.max_retries,
            timeout=api_config.request_timeout,
            api_key=os.getenv("GEMINI_API_KEY"),
        )


# Initialize with validation
validate_environment()
api_config = APIConfig()

# Used for Google Search API
try:
    genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize Google GenAI client: {e}")
    raise ValidationException(AgentError(
        error_type=ErrorType.CONFIGURATION_ERROR,
        message=f"Failed to initialize Google GenAI client: {str(e)}",
        recoverable=False
    ))


# Nodes
@handle_api_errors
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini to create optimized search queries for web research based on
    the User's question with comprehensive error handling and validation.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
        
    Raises:
        ValidationException: If required state keys are missing
        APIException: If API calls fail after retries
    """
    try:
        logger.info("Starting query generation")
        
        # Validate inputs
        validate_state_input(state, ["messages"], "generate_query")
        
        configurable = Configuration.from_runnable_config(config)

        # Set defaults with validation
        if state.get("initial_search_query_count") is None:
            state["initial_search_query_count"] = (
                configurable.number_of_initial_queries 
                if hasattr(configurable, 'number_of_initial_queries')
                else api_config.initial_search_query_count
            )

        # Initialize LLM with fallback
        llm = create_llm_with_fallback(
            primary_model=configurable.query_generator_model,
            fallback_model=api_config.fallback_query_model,
            temperature=api_config.query_generator_temperature,
            api_config=api_config
        )
        
        try:
            structured_llm = llm.with_structured_output(SearchQueryList)
        except Exception as e:
            raise APIException(AgentError(
                error_type=ErrorType.MALFORMED_OUTPUT,
                message=f"Failed to create structured output: {str(e)}"
            ))

        # Format the prompt with validation
        try:
            current_date = get_current_date()
            research_topic = get_research_topic(state["messages"])
            
            if not research_topic:
                raise ValidationException(AgentError(
                    error_type=ErrorType.VALIDATION_ERROR,
                    message="Could not extract research topic from messages"
                ))
                
            formatted_prompt = query_writer_instructions.format(
                current_date=current_date,
                research_topic=research_topic,
                number_queries=state["initial_search_query_count"],
            )
        except Exception as e:
            raise ValidationException(AgentError(
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Failed to format prompt: {str(e)}"
            ))

        # Generate the search queries with timeout
        try:
            result = structured_llm.invoke(formatted_prompt)
            
            # Validate the result
            if not hasattr(result, 'query') or not result.query:
                raise APIException(AgentError(
                    error_type=ErrorType.MALFORMED_OUTPUT,
                    message="LLM returned empty or malformed query list"
                ))
                
            logger.info(f"Generated {len(result.query)} search queries")
            return {"query_list": result.query}
            
        except LangChainException as e:
            raise APIException(AgentError(
                error_type=ErrorType.API_FAILURE,
                message=f"LangChain error during query generation: {str(e)}"
            ))
            
    except (ValidationException, APIException):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_query: {e}")
        raise APIException(AgentError(
            error_type=ErrorType.API_FAILURE,
            message=f"Unexpected error during query generation: {str(e)}"
        ))


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    try:
        validate_state_input(state, ["query_list"], "continue_to_web_research")
        
        if not isinstance(state["query_list"], list) or len(state["query_list"]) == 0:
            logger.warning("No valid queries to process")
            return []
            
        logger.info(f"Dispatching {len(state['query_list'])} web research tasks")
        
        return [
            Send("web_research", {"search_query": search_query, "id": int(idx)})
            for idx, search_query in enumerate(state["query_list"])
            if search_query and search_query.strip()  # Filter out empty queries
        ]
    except Exception as e:
        logger.error(f"Error in continue_to_web_research: {e}")
        return []


@handle_api_errors
def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini
    with comprehensive error handling.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
        
    Raises:
        ValidationException: If required state keys are missing
        APIException: If API calls fail after retries
    """
    try:
        logger.info(f"Starting web research for query: {state.get('search_query', 'Unknown')}")
        
        # Validate inputs
        validate_state_input(state, ["search_query"], "web_research")
        
        configurable = Configuration.from_runnable_config(config)
        
        # Format prompt with validation
        try:
            current_date = get_current_date()
            formatted_prompt = web_searcher_instructions.format(
                current_date=current_date,
                research_topic=state["search_query"],
            )
        except Exception as e:
            raise ValidationException(AgentError(
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Failed to format web search prompt: {str(e)}"
            ))

        # Rate limiting
        time.sleep(api_config.api_rate_limit_delay)

        # Perform search with timeout and error handling
        try:
            response = genai_client.models.generate_content(
                model=configurable.query_generator_model,
                contents=formatted_prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "temperature": api_config.reasoning_temperature,
                },
                request_options={
                    "timeout": api_config.request_timeout
                }
            )
            
            # Validate response
            if not response or not hasattr(response, 'candidates') or not response.candidates:
                raise APIException(AgentError(
                    error_type=ErrorType.INVALID_RESPONSE,
                    message="Empty or invalid response from Google Search API"
                ))
                
            candidate = response.candidates[0]
            if not hasattr(candidate, 'grounding_metadata'):
                logger.warning("No grounding metadata in response")
                grounding_chunks = []
            else:
                grounding_chunks = candidate.grounding_metadata.grounding_chunks
                
        except google_exceptions.DeadlineExceeded:
            raise APIException(AgentError(
                error_type=ErrorType.NETWORK_TIMEOUT,
                message="Google Search API request timed out"
            ))
        except google_exceptions.GoogleAPIError as e:
            raise APIException(AgentError(
                error_type=ErrorType.API_FAILURE,
                message=f"Google Search API error: {str(e)}"
            ))

        # Process results with error handling
        try:
            resolved_urls = resolve_urls(grounding_chunks, state.get("id", 0))
            citations = get_citations(response, resolved_urls)
            modified_text = insert_citation_markers(response.text, citations)
            sources_gathered = [item for citation in citations for item in citation["segments"]]
            
            logger.info(f"Web research completed. Found {len(sources_gathered)} sources")
            
            return {
                "sources_gathered": sources_gathered,
                "search_query": [state["search_query"]],
                "web_research_result": [modified_text],
            }
            
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            # Return partial results rather than failing completely
            return {
                "sources_gathered": [],
                "search_query": [state["search_query"]],
                "web_research_result": [response.text if response and hasattr(response, 'text') else ""],
            }
            
    except (ValidationException, APIException):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in web_research: {e}")
        raise APIException(AgentError(
            error_type=ErrorType.API_FAILURE,
            message=f"Unexpected error during web research: {str(e)}"
        ))


@handle_api_errors
def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries with comprehensive error handling.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
        
    Raises:
        ValidationException: If required state keys are missing
        APIException: If API calls fail after retries
    """
    try:
        logger.info("Starting reflection phase")
        
        # Validate inputs
        validate_state_input(state, ["web_research_result", "messages"], "reflection")
        
        configurable = Configuration.from_runnable_config(config)
        
        # Increment the research loop count safely
        state["research_loop_count"] = state.get("research_loop_count", 0) + 1
        reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

        # Format the prompt with validation
        try:
            current_date = get_current_date()
            research_topic = get_research_topic(state["messages"])
            
            if not research_topic:
                raise ValidationException(AgentError(
                    error_type=ErrorType.VALIDATION_ERROR,
                    message="Could not extract research topic for reflection"
                ))
                
            summaries = state.get("web_research_result", [])
            if not summaries:
                logger.warning("No web research results available for reflection")
                summaries = ["No research results available"]
                
            formatted_prompt = reflection_instructions.format(
                current_date=current_date,
                research_topic=research_topic,
                summaries="\n\n---\n\n".join(summaries),
            )
        except Exception as e:
            raise ValidationException(AgentError(
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Failed to format reflection prompt: {str(e)}"
            ))

        # Initialize LLM with fallback
        llm = create_llm_with_fallback(
            primary_model=reasoning_model,
            fallback_model=api_config.fallback_reasoning_model,
            temperature=api_config.query_generator_temperature,
            api_config=api_config
        )
        
        try:
            structured_llm = llm.with_structured_output(Reflection)
            result = structured_llm.invoke(formatted_prompt)
            
            # Validate structured output
            if not hasattr(result, 'is_sufficient'):
                raise APIException(AgentError(
                    error_type=ErrorType.MALFORMED_OUTPUT,
                    message="Reflection result missing 'is_sufficient' field"
                ))
                
            logger.info(f"Reflection complete. Research sufficient: {result.is_sufficient}")
            
            return {
                "is_sufficient": result.is_sufficient,
                "knowledge_gap": getattr(result, 'knowledge_gap', ''),
                "follow_up_queries": getattr(result, 'follow_up_queries', []),
                "research_loop_count": state["research_loop_count"],
                "number_of_ran_queries": len(state.get("search_query", [])),
            }
            
        except LangChainException as e:
            raise APIException(AgentError(
                error_type=ErrorType.API_FAILURE,
                message=f"LangChain error during reflection: {str(e)}"
            ))
            
    except (ValidationException, APIException):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in reflection: {e}")
        raise APIException(AgentError(
            error_type=ErrorType.API_FAILURE,
            message=f"Unexpected error during reflection: {str(e)}"
        ))


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> Union[str, List[Send]]:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("finalize_answer") or list of Send objects
    """
    try:
        logger.info("Evaluating research progress")
        
        configurable = Configuration.from_runnable_config(config)
        max_research_loops = (
            state.get("max_research_loops")
            if state.get("max_research_loops") is not None
            else getattr(configurable, 'max_research_loops', api_config.max_research_loops)
        )
        
        current_loop = state.get("research_loop_count", 0)
        is_sufficient = state.get("is_sufficient", False)
        
        logger.info(f"Research loop {current_loop}/{max_research_loops}, sufficient: {is_sufficient}")
        
        if is_sufficient or current_loop >= max_research_loops:
            logger.info("Research complete, finalizing answer")
            return "finalize_answer"
        else:
            follow_up_queries = state.get("follow_up_queries", [])
            if not follow_up_queries:
                logger.warning("No follow-up queries generated, finalizing answer")
                return "finalize_answer"
                
            number_of_ran_queries = state.get("number_of_ran_queries", 0)
            
            logger.info(f"Continuing research with {len(follow_up_queries)} follow-up queries")
            
            return [
                Send(
                    "web_research",
                    {
                        "search_query": follow_up_query,
                        "id": number_of_ran_queries + int(idx),
                    },
                )
                for idx, follow_up_query in enumerate(follow_up_queries)
                if follow_up_query and follow_up_query.strip()  # Filter empty queries
            ]
            
    except Exception as e:
        logger.error(f"Error in evaluate_research: {e}")
        # Default to finalize_answer on error
        return "finalize_answer"


@handle_api_errors
def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations and comprehensive error handling.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
        
    Raises:
        ValidationException: If required state keys are missing
        APIException: If API calls fail after retries
    """
    try:
        logger.info("Finalizing answer")
        
        # Validate inputs
        validate_state_input(state, ["messages"], "finalize_answer")
        
        configurable = Configuration.from_runnable_config(config)
        reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

        # Handle case where no research results are available
        summaries = state.get("web_research_result", [])
        if not summaries:
            logger.warning("No research results available for final answer")
            return {
                "messages": [AIMessage(content="I apologize, but I was unable to gather sufficient research information to provide a comprehensive answer.")],
                "sources_gathered": [],
            }

        # Format the prompt with validation
        try:
            current_date = get_current_date()
            research_topic = get_research_topic(state["messages"])
            
            if not research_topic:
                raise ValidationException(AgentError(
                    error_type=ErrorType.VALIDATION_ERROR,
                    message="Could not extract research topic for final answer"
                ))
                
            formatted_prompt = answer_instructions.format(
                current_date=current_date,
                research_topic=research_topic,
                summaries="\n---\n\n".join(summaries),
            )
        except Exception as e:
            raise ValidationException(AgentError(
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Failed to format final answer prompt: {str(e)}"
            ))

        # Initialize LLM with fallback
        llm = create_llm_with_fallback(
            primary_model=reasoning_model,
            fallback_model=api_config.fallback_reasoning_model,
            temperature=api_config.reasoning_temperature,
            api_config=api_config
        )
        
        try:
            result = llm.invoke(formatted_prompt)
            
            if not result or not hasattr(result, 'content') or not result.content:
                raise APIException(AgentError(
                    error_type=ErrorType.INVALID_RESPONSE,
                    message="Empty response from LLM during final answer generation"
                ))
                
        except LangChainException as e:
            raise APIException(AgentError(
                error_type=ErrorType.API_FAILURE,
                message=f"LangChain error during final answer generation: {str(e)}"
            ))

        # Process sources and replace URLs with error handling
        try:
            unique_sources = []
            sources_gathered = state.get("sources_gathered", [])
            
            for source in sources_gathered:
                if not isinstance(source, dict) or "short_url" not in source or "value" not in source:
                    logger.warning(f"Invalid source format: {source}")
                    continue
                    
                if source["short_url"] in result.content:
                    result.content = result.content.replace(
                        source["short_url"], source["value"]
                    )
                    unique_sources.append(source)
                    
            logger.info(f"Final answer generated with {len(unique_sources)} sources")
            
            return {
                "messages": [AIMessage(content=result.content)],
                "sources_gathered": unique_sources,
            }
            
        except Exception as e:
            logger.error(f"Error processing sources in final answer: {e}")
            # Return answer without source processing rather than failing
            return {
                "messages": [AIMessage(content=result.content)],
                "sources_gathered": state.get("sources_gathered", []),
            }
            
    except (ValidationException, APIException):
        raise
    except Exception as e:
        logger.error(f"Unexpected error in finalize_answer: {e}")
        raise APIException(AgentError(
            error_type=ErrorType.API_FAILURE,
            message=f"Unexpected error during answer finalization: {str(e)}"
        ))


# Create our Agent Graph with error handling
try:
    builder = StateGraph(OverallState, config_schema=Configuration)

    # Define the nodes we will cycle between
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)

    # Set the entrypoint as `generate_query`
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
    logger.info("Research agent graph compiled successfully")
    
except Exception as e:
    logger.error(f"Failed to compile research agent graph: {e}")
    raise ValidationException(AgentError(
        error_type=ErrorType.CONFIGURATION_ERROR,
        message=f"Failed to compile research agent graph: {str(e)}",
        recoverable=False
    ))
