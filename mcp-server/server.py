#!/usr/bin/env python3
"""
Gemini Research Agent MCP Server

This MCP server provides comprehensive research capabilities using Google's Gemini models
with tiered effort levels, advanced web research functionality, and proper citation tracking.

Features:
- Multi-tier research efforts (low: 10 searches, medium: 100, high: 1000)
- Web research with Google Search API integration using gemini-2.5-flash-preview-05-20
- Citation tracking and source validation
- Comprehensive answer generation with proper sourcing
- Industry-standard code quality and error handling
- Async support for optimal performance
- Detailed logging and monitoring

Author: Open Source Project
License: MIT
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from google.genai import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import json
import re
from urllib.parse import urlparse
import hashlib

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP is required. Install with: pip install mcp"
    )

# Load environment variables
load_dotenv()

# Configure logging with proper formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Validate environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is required. "
        "Please set it in your .env file or environment."
    )

# Model configuration - Using the specified model
RESEARCH_MODEL = "gemini-2.5-flash-preview-05-20"
QUERY_MODEL = "gemini-2.0-flash"  # For query generation
REFLECTION_MODEL = "gemini-2.5-flash-preview-04-17"  # For reflection
ANSWER_MODEL = "gemini-2.5-pro-preview-05-06"  # For final answer

# Effort tier configurations with clear limits
EFFORT_TIERS = {
    "low": {
        "max_searches": 10,
        "max_research_loops": 1,
        "initial_queries": 2,
        "description": "Quick research with up to 10 searches and 1 research loop"
    },
    "medium": {
        "max_searches": 100,
        "max_research_loops": 3,
        "initial_queries": 4,
        "description": "Balanced research with up to 100 searches and 3 research loops"
    },
    "high": {
        "max_searches": 1000,
        "max_research_loops": 5,
        "initial_queries": 6,
        "description": "Comprehensive research with up to 1000 searches and 5 research loops"
    }
}

# Initialize Google GenAI client with error handling
try:
    genai_client = Client(api_key=GEMINI_API_KEY)
    logger.info("Successfully initialized Google GenAI client")
except Exception as e:
    logger.error(f"Failed to initialize Google GenAI client: {e}")
    raise


# Pydantic models for structured data
class SearchQuery(BaseModel):
    """Individual search query with rationale and metadata."""
    query: str = Field(description="The search query string", min_length=1, max_length=500)
    rationale: str = Field(description="Explanation for this query", min_length=1)
    
    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not empty and reasonable length."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchQueryList(BaseModel):
    """List of search queries with overall rationale."""
    queries: List[SearchQuery] = Field(
        description="List of search queries", 
        min_items=1, 
        max_items=10
    )
    rationale: str = Field(description="Overall rationale for these queries")


class ResearchReflection(BaseModel):
    """Reflection on research progress and knowledge gaps."""
    is_sufficient: bool = Field(description="Whether current research is sufficient")
    knowledge_gap: str = Field(description="Description of remaining information gaps")
    follow_up_queries: List[str] = Field(
        description="Suggested follow-up queries",
        max_items=5
    )
    confidence_score: float = Field(
        description="Confidence in current research (0-1)",
        ge=0.0,
        le=1.0,
        default=0.5
    )


class CitationSegment(BaseModel):
    """Individual citation segment with metadata."""
    url: str = Field(description="Original URL")
    short_url: str = Field(description="Shortened URL for display")
    title: str = Field(description="Source title")
    snippet: Optional[str] = Field(description="Content snippet", default=None)


class SearchResult(BaseModel):
    """Individual search result with comprehensive metadata."""
    content: str = Field(description="Research content with citations")
    citations: List[CitationSegment] = Field(description="Source citations")
    query_used: str = Field(description="Original search query")
    timestamp: datetime = Field(default_factory=datetime.now)
    search_id: int = Field(description="Unique search identifier")


@dataclass
class ResearchState:
    """Current state of research process with comprehensive tracking."""
    topic: str
    effort_level: Literal["low", "medium", "high"]
    search_count: int = 0
    loop_count: int = 0
    results: List[SearchResult] = field(default_factory=list)
    all_citations: List[CitationSegment] = field(default_factory=list)
    is_complete: bool = False
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def max_searches(self) -> int:
        """Get maximum searches allowed for current effort level."""
        return EFFORT_TIERS[self.effort_level]["max_searches"]
    
    @property
    def max_loops(self) -> int:
        """Get maximum research loops allowed for current effort level."""
        return EFFORT_TIERS[self.effort_level]["max_research_loops"]
    
    @property
    def searches_remaining(self) -> int:
        """Get remaining searches allowed."""
        return max(0, self.max_searches - self.search_count)
    
    @property
    def can_continue(self) -> bool:
        """Check if research can continue based on limits."""
        return (
            self.search_count < self.max_searches and 
            self.loop_count < self.max_loops and 
            not self.is_complete
        )


# Initialize FastMCP server with proper configuration
mcp = FastMCP(
    name="Gemini Research Agent",
    version="1.0.0"
)

# Global research state tracking
active_research_sessions: Dict[str, ResearchState] = {}


def get_current_date() -> str:
    """Get current date in human-readable format."""
    return datetime.now().strftime("%B %d, %Y")


def generate_session_id(topic: str, effort_level: str) -> str:
    """Generate unique session ID for research tracking."""
    content = f"{topic}_{effort_level}_{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def validate_url(url: str) -> bool:
    """Validate if URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def resolve_urls(grounding_chunks: List[Any], search_id: int) -> List[CitationSegment]:
    """Process and resolve URLs from grounding chunks with validation."""
    citations = []
    
    for i, chunk in enumerate(grounding_chunks):
        try:
            if hasattr(chunk, 'web') and chunk.web and hasattr(chunk.web, 'uri'):
                url = chunk.web.uri
                if validate_url(url):
                    title = getattr(chunk.web, 'title', 'Unknown Source')
                    # Clean and truncate title
                    title = re.sub(r'\s+', ' ', title).strip()
                    if len(title) > 100:
                        title = title[:97] + "..."
                    
                    citations.append(CitationSegment(
                        url=url,
                        short_url=f"[{search_id}-{i}]",
                        title=title,
                        snippet=getattr(chunk.web, 'snippet', None)
                    ))
        except Exception as e:
            logger.warning(f"Error processing grounding chunk {i}: {e}")
            continue
    
    return citations


def insert_citation_markers(text: str, citations: List[CitationSegment]) -> str:
    """Insert citation markers into text with improved formatting."""
    if not citations:
        return text
    
    # Add citations at the end of sentences or paragraphs
    citation_text = "\n\n**Sources:**\n"
    for citation in citations:
        citation_text += f"- {citation.short_url} [{citation.title}]({citation.url})\n"
    
    return text + citation_text


async def generate_search_queries(
    research_topic: str, 
    num_queries: int,
    existing_results: Optional[List[SearchResult]] = None
) -> SearchQueryList:
    """Generate sophisticated search queries for research topic."""
    
    # Build context from existing results if available
    context = ""
    if existing_results:
        context = "\n\nPrevious research context:\n"
        for result in existing_results[-3:]:  # Last 3 results for context
            context += f"- Query: {result.query_used}\n"
            context += f"  Key findings: {result.content[:200]}...\n"
    
    prompt = f"""You are a research query specialist. Generate {num_queries} sophisticated and diverse web search queries for comprehensive research on: {research_topic}

Current date: {get_current_date()}

Instructions:
- Create queries that explore different aspects and perspectives of the topic
- Ensure queries are specific enough to find relevant, authoritative sources
- Include queries for recent developments, expert opinions, and factual data
- Avoid duplicate or overly similar queries
- Each query should have a clear rationale explaining its purpose
{context}

Format your response as a JSON object with:
- queries: array of objects with 'query' and 'rationale' fields
- rationale: overall explanation for the query selection strategy

Topic: {research_topic}"""

    try:
        llm = ChatGoogleGenerativeAI(
            model=QUERY_MODEL,
            temperature=0.7,  # Slightly creative for query diversity
            max_retries=3,
            api_key=GEMINI_API_KEY,
        )
        
        structured_llm = llm.with_structured_output(SearchQueryList)
        result = structured_llm.invoke(prompt)
        
        logger.info(f"Generated {len(result.queries)} search queries for topic: {research_topic}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating search queries: {e}")
        # Fallback to simple queries
        fallback_queries = [
            SearchQuery(query=research_topic, rationale="Direct topic search"),
            SearchQuery(query=f"{research_topic} recent developments", rationale="Recent information"),
        ]
        return SearchQueryList(
            queries=fallback_queries[:num_queries],
            rationale="Fallback queries due to generation error"
        )


async def perform_web_search(
    query: str, 
    search_id: int,
    max_retries: int = 3
) -> SearchResult:
    """Perform web search using Google Search API with comprehensive error handling."""
    
    prompt = f"""Conduct a comprehensive Google Search on "{query}" and provide a detailed, well-structured summary.

Instructions:
- Current date: {get_current_date()}
- Search for the most recent, credible, and authoritative information
- Provide a comprehensive summary with key findings, facts, and insights
- Structure your response with clear sections and bullet points where appropriate
- Focus on factual information from reliable sources
- Include relevant statistics, expert opinions, and recent developments
- Only include information that can be verified from search results

Search Query: {query}"""

    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=RESEARCH_MODEL,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "temperature": 0.1,  # Low temperature for factual accuracy
                },
            )
            
            # Process grounding metadata for citations
            citations = []
            
            if (hasattr(response, 'candidates') and 
                response.candidates and 
                hasattr(response.candidates[0], 'grounding_metadata') and
                response.candidates[0].grounding_metadata and
                hasattr(response.candidates[0].grounding_metadata, 'grounding_chunks')):
                
                grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
                citations = resolve_urls(grounding_chunks, search_id)
            
            # Format content with citations
            content = response.text if response.text else "No content retrieved"
            formatted_content = insert_citation_markers(content, citations)
            
            search_result = SearchResult(
                content=formatted_content,
                citations=citations,
                query_used=query,
                search_id=search_id
            )
            
            logger.info(f"Successfully completed search {search_id} for query: {query}")
            return search_result
            
        except Exception as e:
            logger.warning(f"Search attempt {attempt + 1} failed for query '{query}': {e}")
            if attempt == max_retries - 1:
                # Final fallback
                return SearchResult(
                    content=f"Search failed for query: {query}. Error: {str(e)}",
                    citations=[],
                    query_used=query,
                    search_id=search_id
                )
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


async def reflect_on_research(
    research_topic: str,
    current_results: List[SearchResult],
    effort_level: str
) -> ResearchReflection:
    """Analyze research progress and identify knowledge gaps."""
    
    if not current_results:
        return ResearchReflection(
            is_sufficient=False,
            knowledge_gap="No research results available yet",
            follow_up_queries=[research_topic],
            confidence_score=0.0
        )
    
    # Prepare summary of current findings
    findings_summary = "\n\n".join([
        f"Query: {result.query_used}\nFindings: {result.content[:400]}..."
        for result in current_results[-5:]  # Last 5 results
    ])
    
    effort_context = EFFORT_TIERS[effort_level]["description"]
    
    prompt = f"""Analyze the current research progress and determine if we have sufficient information or need additional research.

Research Topic: {research_topic}
Research Effort Level: {effort_level} ({effort_context})
Current Date: {get_current_date()}

Current Research Findings:
{findings_summary}

Please evaluate:
1. Are the current findings comprehensive enough to answer the research topic?
2. What specific knowledge gaps or areas need more investigation?
3. What follow-up queries would address these gaps most effectively?
4. Rate your confidence in the current research completeness (0-1 scale)

Provide your analysis in the specified JSON format."""

    try:
        llm = ChatGoogleGenerativeAI(
            model=REFLECTION_MODEL,
            temperature=0.3,
            max_retries=3,
            api_key=GEMINI_API_KEY,
        )
        
        structured_llm = llm.with_structured_output(ResearchReflection)
        result = structured_llm.invoke(prompt)
        
        logger.info(f"Research reflection completed. Sufficient: {result.is_sufficient}, Confidence: {result.confidence_score}")
        return result
        
    except Exception as e:
        logger.error(f"Error in research reflection: {e}")
        # Conservative fallback
        return ResearchReflection(
            is_sufficient=len(current_results) >= 3,  # Simple heuristic
            knowledge_gap="Unable to analyze research completeness due to processing error",
            follow_up_queries=[f"{research_topic} additional information"],
            confidence_score=0.5
        )


async def finalize_research_answer(
    research_topic: str,
    results: List[SearchResult],
    effort_level: str
) -> str:
    """Generate comprehensive final answer from research results."""
    
    if not results:
        return f"No research results were obtained for the topic: {research_topic}"
    
    # Combine all research content
    combined_research = "\n\n".join([
        f"Research Query: {result.query_used}\n{result.content}"
        for result in results
    ])
    
    # Collect all unique citations
    all_citations = []
    seen_urls = set()
    for result in results:
        for citation in result.citations:
            if citation.url not in seen_urls:
                all_citations.append(citation)
                seen_urls.add(citation.url)
    
    effort_context = EFFORT_TIERS[effort_level]["description"]
    
    prompt = f"""Based on the comprehensive research conducted, provide a detailed, well-structured answer to the research topic.

Research Topic: {research_topic}
Research Effort: {effort_level} ({effort_context})
Current Date: {get_current_date()}
Total Research Queries: {len(results)}

Research Content:
{combined_research}

Instructions:
- Synthesize all research findings into a comprehensive, coherent answer
- Structure your response with clear headings and sections
- Include key facts, statistics, expert opinions, and recent developments
- Maintain objectivity and cite multiple perspectives where relevant
- Ensure accuracy and avoid speculation beyond the research findings
- Provide actionable insights or conclusions where appropriate
- Keep the response informative yet accessible

Generate a comprehensive research report that fully addresses the topic."""

    try:
        llm = ChatGoogleGenerativeAI(
            model=ANSWER_MODEL,
            temperature=0.2,  # Low temperature for accuracy
            max_retries=3,
            api_key=GEMINI_API_KEY,
        )
        
        response = llm.invoke(prompt)
        final_answer = response.content
        
        # Add citation appendix
        if all_citations:
            final_answer += "\n\n## Sources\n"
            for i, citation in enumerate(all_citations, 1):
                final_answer += f"{i}. [{citation.title}]({citation.url})\n"
        
        # Add research metadata
        final_answer += f"\n\n---\n*Research completed on {get_current_date()} using {effort_level} effort level with {len(results)} queries*"
        
        logger.info(f"Finalized research answer for topic: {research_topic}")
        return final_answer
        
    except Exception as e:
        logger.error(f"Error finalizing research answer: {e}")
        # Fallback to simple summary
        return f"Research Summary for: {research_topic}\n\n" + combined_research


# MCP Tool Implementations

@mcp.tool(
    description="Conduct comprehensive research on any topic with configurable effort levels and intelligent search capabilities"
)
async def research_topic(
    topic: str = Field(
        description="The research topic, question, or subject to investigate thoroughly",
        min_length=1,
        max_length=500
    ),
    effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Research effort level: low (10 searches max, 1 loop), medium (100 searches max, 3 loops), high (1000 searches max, 5 loops)"
    )
) -> str:
    """
    Conduct comprehensive research on any topic using Google's Gemini AI with tiered effort levels.
    
    This tool performs intelligent web research with:
    - Multi-stage search query generation
    - Iterative research loops with reflection
    - Citation tracking and source validation
    - Comprehensive answer synthesis
    
    Args:
        topic: The research topic or question to investigate
        effort: Research intensity level (low/medium/high)
    
    Returns:
        Comprehensive research report with citations and sources
    """
    
    if not topic.strip():
        return "Error: Research topic cannot be empty."
    
    topic = topic.strip()
    session_id = generate_session_id(topic, effort)
    
    logger.info(f"Starting research session {session_id} for topic: {topic} (effort: {effort})")
    
    try:
        # Initialize research state
        research_state = ResearchState(topic=topic, effort_level=effort)
        active_research_sessions[session_id] = research_state
        
        config = EFFORT_TIERS[effort]
        
        # Phase 1: Generate initial search queries
        initial_query_list = await generate_search_queries(
            topic, 
            config["initial_queries"]
        )
        
        # Phase 2: Conduct initial research
        for i, query_obj in enumerate(initial_query_list.queries):
            if not research_state.can_continue:
                break
                
            search_result = await perform_web_search(
                query_obj.query, 
                research_state.search_count
            )
            
            research_state.results.append(search_result)
            research_state.all_citations.extend(search_result.citations)
            research_state.search_count += 1
        
        # Phase 3: Iterative research loops with reflection
        while research_state.can_continue and research_state.loop_count < research_state.max_loops:
            research_state.loop_count += 1
            
            # Reflect on current research
            reflection = await reflect_on_research(
                topic, 
                research_state.results, 
                effort
            )
            
            # Check if research is sufficient
            if reflection.is_sufficient or reflection.confidence_score > 0.8:
                logger.info(f"Research deemed sufficient after {research_state.loop_count} loops")
                break
            
            # Generate follow-up queries based on knowledge gaps
            if reflection.follow_up_queries and research_state.searches_remaining > 0:
                follow_up_queries = reflection.follow_up_queries[:research_state.searches_remaining]
                
                for query in follow_up_queries:
                    if not research_state.can_continue:
                        break
                    
                    search_result = await perform_web_search(
                        query, 
                        research_state.search_count
                    )
                    
                    research_state.results.append(search_result)
                    research_state.all_citations.extend(search_result.citations)
                    research_state.search_count += 1
        
        # Phase 4: Finalize comprehensive answer
        research_state.is_complete = True
        final_answer = await finalize_research_answer(
            topic, 
            research_state.results, 
            effort
        )
        
        # Cleanup session
        if session_id in active_research_sessions:
            del active_research_sessions[session_id]
        
        logger.info(f"Research completed for topic: {topic}. Total searches: {research_state.search_count}")
        return final_answer
        
    except Exception as e:
        logger.error(f"Error during research for topic '{topic}': {e}")
        if session_id in active_research_sessions:
            del active_research_sessions[session_id]
        return f"Research failed due to an error: {str(e)}. Please try again or contact support."


@mcp.tool(
    description="Get detailed information about available research effort levels and their capabilities"
)
async def get_effort_levels() -> str:
    """
    Get comprehensive information about available research effort levels.
    
    Returns:
        Detailed breakdown of effort levels and their specifications
    """
    
    info = "# Research Effort Levels\n\n"
    info += "The Gemini Research Agent supports three effort levels for different research needs:\n\n"
    
    for level, config in EFFORT_TIERS.items():
        info += f"## {level.title()} Effort\n"
        info += f"- **Max Searches**: {config['max_searches']}\n"
        info += f"- **Max Research Loops**: {config['max_research_loops']}\n"
        info += f"- **Initial Queries**: {config['initial_queries']}\n"
        info += f"- **Description**: {config['description']}\n\n"
    
    info += "## Recommendations\n"
    info += "- **Low**: Quick fact-checking, simple questions, time-sensitive research\n"
    info += "- **Medium**: Most general research needs, balanced depth and speed\n"
    info += "- **High**: Complex topics, academic research, comprehensive analysis\n\n"
    info += f"*Server Model: {RESEARCH_MODEL}*\n"
    info += f"*Last Updated: {get_current_date()}*"
    
    return info


@mcp.tool(
    description="Get current server status and active research session information"
)
async def get_server_status() -> str:
    """
    Get current server status, configuration, and active research sessions.
    
    Returns:
        Server status and configuration information
    """
    
    status = f"# Gemini Research Agent Server Status\n\n"
    status += f"**Server Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    status += f"**Research Model**: {RESEARCH_MODEL}\n"
    status += f"**Query Model**: {QUERY_MODEL}\n"
    status += f"**Active Sessions**: {len(active_research_sessions)}\n\n"
    
    if active_research_sessions:
        status += "## Active Research Sessions\n"
        for session_id, state in active_research_sessions.items():
            elapsed = datetime.now() - state.start_time
            status += f"- **{session_id}**: {state.topic[:50]}{'...' if len(state.topic) > 50 else ''}\n"
            status += f"  - Effort: {state.effort_level}\n"
            status += f"  - Searches: {state.search_count}/{state.max_searches}\n"
            status += f"  - Loops: {state.loop_count}/{state.max_loops}\n"
            status += f"  - Elapsed: {elapsed.seconds}s\n"
    
    status += "\n## Configuration\n"
    for level, config in EFFORT_TIERS.items():
        status += f"**{level.title()}**: {config['max_searches']} searches, {config['max_research_loops']} loops\n"
    
    return status


# MCP Resource for research documentation
@mcp.resource("research://documentation")
async def get_research_documentation() -> str:
    """
    Comprehensive documentation for the Gemini Research Agent MCP Server.
    
    Returns:
        Complete documentation and usage guide
    """
    
    doc = f"""# Gemini Research Agent MCP Server

## Overview
The Gemini Research Agent is an advanced Model Context Protocol (MCP) server that provides comprehensive research capabilities using Google's Gemini AI models. It offers tiered effort levels to balance thoroughness with efficiency.

## Features
- **Multi-tier Research**: Three effort levels (low, medium, high) with different search limits
- **Intelligent Query Generation**: AI-powered search query creation and optimization
- **Iterative Research**: Multiple research loops with reflection and gap analysis
- **Citation Tracking**: Comprehensive source validation and citation management
- **Error Handling**: Robust error handling with fallback mechanisms
- **Async Performance**: Optimized for concurrent operations

## Models Used
- **Primary Research**: {RESEARCH_MODEL}
- **Query Generation**: {QUERY_MODEL}
- **Reflection**: {REFLECTION_MODEL}
- **Final Answer**: {ANSWER_MODEL}

## Available Tools

### research_topic(topic, effort="medium")
Conduct comprehensive research on any topic with configurable effort levels.

**Parameters:**
- `topic` (str): The research topic or question to investigate
- `effort` (str): Research effort level - "low", "medium", or "high"

**Returns:** Comprehensive research report with citations

### get_effort_levels()
Get detailed information about available research effort levels.

**Returns:** Detailed breakdown of effort levels and capabilities

### get_server_status()
Get current server status and active research sessions.

**Returns:** Server status and configuration information

## Effort Levels

| Level  | Max Searches | Max Loops | Initial Queries | Best For |
|--------|-------------|-----------|-----------------|----------|
| Low    | 10          | 1         | 2               | Quick facts, simple questions |
| Medium | 100         | 3         | 4               | General research, balanced approach |
| High   | 1000        | 5         | 6               | Complex topics, comprehensive analysis |

## Usage Examples

### Basic Research
```
research_topic("artificial intelligence trends 2024")
```

### Low Effort Quick Research
```
research_topic("current weather in Paris", effort="low")
```

### High Effort Comprehensive Research
```
research_topic("impact of climate change on global agriculture", effort="high")
```

## Technical Details

### Architecture
- Built with FastMCP for optimal performance
- Async/await pattern for concurrent operations
- Comprehensive error handling and logging
- State management for active research sessions

### Citation System
- URL validation and normalization
- Source title extraction and cleaning
- Citation formatting with markdown links
- Duplicate source detection and removal

### Rate Limiting
- Effort-based search limits prevent API abuse
- Exponential backoff for failed requests
- Session tracking for resource management

## Error Handling
The server includes robust error handling:
- API failures with automatic retries
- Malformed query graceful degradation
- Network timeout handling
- Fallback responses for critical failures

## Logging
Comprehensive logging includes:
- Research session tracking
- Search query performance metrics
- Error reporting and debugging information
- Resource usage monitoring

---
*Documentation generated on {get_current_date()}*
*Version: 1.0.0*
"""
    
    return doc


# Server startup and configuration
if __name__ == "__main__":
    try:
        logger.info(f"Starting Gemini Research Agent MCP Server")
        logger.info(f"Using model: {RESEARCH_MODEL}")
        logger.info(f"Effort tiers configured: {', '.join(EFFORT_TIERS.keys())}")
        
        # Run the FastMCP server
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
    finally:
        # Cleanup active sessions
        active_research_sessions.clear()
        logger.info("Server shutdown complete")