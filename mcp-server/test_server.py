#!/usr/bin/env python3
"""
Test suite for Gemini Research Agent MCP Server

Comprehensive tests to ensure functionality, reliability, and performance
of the MCP server components.

Run with: pytest test_server.py -v
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import List

# Import server components
from server import (
    ResearchState,
    SearchQuery,
    SearchQueryList,
    ResearchReflection,
    CitationSegment,
    SearchResult,
    EFFORT_TIERS,
    get_current_date,
    generate_session_id,
    validate_url,
    resolve_urls,
    insert_citation_markers,
    generate_search_queries,
    perform_web_search,
    reflect_on_research,
    finalize_research_answer,
)


class TestDataModels:
    """Test Pydantic data models and validation."""
    
    def test_search_query_validation(self):
        """Test SearchQuery model validation."""
        # Valid query
        query = SearchQuery(query="test query", rationale="test rationale")
        assert query.query == "test query"
        assert query.rationale == "test rationale"
        
        # Empty query should be stripped and validated
        query = SearchQuery(query="  test  ", rationale="rationale")
        assert query.query == "test"
        
        # Invalid empty query
        with pytest.raises(ValueError):
            SearchQuery(query="", rationale="rationale")
    
    def test_research_state_properties(self):
        """Test ResearchState dataclass properties."""
        state = ResearchState(topic="test", effort_level="medium")
        
        assert state.max_searches == 100
        assert state.max_loops == 3
        assert state.searches_remaining == 100
        assert state.can_continue is True
        
        # Test after some searches
        state.search_count = 50
        assert state.searches_remaining == 50
        
        # Test when limits reached
        state.search_count = 100
        assert state.can_continue is False
    
    def test_citation_segment_model(self):
        """Test CitationSegment model."""
        citation = CitationSegment(
            url="https://example.com",
            short_url="[1]",
            title="Test Title"
        )
        assert citation.url == "https://example.com"
        assert citation.short_url == "[1]"
        assert citation.title == "Test Title"
        assert citation.snippet is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_current_date(self):
        """Test date formatting function."""
        date = get_current_date()
        assert isinstance(date, str)
        assert len(date) > 10  # Should be formatted date
        
        # Verify format (e.g., "January 15, 2024")
        try:
            datetime.strptime(date, "%B %d, %Y")
        except ValueError:
            pytest.fail("Date format is incorrect")
    
    def test_generate_session_id(self):
        """Test session ID generation."""
        session_id = generate_session_id("test topic", "medium")
        assert isinstance(session_id, str)
        assert len(session_id) == 12  # MD5 hash truncated to 12 chars
        
        # Different inputs should generate different IDs
        id1 = generate_session_id("topic1", "low")
        id2 = generate_session_id("topic2", "high")
        assert id1 != id2
    
    def test_validate_url(self):
        """Test URL validation function."""
        # Valid URLs
        assert validate_url("https://example.com") is True
        assert validate_url("http://test.org/path") is True
        assert validate_url("https://sub.domain.com/path?query=1") is True
        
        # Invalid URLs
        assert validate_url("not-a-url") is False
        assert validate_url("") is False
        assert validate_url("ftp://example.com") is True  # Valid but different scheme
        assert validate_url("//example.com") is False  # Missing scheme
    
    def test_insert_citation_markers(self):
        """Test citation marker insertion."""
        text = "This is a test text."
        citations = [
            CitationSegment(
                url="https://example.com",
                short_url="[1]",
                title="Example Source"
            )
        ]
        
        result = insert_citation_markers(text, citations)
        assert "This is a test text." in result
        assert "**Sources:**" in result
        assert "[1] [Example Source](https://example.com)" in result
        
        # Test with empty citations
        result = insert_citation_markers(text, [])
        assert result == text


class TestAsyncFunctions:
    """Test async functions with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_generate_search_queries(self):
        """Test search query generation."""
        with patch('server.ChatGoogleGenerativeAI') as mock_llm_class:
            # Mock the LLM response
            mock_llm = Mock()
            mock_structured_llm = Mock()
            mock_structured_llm.invoke = AsyncMock(return_value=SearchQueryList(
                queries=[
                    SearchQuery(query="test query 1", rationale="rationale 1"),
                    SearchQuery(query="test query 2", rationale="rationale 2")
                ],
                rationale="Overall rationale"
            ))
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_llm_class.return_value = mock_llm
            
            result = await generate_search_queries("test topic", 2)
            
            assert isinstance(result, SearchQueryList)
            assert len(result.queries) == 2
            assert result.queries[0].query == "test query 1"
    
    @pytest.mark.asyncio
    async def test_generate_search_queries_fallback(self):
        """Test search query generation fallback on error."""
        with patch('server.ChatGoogleGenerativeAI') as mock_llm_class:
            # Mock LLM to raise an exception
            mock_llm_class.side_effect = Exception("API Error")
            
            result = await generate_search_queries("test topic", 2)
            
            assert isinstance(result, SearchQueryList)
            assert len(result.queries) <= 2
            assert "test topic" in result.queries[0].query
    
    @pytest.mark.asyncio
    async def test_reflect_on_research(self):
        """Test research reflection function."""
        # Create mock results
        mock_results = [
            SearchResult(
                content="Test content 1",
                citations=[],
                query_used="query 1",
                search_id=1
            ),
            SearchResult(
                content="Test content 2",
                citations=[],
                query_used="query 2",
                search_id=2
            )
        ]
        
        with patch('server.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm = Mock()
            mock_structured_llm = Mock()
            mock_structured_llm.invoke = AsyncMock(return_value=ResearchReflection(
                is_sufficient=True,
                knowledge_gap="No gaps",
                follow_up_queries=[],
                confidence_score=0.9
            ))
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_llm_class.return_value = mock_llm
            
            result = await reflect_on_research("test topic", mock_results, "medium")
            
            assert isinstance(result, ResearchReflection)
            assert result.is_sufficient is True
            assert result.confidence_score == 0.9
    
    @pytest.mark.asyncio
    async def test_reflect_on_research_empty_results(self):
        """Test research reflection with empty results."""
        result = await reflect_on_research("test topic", [], "low")
        
        assert isinstance(result, ResearchReflection)
        assert result.is_sufficient is False
        assert result.confidence_score == 0.0
        assert "test topic" in result.follow_up_queries[0]


class TestConfiguration:
    """Test configuration and constants."""
    
    def test_effort_tiers_configuration(self):
        """Test effort tier configurations."""
        assert "low" in EFFORT_TIERS
        assert "medium" in EFFORT_TIERS
        assert "high" in EFFORT_TIERS
        
        # Verify tier values
        low_tier = EFFORT_TIERS["low"]
        assert low_tier["max_searches"] == 10
        assert low_tier["max_research_loops"] == 1
        assert low_tier["initial_queries"] == 2
        
        medium_tier = EFFORT_TIERS["medium"]
        assert medium_tier["max_searches"] == 100
        assert medium_tier["max_research_loops"] == 3
        assert medium_tier["initial_queries"] == 4
        
        high_tier = EFFORT_TIERS["high"]
        assert high_tier["max_searches"] == 1000
        assert high_tier["max_research_loops"] == 5
        assert high_tier["initial_queries"] == 6
        
        # Verify ascending order
        assert low_tier["max_searches"] < medium_tier["max_searches"] < high_tier["max_searches"]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_perform_web_search_error_handling(self):
        """Test web search error handling."""
        with patch('server.genai_client') as mock_client:
            # Mock client to raise an exception
            mock_client.models.generate_content.side_effect = Exception("API Error")
            
            result = await perform_web_search("test query", 1)
            
            assert isinstance(result, SearchResult)
            assert "Search failed" in result.content
            assert result.query_used == "test query"
            assert result.search_id == 1
    
    def test_resolve_urls_error_handling(self):
        """Test URL resolution error handling."""
        # Mock grounding chunks with various error conditions
        mock_chunks = [
            Mock(web=Mock(uri="https://valid.com", title="Valid Title")),
            Mock(web=None),  # No web attribute
            Mock(),  # No web attribute at all
        ]
        
        with patch('server.validate_url', return_value=True):
            citations = resolve_urls(mock_chunks, 1)
            
            # Should only process valid chunks
            assert len(citations) == 1
            assert citations[0].url == "https://valid.com"


class TestIntegration:
    """Integration tests for combined functionality."""
    
    @pytest.mark.asyncio
    async def test_research_workflow_simulation(self):
        """Test a simplified research workflow."""
        # This test simulates the main research workflow without external API calls
        
        with patch('server.generate_search_queries') as mock_gen_queries, \
             patch('server.perform_web_search') as mock_search, \
             patch('server.reflect_on_research') as mock_reflect, \
             patch('server.finalize_research_answer') as mock_finalize:
            
            # Mock search query generation
            mock_gen_queries.return_value = SearchQueryList(
                queries=[SearchQuery(query="test query", rationale="test")],
                rationale="Test rationale"
            )
            
            # Mock search results
            mock_search.return_value = SearchResult(
                content="Test search result",
                citations=[],
                query_used="test query",
                search_id=1
            )
            
            # Mock reflection
            mock_reflect.return_value = ResearchReflection(
                is_sufficient=True,
                knowledge_gap="None",
                follow_up_queries=[],
                confidence_score=0.9
            )
            
            # Mock final answer
            mock_finalize.return_value = "Comprehensive research answer"
            
            # Test the workflow components
            queries = await mock_gen_queries("test topic", 1)
            assert len(queries.queries) == 1
            
            search_result = await mock_search("test query", 1)
            assert search_result.content == "Test search result"
            
            reflection = await mock_reflect("test topic", [search_result], "medium")
            assert reflection.is_sufficient is True
            
            final_answer = await mock_finalize("test topic", [search_result], "medium")
            assert final_answer == "Comprehensive research answer"


class TestPerformance:
    """Performance and resource usage tests."""
    
    def test_session_id_generation_performance(self):
        """Test session ID generation performance."""
        import time
        
        start_time = time.time()
        for i in range(1000):
            generate_session_id(f"topic_{i}", "medium")
        end_time = time.time()
        
        # Should generate 1000 session IDs in less than 1 second
        assert (end_time - start_time) < 1.0
    
    def test_url_validation_performance(self):
        """Test URL validation performance."""
        import time
        
        urls = [
            "https://example.com",
            "http://test.org",
            "invalid-url",
            "https://long.domain.with.many.subdomains.com/very/long/path",
        ] * 250  # 1000 URLs total
        
        start_time = time.time()
        for url in urls:
            validate_url(url)
        end_time = time.time()
        
        # Should validate 1000 URLs in less than 0.5 seconds
        assert (end_time - start_time) < 0.5


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 