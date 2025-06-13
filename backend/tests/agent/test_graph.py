import unittest
import os # Keep one os import if os.environ is used by side_effect, otherwise not needed at top
from unittest.mock import patch, MagicMock

from langchain_core.runnables import RunnableConfig

# Global os.getenv patch for ChatGoogleGenerativeAI call in generate_query
def mock_getenv_side_effect(key, default=None):
    if key == "GEMINI_API_KEY":
        return "test_key_for_gemini"
    return os.environ.get(key, default) # Fallback for any other keys

patch_os_getenv = patch('os.getenv', MagicMock(side_effect=mock_getenv_side_effect))
patch_os_getenv.start()

# No need for global google.genai.Client patch anymore.

# Import modules under test AFTER global patches are started
from src.agent.graph import generate_query, web_research, reflection, finalize_answer, web_search_cache # Added web_search_cache
from src.agent.state import OverallState, WebSearchState
from src.agent.tools_and_schemas import SearchQueryList, Reflection
from src.agent.prompts import web_searcher_instructions, reflection_instructions, answer_instructions
from langchain_core.messages import AIMessage


class TestGraphNodes(unittest.TestCase):
    def setUp(self):
        """Clear the web_search_cache before each test."""
        web_search_cache.clear()
        # It's also good practice to reset mocks that might be shared or have persistent state
        # if they are not re-patched in every test method. However, most of our critical mocks
        # are method-scoped via decorators. The global 'patch_os_getenv' is stateless for our use.

    @patch('src.agent.graph.ChatGoogleGenerativeAI')
    @patch('src.agent.graph.Configuration')
    @patch('src.agent.graph.get_current_date')
    @patch('src.agent.graph.get_research_topic')
    def test_generate_query_success(self,
                                     mock_get_research_topic,
                                     mock_get_current_date,
                                     mock_Configuration,
                                     mock_ChatGoogleGenerativeAI):

        # --- Arrange ---
        # IS_TESTING logic removed as it's no longer relevant for generate_query tests
        # Mock get_current_date
        mock_get_current_date.return_value = "January 1, 2024"

        # Mock get_research_topic
        mock_get_research_topic.return_value = "What is LangGraph?"

        # Mock Configuration
        mock_config_instance = MagicMock()
        mock_config_instance.query_generator_model = "gemini-flash"
        mock_config_instance.number_of_initial_queries = 1
        mock_config_instance.llm_temperature = 0.5 # New value
        mock_config_instance.num_search_results = 3 # New value, though not used by generate_query directly
        mock_Configuration.from_runnable_config.return_value = mock_config_instance

        # Mock ChatGoogleGenerativeAI
        mock_llm_instance = MagicMock()
        mock_structured_llm_instance = MagicMock()
        expected_query_strings = ["LangGraph basics"]
        expected_rationale_str = "To understand the fundamentals"
        mock_structured_llm_instance.invoke.return_value = SearchQueryList(
            query=expected_query_strings,
            rationale=expected_rationale_str
        )
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm_instance
        mock_ChatGoogleGenerativeAI.return_value = mock_llm_instance

        # Prepare state and config
        initial_state: OverallState = {
            "messages": [("user", "What is LangGraph?")],
            "search_query": [],
            "web_research_result": [],
            "sources_gathered": [],
            "initial_search_query_count": None,
            "max_research_loops": 3,
            "research_loop_count": 0,
            "reasoning_model": "gemini-pro"
        }
        runnable_config = RunnableConfig(configurable={})

        # --- Act ---
        result_state_update = generate_query(initial_state, runnable_config)

        # --- Assert ---
        self.assertIn("query_list", result_state_update)
        self.assertIsInstance(result_state_update["query_list"], list)
        self.assertEqual(len(result_state_update["query_list"]), 1)
        self.assertEqual(result_state_update["query_list"][0], "LangGraph basics")

        mock_get_current_date.assert_called_once()
        mock_get_research_topic.assert_called_once_with(initial_state["messages"])
        mock_Configuration.from_runnable_config.assert_called_once_with(runnable_config)
        mock_ChatGoogleGenerativeAI.assert_called_once_with(
            model="gemini-flash",
            temperature=0.5, # Check for new temperature
            max_retries=2,
            api_key='test_key_for_gemini',
        )
        mock_llm_instance.with_structured_output.assert_called_once_with(SearchQueryList)
        mock_structured_llm_instance.invoke.assert_called_once()

    # Add more tests for generate_query (e.g., different initial_search_query_count)

    @patch('src.agent.graph.Configuration') # For Configuration.from_runnable_config
    @patch('src.agent.graph.insert_citation_markers')
    @patch('src.agent.graph.get_citations')
    @patch('src.agent.graph.resolve_urls')
    @patch('src.agent.graph.get_current_date')
    @patch('src.agent.graph.initialize_genai_client')
    def test_web_research_cache_logic(self,
                                      mock_initialize_genai_client,
                                      mock_get_current_date,
                                      mock_resolve_urls,
                                      mock_get_citations,
                                      mock_insert_citation_markers,
                                      mock_Configuration):
        # --- Arrange common mock setup ---
        mock_config_instance = MagicMock()
        mock_config_instance.query_generator_model = "cache-test-model"
        mock_config_instance.num_search_results = 3 # Default for config
        mock_Configuration.from_runnable_config.return_value = mock_config_instance

        mock_genai_client = MagicMock()
        mock_initialize_genai_client.return_value = mock_genai_client

        mock_response_content = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata.grounding_chunks = [{"url": "http://example.com/cached_source"}]
        mock_response_content.candidates = [mock_candidate]
        mock_response_content.text = "Cached research result."
        mock_genai_client.models.generate_content.return_value = mock_response_content

        mock_get_current_date.return_value = "May 5, 2024" # Consistent date for prompts

        # Mock utility functions to return consistent distinct values based on input if necessary,
        # or just generic distinct values if their specific output doesn't affect cache key.
        # For simplicity, they return fixed values here.
        mock_resolved_urls_val = [{"url": "http://example.com/cached_source", "short_url": "[c1]"}]
        mock_resolve_urls.return_value = mock_resolved_urls_val

        mock_citations_val = [{"segments": [{"source_index": 1, "text_segment": "Cached segment", "value": "http://example.com/cached_source"}]}]
        mock_get_citations.return_value = mock_citations_val

        mock_modified_text_val = "Cached research result with [c1]."
        mock_insert_citation_markers.return_value = mock_modified_text_val

        runnable_config = RunnableConfig(configurable={})

        # --- Test 1: Cache Miss ---
        state_miss: WebSearchState = {"search_query": "query for cache", "id": "id_miss", "num_search_results": 3}
        result_miss = web_research(state_miss, runnable_config)

        mock_initialize_genai_client.assert_called_once() # Called on miss
        mock_genai_client.models.generate_content.assert_called_once() # Called on miss
        self.assertEqual(result_miss["web_research_result"][0], mock_modified_text_val)

        # --- Test 2: Cache Hit ---
        mock_initialize_genai_client.reset_mock() # Reset for the next call check
        mock_genai_client.models.generate_content.reset_mock()

        state_hit: WebSearchState = {"search_query": "query for cache", "id": "id_hit", "num_search_results": 3} # Same query and num_results
        result_hit = web_research(state_hit, runnable_config)

        mock_initialize_genai_client.assert_not_called() # NOT called on hit
        mock_genai_client.models.generate_content.assert_not_called() # NOT called on hit
        self.assertEqual(result_hit["web_research_result"][0], mock_modified_text_val) # Same result
        # Also check other parts of the result if necessary
        self.assertEqual(result_hit["sources_gathered"], result_miss["sources_gathered"])


        # --- Test 3: Cache Miss (different num_search_results) ---
        mock_initialize_genai_client.reset_mock()
        mock_genai_client.models.generate_content.reset_mock()
        # Re-configure mocks for resolve_urls etc. if their behavior should change for a new call,
        # or if they were called with different args that would make them fail the new assert_called_once.
        # For this test, we assume they produce similar structured but potentially different data.
        # To keep it simple, we'll let them return the same mocked values for now.

        state_miss_diff_num: WebSearchState = {"search_query": "query for cache", "id": "id_miss_2", "num_search_results": 5}
        result_miss_diff_num = web_research(state_miss_diff_num, runnable_config)

        mock_initialize_genai_client.assert_called_once() # Called again
        mock_genai_client.models.generate_content.assert_called_once() # Called again
        self.assertEqual(result_miss_diff_num["web_research_result"][0], mock_modified_text_val)

        # --- Test 4: Cache Miss (different search_query) ---
        mock_initialize_genai_client.reset_mock()
        mock_genai_client.models.generate_content.reset_mock()

        state_miss_diff_query: WebSearchState = {"search_query": "DIFFERENT query", "id": "id_miss_3", "num_search_results": 3}
        result_miss_diff_query = web_research(state_miss_diff_query, runnable_config)

        mock_initialize_genai_client.assert_called_once() # Called again
        mock_genai_client.models.generate_content.assert_called_once() # Called again
        self.assertEqual(result_miss_diff_query["web_research_result"][0], mock_modified_text_val)


    @patch('src.agent.graph.Configuration')
    @patch('src.agent.graph.get_research_topic') # For formatting the prompt
    @patch('src.agent.graph.get_current_date') # For formatting the prompt
    @patch('src.agent.graph.ChatGoogleGenerativeAI') # For the LLM call
    def test_finalize_answer_success(self,
                                     mock_ChatGoogleGenerativeAI,
                                     mock_get_current_date,
                                     mock_get_research_topic,
                                     mock_Configuration):
        # --- Arrange ---
        # Mock Configuration
        mock_config_instance = MagicMock()
        mock_config_instance.reasoning_model = "gemini-finalizer"
        mock_config_instance.llm_temperature = 0.9 # New value
        mock_config_instance.num_search_results = 2 # New value, though not used by finalize_answer node
        mock_Configuration.from_runnable_config.return_value = mock_config_instance

        # Mock LLM (ChatGoogleGenerativeAI)
        mock_llm_instance = MagicMock()
        mock_llm_response = MagicMock() # Simulates AIMessage-like structure from LLM
        # LLM's raw response will contain short_urls
        mock_llm_response.content = "Final answer incorporating [src1] and some other text. [src3] is also relevant."
        mock_llm_instance.invoke.return_value = mock_llm_response
        mock_ChatGoogleGenerativeAI.return_value = mock_llm_instance

        # Mock utility functions
        mock_get_current_date.return_value = "April 4, 2024"
        mock_get_research_topic.return_value = "Final Research Question"

        # Prepare initial state for finalize_answer node
        initial_finalize_state: OverallState = {
            "messages": [("user", "Please give me the final answer for Final Research Question")],
            "web_research_result": ["Comprehensive summary 1.", "Detailed summary 2."],
            "sources_gathered": [
                {"short_url": "[src1]", "value": "http://example.com/original_source_1", "title": "Source 1"},
                {"short_url": "[src2]", "value": "http://example.com/original_source_2", "title": "Source 2 (Unused)"},
                {"short_url": "[src3]", "value": "http://example.com/original_source_3", "title": "Source 3"},
            ],
            # Other OverallState keys - provide defaults or typical values
            "search_query": ["final_query"],
            "research_loop_count": 3,
            "initial_search_query_count": 1,
            "max_research_loops": 3,
            "reasoning_model": None # To test default model from config
        }
        runnable_config = RunnableConfig(configurable={})

        # --- Act ---
        result = finalize_answer(initial_finalize_state, runnable_config)

        # --- Assert ---
        mock_Configuration.from_runnable_config.assert_called_once_with(runnable_config)
        mock_get_current_date.assert_called_once()
        mock_get_research_topic.assert_called_once_with(initial_finalize_state["messages"])

        mock_ChatGoogleGenerativeAI.assert_called_once_with(
            model="gemini-finalizer",
            temperature=0.9, # Check for new temperature
            max_retries=2,
            api_key='test_key_for_gemini'
        )

        expected_prompt_format = answer_instructions.format( # Need to import answer_instructions
            current_date="April 4, 2024",
            research_topic="Final Research Question",
            summaries="Comprehensive summary 1.\n---\n\nDetailed summary 2."
        )
        mock_llm_instance.invoke.assert_called_once_with(expected_prompt_format)

        # Check the structure of the output
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        final_message = result["messages"][0]
        self.assertIsInstance(final_message, AIMessage) # Need to import AIMessage

        # Check content for URL replacement
        expected_final_content = "Final answer incorporating http://example.com/original_source_1 and some other text. http://example.com/original_source_3 is also relevant."
        self.assertEqual(final_message.content, expected_final_content)

        # Check filtering of sources_gathered
        self.assertIn("sources_gathered", result)
        final_sources = result["sources_gathered"]
        self.assertEqual(len(final_sources), 2)
        self.assertTrue(any(s["value"] == "http://example.com/original_source_1" for s in final_sources))
        self.assertTrue(any(s["value"] == "http://example.com/original_source_3" for s in final_sources))
        self.assertFalse(any(s["value"] == "http://example.com/original_source_2" for s in final_sources))


    @patch('src.agent.graph.Configuration') # For Configuration.from_runnable_config
    @patch('src.agent.graph.get_research_topic') # For formatting the prompt
    @patch('src.agent.graph.get_current_date') # For formatting the prompt
    @patch('src.agent.graph.ChatGoogleGenerativeAI') # For the LLM call
    def test_reflection_success(self,
                                mock_ChatGoogleGenerativeAI,
                                mock_get_current_date,
                                mock_get_research_topic,
                                mock_Configuration):
        # --- Arrange ---
        # Mock Configuration
        mock_config_instance = MagicMock()
        mock_config_instance.reasoning_model = "gemini-experimental"
        mock_config_instance.llm_temperature = 0.8 # New value
        mock_config_instance.num_search_results = 4 # New value, though not used by reflection node
        mock_Configuration.from_runnable_config.return_value = mock_config_instance

        # Mock LLM (ChatGoogleGenerativeAI)
        mock_llm_instance = MagicMock()
        mock_structured_llm_instance = MagicMock()

        expected_llm_response = Reflection( # From tools_and_schemas
            is_sufficient=False,
            knowledge_gap="Need more details on X.",
            follow_up_queries=["What about X in Y context?"]
        )
        mock_structured_llm_instance.invoke.return_value = expected_llm_response
        mock_llm_instance.with_structured_output.return_value = mock_structured_llm_instance
        mock_ChatGoogleGenerativeAI.return_value = mock_llm_instance

        # Mock utility functions
        mock_get_current_date.return_value = "March 3, 2024"
        mock_get_research_topic.return_value = "User's Research Topic"

        # Prepare initial state for reflection node
        # ReflectionState is the output, OverallState is the input type hint,
        # but the function accesses specific fields from OverallState.
        initial_reflection_state: OverallState = {
            "messages": [("user", "Tell me about User's Research Topic")],
            "web_research_result": ["Summary 1 from web research.", "Summary 2 from web research."],
            "search_query": ["query1", "query2"], # Used for number_of_ran_queries
            "research_loop_count": 1, # Initial loop count
            # Other OverallState keys that might be needed if accessed, otherwise provide defaults
            "sources_gathered": [],
            "initial_search_query_count": 2,
            "max_research_loops": 5,
            "reasoning_model": None # To test default model from config
        }
        runnable_config = RunnableConfig(configurable={})

        # --- Act ---
        result = reflection(initial_reflection_state, runnable_config)

        # --- Assert ---
        mock_Configuration.from_runnable_config.assert_called_once_with(runnable_config)
        mock_get_current_date.assert_called_once()
        mock_get_research_topic.assert_called_once_with(initial_reflection_state["messages"])

        mock_ChatGoogleGenerativeAI.assert_called_once_with(
            model="gemini-experimental",
            temperature=0.8, # Check for new temperature
            max_retries=2,
            api_key='test_key_for_gemini'
        )
        mock_llm_instance.with_structured_output.assert_called_once_with(Reflection)

        expected_prompt_format = reflection_instructions.format( # Need to import reflection_instructions
            current_date="March 3, 2024",
            research_topic="User's Research Topic",
            summaries="Summary 1 from web research.\n\n---\n\nSummary 2 from web research."
        )
        mock_structured_llm_instance.invoke.assert_called_once_with(expected_prompt_format)

        self.assertEqual(result["is_sufficient"], False)
        self.assertEqual(result["knowledge_gap"], "Need more details on X.")
        self.assertEqual(result["follow_up_queries"], ["What about X in Y context?"])
        self.assertEqual(result["research_loop_count"], 2) # Incremented from 1
        self.assertEqual(result["number_of_ran_queries"], 2) # Length of initial_reflection_state["search_query"]

    @patch('src.agent.graph.Configuration')
    @patch('src.agent.graph.insert_citation_markers')
    @patch('src.agent.graph.get_citations')
    @patch('src.agent.graph.resolve_urls')
    @patch('src.agent.graph.get_current_date')
    @patch('src.agent.graph.initialize_genai_client')
    def test_web_research_success(self,
                                  mock_initialize_genai_client,
                                  mock_get_current_date,
                                  mock_resolve_urls,
                                  mock_get_citations,
                                  mock_insert_citation_markers,
                                  mock_Configuration):
        # --- Arrange ---
        # Mock Configuration
        mock_config_instance = MagicMock()
        mock_config_instance.query_generator_model = "gemini-pro-1.5"
        # llm_temperature is part of config but not used by web_research's LLM call directly
        mock_config_instance.llm_temperature = 0.6
        mock_config_instance.num_search_results = 3 # Default from config, will be overridden by state
        mock_Configuration.from_runnable_config.return_value = mock_config_instance

        # Mock initialize_genai_client and the client methods
        mock_genai_client = MagicMock()
        mock_initialize_genai_client.return_value = mock_genai_client

        mock_response = MagicMock()
        # Setup mock_response.candidates[0].grounding_metadata.grounding_chunks
        # Need to ensure this structure is correctly mocked.
        # response.candidates[0].grounding_metadata.grounding_chunks
        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata.grounding_chunks = [{"url": "http://example.com/source1"}] # Sample data
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Research result text."
        mock_genai_client.models.generate_content.return_value = mock_response

        # Mock utility functions
        mock_get_current_date.return_value = "February 2, 2024"
        mock_resolved_urls = [{"url": "http://example.com/source1", "short_url": "[1]"}] # Sample
        mock_resolve_urls.return_value = mock_resolved_urls

        expected_citations = [{"segments": [{"source_index": 1, "text_segment": "Segment text", "value": "http://example.com/source1"}]}] # Sample
        mock_get_citations.return_value = expected_citations

        expected_modified_text = "Research result text with [1]."
        mock_insert_citation_markers.return_value = expected_modified_text

        # Prepare state and config for web_research
        # Focusing on the case where num_search_results is provided in state
        web_search_state: WebSearchState = {
            "search_query": "What is the weather like?",
            "id": "query789",
            "num_search_results": 5 # Explicitly set in state
        }
        runnable_config = RunnableConfig(configurable={})

        # --- Act ---
        result = web_research(web_search_state, runnable_config)

        # --- Assert ---
        mock_initialize_genai_client.assert_called_once()
        mock_Configuration.from_runnable_config.assert_called_once_with(runnable_config)
        mock_get_current_date.assert_called_once() # Called once for this execution path

        expected_prompt_format = web_searcher_instructions.format(
            current_date="February 2, 2024", # Value from mock_get_current_date
            research_topic="What is the weather like?"
        )
        mock_genai_client.models.generate_content.assert_called_once_with(
            model="gemini-pro-1.5",
            contents=expected_prompt_format,
            config={
                "tools": [{"google_search": {"num_results": 5}}], # Asserting num_results from state
                "temperature": 0,
            },
        )

        mock_resolve_urls.assert_called_once_with(
            mock_response.candidates[0].grounding_metadata.grounding_chunks, "query789"
        )
        mock_get_citations.assert_called_once_with(mock_response, mock_resolved_urls)
        mock_insert_citation_markers.assert_called_once_with(mock_response.text, expected_citations)

        expected_sources_gathered = [item for citation in expected_citations for item in citation["segments"]]
        self.assertEqual(result["sources_gathered"], expected_sources_gathered)
        self.assertEqual(result["search_query"], ["What is the weather like?"])
        self.assertEqual(result["web_research_result"], [expected_modified_text])

def tearDownModule():
    patch_os_getenv.stop()
    # patch_google_client is no longer started, so no need to stop it here.

if __name__ == '__main__':
    unittest.main()
