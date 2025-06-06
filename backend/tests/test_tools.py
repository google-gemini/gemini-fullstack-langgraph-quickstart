import pytest
from unittest.mock import patch, MagicMock

import requests

# Module to test
from agent.tools_and_schemas import LocalSearchTool, LocalSearchInput, LocalSearchResult, LocalSearchOutput


class TestLocalSearchTool:

    @pytest.fixture
    def local_search_tool_instance(self):
        return LocalSearchTool()

    @patch('agent.tools_and_schemas.requests.get')
    def test_run_successful_search_query_found(self, mock_requests_get, local_search_tool_instance):
        # Mock requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = """
        <html>
            <head><title>Test Page Title</title></head>
            <body>
                <p>Some content here. The quick brown fox jumps over the lazy dog.</p>
                <p>This page contains the test_query we are looking for.</p>
                <p>More content after the query.</p>
            </body>
        </html>
        """
        mock_response.url = "http://testdomain.com/page"
        mock_requests_get.return_value = mock_response

        tool_input = LocalSearchInput(query="test_query", local_domains=["http://testdomain.com"])
        result = local_search_tool_instance._run(**tool_input.model_dump())

        mock_requests_get.assert_called_once_with("http://testdomain.com", timeout=5, allow_redirects=True)
        assert len(result.results) == 1
        search_result = result.results[0]
        assert search_result.url == "http://testdomain.com/page" # requests.get might update the URL due to redirects
        assert search_result.title == "Test Page Title"
        assert "test_query" in search_result.snippet
        assert "... page contains the test_query we are looking for. More content ..." in search_result.snippet
        assert search_result.snippet.startswith("... ")
        assert search_result.snippet.endswith(" ...")


    @patch('agent.tools_and_schemas.requests.get')
    def test_run_query_not_found(self, mock_requests_get, local_search_tool_instance):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><body><p>Some other content without the query.</p></body></html>"
        mock_response.url = "http://testdomain.com/other"
        mock_requests_get.return_value = mock_response

        tool_input = LocalSearchInput(query="missing_query", local_domains=["http://testdomain.com"])
        result = local_search_tool_instance._run(**tool_input.model_dump())

        assert len(result.results) == 0

    @patch('agent.tools_and_schemas.requests.get')
    def test_run_http_then_https_try(self, mock_requests_get, local_search_tool_instance):
        # First call (http) fails, second (https) succeeds
        mock_http_response_fail = MagicMock(spec=requests.Response) # Use spec for attribute checking
        mock_http_response_fail.status_code = 500 # Simulate server error
        mock_http_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")

        mock_https_response_success = MagicMock(spec=requests.Response)
        mock_https_response_success.status_code = 200
        mock_https_response_success.headers = {"content-type": "text/html"}
        mock_https_response_success.text = "<html><title>Secure Page</title><body>Secure query found</body></html>"
        mock_https_response_success.url = "https://secure.com"

        # Configure side_effect to simulate different responses for different calls
        mock_requests_get.side_effect = [
            requests.exceptions.RequestException("Connection failed for http"), # for http://domain.com
            mock_https_response_success # for https://domain.com
        ]

        tool_input = LocalSearchInput(query="query", local_domains=["domain.com"]) # No scheme
        result = local_search_tool_instance._run(**tool_input.model_dump())

        assert mock_requests_get.call_count == 2
        mock_requests_get.assert_any_call("http://domain.com", timeout=5, allow_redirects=True)
        mock_requests_get.assert_any_call("https://domain.com", timeout=5, allow_redirects=True)

        assert len(result.results) == 1
        assert result.results[0].url == "https://secure.com"
        assert result.results[0].title == "Secure Page"

    @patch('agent.tools_and_schemas.requests.get')
    def test_run_non_html_content(self, mock_requests_get, local_search_tool_instance):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"} # Non-HTML
        mock_response.text = "{'data': 'not html'}"
        mock_response.url = "http://testdomain.com/api"
        mock_requests_get.return_value = mock_response

        tool_input = LocalSearchInput(query="any_query", local_domains=["http://testdomain.com"])
        result = local_search_tool_instance._run(**tool_input.model_dump())

        assert len(result.results) == 0

    @patch('agent.tools_and_schemas.requests.get')
    def test_run_request_exception(self, mock_requests_get, local_search_tool_instance):
        mock_requests_get.side_effect = requests.exceptions.RequestException("Test connection error")

        tool_input = LocalSearchInput(query="any_query", local_domains=["http://error.domain.com"])
        result = local_search_tool_instance._run(**tool_input.model_dump())

        assert len(result.results) == 0
        # Optionally, check logs if your tool logs errors, but here we just check output

    def test_run_empty_domains_list(self, local_search_tool_instance):
        tool_input = LocalSearchInput(query="any_query", local_domains=[])
        result = local_search_tool_instance._run(**tool_input.model_dump())
        assert len(result.results) == 0

    @patch('agent.tools_and_schemas.requests.get')
    def test_snippet_generation_edges(self, mock_requests_get, local_search_tool_instance):
        # Query at the beginning
        mock_response_start = MagicMock()
        mock_response_start.status_code = 200
        mock_response_start.headers = {"content-type": "text/html"}
        mock_response_start.text = "<html><body>test_query is at the start of this very long text that will surely exceed 100 characters for the snippet generation to show truncation at the end.</body></html>"
        mock_response_start.url = "http://edge.com/start"

        # Query at the end
        mock_response_end = MagicMock()
        mock_response_end.status_code = 200
        mock_response_end.headers = {"content-type": "text/html"}
        mock_response_end.text = "<html><body>This very long text that will surely exceed 100 characters for the snippet generation to show truncation at the beginning ends with the test_query.</body></html>"
        mock_response_end.url = "http://edge.com/end"

        mock_requests_get.side_effect = [mock_response_start, mock_response_end]

        tool_input_start = LocalSearchInput(query="test_query", local_domains=["http://edge.com/start"])
        result_start = local_search_tool_instance._run(**tool_input_start.model_dump())
        assert len(result_start.results) == 1
        assert result_start.results[0].snippet.startswith("test_query")
        assert result_start.results[0].snippet.endswith(" ...")

        tool_input_end = LocalSearchInput(query="test_query", local_domains=["http://edge.com/end"])
        result_end = local_search_tool_instance._run(**tool_input_end.model_dump())
        assert len(result_end.results) == 1
        assert result_end.results[0].snippet.startswith("... ")
        assert result_end.results[0].snippet.endswith("test_query.") # Period from original text included

    @patch('agent.tools_and_schemas.requests.get')
    def test_run_multiple_domains_mixed_results(self, mock_requests_get, local_search_tool_instance):
        mock_res1_found = MagicMock()
        mock_res1_found.status_code = 200
        mock_res1_found.headers = {"content-type": "text/html"}
        mock_res1_found.text = "<html><title>Page 1</title><body>query_here for all.</body></html>"
        mock_res1_found.url = "http://domain1.com"

        mock_res2_not_found = MagicMock()
        mock_res2_not_found.status_code = 200
        mock_res2_not_found.headers = {"content-type": "text/html"}
        mock_res2_not_found.text = "<html><title>Page 2</title><body>Nothing relevant.</body></html>"
        mock_res2_not_found.url = "http://domain2.com"

        mock_res3_error = requests.exceptions.RequestException("Failed domain3")

        mock_res4_found_again = MagicMock()
        mock_res4_found_again.status_code = 200
        mock_res4_found_again.headers = {"content-type": "text/html"}
        mock_res4_found_again.text = "<html><title>Page 4</title><body>Another query_here.</body></html>"
        mock_res4_found_again.url = "http://domain4.com"


        mock_requests_get.side_effect = [mock_res1_found, mock_res2_not_found, mock_res3_error, mock_res4_found_again]

        tool_input = LocalSearchInput(
            query="query_here",
            local_domains=["http://domain1.com", "http://domain2.com", "http://domain3.com", "http://domain4.com"]
        )
        result = local_search_tool_instance._run(**tool_input.model_dump())

        assert mock_requests_get.call_count == 4
        assert len(result.results) == 2
        assert result.results[0].url == "http://domain1.com"
        assert result.results[0].title == "Page 1"
        assert "query_here" in result.results[0].snippet

        assert result.results[1].url == "http://domain4.com"
        assert result.results[1].title == "Page 4"
        assert "query_here" in result.results[1].snippet

    # Test for _arun if it were truly async, but it currently wraps _run
    async def test_arun_wrapper(self, local_search_tool_instance, mocker):
        # Mock the synchronous _run method
        mock_sync_run_result = LocalSearchOutput(results=[
            LocalSearchResult(url="http://async.com", title="Async Test", snippet="Async snippet")
        ])
        mocker.patch.object(local_search_tool_instance, '_run', return_value=mock_sync_run_result)

        tool_input = LocalSearchInput(query="async_query", local_domains=["http://async.com"])
        # Since _arun directly calls _run, we test it by calling _arun
        # In a real async test with an async http client, this would be different.
        result = await local_search_tool_instance._arun(**tool_input.model_dump())

        local_search_tool_instance._run.assert_called_once_with(query="async_query", local_domains=["http://async.com"])
        assert result == mock_sync_run_result

```
