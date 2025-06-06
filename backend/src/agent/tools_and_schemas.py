from typing import List, Dict, Any, Type
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


# --- Local Search Tool Schemas and Implementation ---

class LocalSearchInput(BaseModel):
    query: str = Field(description="The search query to run on local domains.")
    local_domains: List[str] = Field(description="A list of base URLs/domains to search within.")

class LocalSearchResult(BaseModel):
    url: str = Field(description="The URL of the found content.")
    title: str = Field(description="The title of the page, if available.")
    snippet: str = Field(description="A short snippet of relevant text from the page.")

class LocalSearchOutput(BaseModel):
    results: List[LocalSearchResult] = Field(description="A list of search results from local domains.")

class LocalSearchTool(BaseTool):
    name: str = "local_network_search"
    description: str = (
        "Searches for information within a predefined list of local network domains/URLs. "
        "Input should be the search query and the list of domains to search."
    )
    args_schema: Type[BaseModel] = LocalSearchInput
    return_schema: Type[BaseModel] = LocalSearchOutput

    def _run(self, query: str, local_domains: List[str], **kwargs: Any) -> LocalSearchOutput:
        all_results: List[LocalSearchResult] = []
        query_lower = query.lower()

        for domain_url in local_domains:
            try:
                # For simplicity, trying HTTP first, then HTTPS if it fails or not specified
                if not domain_url.startswith(('http://', 'https://')):
                    try_urls = [f"http://{domain_url}", f"https://{domain_url}"]
                else:
                    try_urls = [domain_url]

                response = None
                for url_to_try in try_urls:
                    try:
                        response = requests.get(url_to_try, timeout=5, allow_redirects=True)
                        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                        if response.status_code == 200:
                            break # Success
                    except requests.RequestException:
                        response = None # Ensure response is None if this attempt fails
                        continue # Try next URL if current one fails

                if not response or response.status_code != 200:
                    print(f"Failed to fetch content from {domain_url} after trying variants.")
                    continue

                content_type = response.headers.get("content-type", "").lower()
                if "html" not in content_type:
                    print(f"Skipping non-HTML content at {response.url}")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract all text
                page_text = soup.get_text(separator=" ", strip=True)
                page_text_lower = page_text.lower()

                # Search for query in text
                found_index = page_text_lower.find(query_lower)

                if found_index != -1:
                    title = soup.title.string.strip() if soup.title else "No title found"

                    # Create snippet
                    snippet_start = max(0, found_index - 100)
                    snippet_end = min(len(page_text), found_index + len(query) + 100)
                    snippet = page_text[snippet_start:snippet_end]

                    # Add ... if snippet is truncated
                    if snippet_start > 0:
                        snippet = "... " + snippet
                    if snippet_end < len(page_text):
                        snippet = snippet + " ..."

                    all_results.append(
                        LocalSearchResult(
                            url=response.url,
                            title=title,
                            snippet=snippet,
                        )
                    )

            except requests.RequestException as e:
                print(f"Error fetching {domain_url}: {e}")
            except Exception as e:
                print(f"Error processing {domain_url}: {e}")

        return LocalSearchOutput(results=all_results)

    async def _arun(self, query: str, local_domains: List[str], **kwargs: Any) -> LocalSearchOutput:
        # For now, just wrapping the sync version.
        # For a truly async version, would use an async HTTP client like aiohttp.
        # This is okay for now as LangGraph can run sync tools in a thread pool.
        return self._run(query, local_domains, **kwargs)
