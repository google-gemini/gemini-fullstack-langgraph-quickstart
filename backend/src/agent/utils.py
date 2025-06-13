from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    resolved_map = {}
    for idx, site in enumerate(urls_to_resolve):
        url = site.web.uri
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"
    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segment_string' (the marker to insert).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by their 'end_index' in descending order.
    # If 'end_index' values are identical, then sort by 'start_index' in descending order.
    # This reverse sorting strategy is crucial: by processing citations from the
    # end of the text towards the beginning, we ensure that inserting a marker
    # does not shift the character positions of text segments for subsequent citations
    # that are yet to be processed. This way, the original start/end indices
    # remain valid for all unprocessed citations.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # The 'end_idx' (and implicitly 'start_idx' due to the sorting) refers to
        # character positions in the *original* text. Because we process from the end,
        # these original indices remain accurate for slicing the `modified_text` string
        # at the correct point, relative to already processed (later) parts of the string.
        end_idx = citation_info["end_index"]

        # Initialize an empty string to build the complete citation marker.
        # This marker will typically consist of one or more markdown-formatted links,
        # e.g., " [Source 1](url1) [Source 2](url2)".
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"

        # Insert the composed citation marker(s) at the 'end_index' of the cited segment.
        # This places the marker(s) immediately after the text content they refer to.
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.

    This function processes the grounding metadata provided in the response to
    construct a list of citation objects. Each citation object includes the
    start and end indices of the text segment it refers to, and a string
    containing formatted markdown links to the supporting web chunks.

    Args:
        response: The response object from the Gemini model, expected to have
                  a structure including `candidates[0].grounding_metadata`.
        resolved_urls_map (Dict[str, str]): A dictionary mapping original long URLs
                                           to their pre-generated short unique IDs.
                                           This is used to create consistent, shorter
                                           links in the citation text.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
    citations = []

    # Ensure response and necessary nested structures are present
    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        # Each 'citation' dict will store start/end indices of the text segment
        # being cited, and a list of 'segments' which are markdown-formatted links
        # to the supporting web chunks.
        citation = {}

        # Ensure segment information is present
        if not hasattr(support, "segment") or support.segment is None:
            continue  # Skip this support if segment info is missing

        # Determine the start index of the citation.
        # Defaults to 0 if not specified, assuming citation starts from the
        # beginning of the text if no explicit start is given by the API.
        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # Ensure end_index is present to form a valid segment
        if support.segment.end_index is None:
            continue  # Skip if end_index is missing, as it's crucial

        # Add 1 to end_index to make it an exclusive end for slicing/range purposes
        # (assuming the API provides an inclusive end_index)
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for chunk_index in support.grounding_chunk_indices:
                # Attempt to retrieve chunk details and its corresponding short URL.
                # This handles cases where chunk data might be malformed,
                # an index might be out of bounds, web attribute is missing,
                # or the URL not found in the resolved_urls_map.
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[chunk_index]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # For simplicity, skip adding this particular segment link if an error occurs.
                    # In a production system, this error might be logged for debugging.
                    pass
        citations.append(citation)
    return citations


import os
from langchain_google_genai import ChatGoogleGenerativeAI

def create_llm(model_name: str, temperature: float, max_retries: int, api_key: str):
    """Initializes and returns a ChatGoogleGenerativeAI instance.

    Args:
        model_name: The name of the model to use.
        temperature: The temperature to use for generation.
        max_retries: The maximum number of retries for the API call.
        api_key: The API key for the Google Generative AI service.

    Returns:
        A ChatGoogleGenerativeAI instance.
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_retries=max_retries,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
