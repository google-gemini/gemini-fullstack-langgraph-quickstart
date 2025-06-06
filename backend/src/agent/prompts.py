from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Generate search queries for the following research topic and provide a brief rationale for why these queries are relevant.

Research Topic: {research_topic}"""


web_searcher_instructions = """Analyze and synthesize information from the provided search results to create a comprehensive summary about "{research_topic}".

Instructions:
- The current date is {current_date}.
- Carefully review each search result and its source.
- Focus on extracting factual, verifiable information from credible sources.
- When citing information, use the provided citation markers in square brackets (e.g., [1], [2], etc.).
- Organize the information logically and create a coherent narrative.
- Only include information found in the search results, don't make up any information.

Search Results:
{search_results}

Please provide a well-structured summary that incorporates the relevant information with proper citations."""


reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Your task is to:
1. Determine if the provided summaries are sufficient to answer the user's question
2. If not, identify specific knowledge gaps or areas that need deeper exploration
3. Generate follow-up queries to address any gaps

Guidelines:
- If summaries are sufficient, set is_sufficient to true and leave knowledge_gap empty and follow_up_queries as an empty list
- If there are gaps, generate 1-3 specific follow-up queries that would help expand understanding
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
- Ensure follow-up queries are self-contained and include necessary context for web search

Research Topic: {research_topic}
Current Date: {current_date}

Summaries:
{summaries}"""


answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- Generate a well-structured answer that directly addresses the user's question.
- Include specific facts and details from your research, always with proper citation markers.
- Be clear and concise while ensuring accuracy and comprehensiveness.
- When citing sources, ensure you maintain the citation markers (e.g., [1], [2], etc.).

User Context:
- {research_topic}

Summaries:
{summaries}"""
