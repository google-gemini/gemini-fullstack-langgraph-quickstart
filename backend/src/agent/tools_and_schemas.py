from typing import List
from pydantic import BaseModel, Field


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


class ConfirmationQuestion(BaseModel):
    """Schema for structured output of confirmation questions"""
    question: str = Field(description="Confirmation question for the user")
    rationale: str = Field(description="Reason why this question is important")
    skip_instruction: str = Field(description="Brief instruction on how to skip confirmation, explaining why this question is being asked")
