# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

- Frontend: React application with TypeScript, Vite, and Tailwind CSS
- Backend: Python-based LangGraph agent leveraging AWS Bedrock foundation models for iterative research
- Agent Flow:
  1. Uses AWS Bedrock Claude to generate search queries from user input
  2. Performs web research via AWS search capabilities
  3. Uses AWS Bedrock Claude to reflect on results and identify knowledge gaps
  4. Uses AWS Bedrock Claude for iterative query refinement
  5. Uses AWS Bedrock Claude to synthesize answers with citations

## Development Commands

Initial Setup:
```bash
make setup       # Install all dependencies for frontend and backend
```

Development:
```bash
make dev         # Run both frontend and backend dev servers 
npm run build    # Build frontend for production (in frontend/)
npm run lint     # Run frontend ESLint
ruff check .     # Run backend linter (in backend/)
mypy .          # Run backend type checker (in backend/)
```

## Environment Setup

Required environment variables:
- AWS_ACCESS_KEY_ID: AWS access key for Bedrock services
- AWS_SECRET_ACCESS_KEY: AWS secret access key for Bedrock services 
- AWS_REGION: AWS region where Bedrock models are deployed (e.g., us-west-2)
- LANGSMITH_API_KEY: LangSmith API key (for production)