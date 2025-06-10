# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Backend setup
cd backend && pip install .

# Frontend setup  
cd frontend && npm install
```

### Development Servers
```bash
# Start both frontend and backend (recommended)
make dev

# Or start individually:
make dev-frontend  # Starts Vite dev server on localhost:5173
make dev-backend   # Starts LangGraph dev server on localhost:2024
```

### Backend Development
```bash
cd backend

# Run tests
make test
make test TEST_FILE=tests/unit_tests/specific_test.py

# Linting and formatting
make lint     # Run linters (ruff, mypy)
make format   # Format code with ruff
```

### Frontend Development
```bash
cd frontend

npm run dev      # Start development server
npm run build    # Build for production
npm run lint     # Run ESLint
```

## Architecture Overview

This is a full-stack research agent application with two main components:

### Backend (LangGraph Agent)
- **Core**: Research agent built with LangGraph that performs iterative web research
- **Agent Flow**: Query generation → Web research → Reflection → Answer finalization
- **State Management**: Uses TypedDict states (`OverallState`, `ReflectionState`, etc.) to track research progress
- **Key Files**:
  - `backend/src/agent/graph.py`: Main agent graph with nodes and edges
  - `backend/src/agent/state.py`: State definitions for the agent workflow
  - `backend/src/agent/configuration.py`: Configurable parameters
- **Models**: Uses Google Gemini models for query generation, web search, and reasoning
- **Web Search**: Integrates with Google Search API for research capabilities

### Frontend (React/TypeScript)
- **Framework**: React with Vite, TypeScript, and Tailwind CSS
- **Components**: Built with Shadcn UI components
- **Real-time Updates**: Uses `@langchain/langgraph-sdk/react` for streaming agent updates
- **Key Features**:
  - Live activity timeline showing agent progress
  - Chat interface with message history
  - Configurable research effort levels (low/medium/high)
  - Model selection (Gemini variants)

### Communication Flow
- Frontend connects to backend via LangGraph SDK
- Backend runs on port 2024 (dev) or 8123 (production)
- Real-time streaming of agent state updates to frontend
- Frontend displays agent progress through activity timeline

## Environment Requirements

### Backend
- Python 3.11+
- `GEMINI_API_KEY` environment variable required
- Dependencies managed via `pyproject.toml`

### Frontend  
- Node.js and npm
- Development server auto-detects backend URL based on environment

## Key Configuration

### Research Parameters
- `initial_search_query_count`: Number of initial search queries (1-5)
- `max_research_loops`: Maximum reflection/refinement loops (1-10)
- `reasoning_model`: Gemini model variant for reasoning tasks

### API Endpoints
- Development: `http://localhost:2024`
- Production: `http://localhost:8123`
- Frontend auto-configures based on `import.meta.env.DEV`

## Testing and Quality

### Backend
- Uses pytest for testing
- Ruff for formatting and linting
- MyPy for type checking
- Development dependencies include test watching capabilities

### Frontend
- ESLint with TypeScript support
- Vite for building and development
- React 19 with modern hooks and patterns