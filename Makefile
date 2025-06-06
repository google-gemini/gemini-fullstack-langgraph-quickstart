.PHONY: help setup dev-frontend dev-backend dev

setup:
	@echo "Installing frontend dependencies..."
	@cd frontend && npm install
	@echo "Installing backend dependencies..."
	@cd backend && uv sync

help:
	@echo "Available commands:"
	@echo "  make dev-frontend    - Starts the frontend development server (Vite)"
	@echo "  make dev-backend     - Starts the backend development server (Uvicorn with reload)"
	@echo "  make dev             - Starts both frontend and backend development servers"

dev-frontend:
	@echo "Starting frontend development server..."
	@cd frontend && npm run dev

dev-backend:
	@echo "Starting backend development server..."
	@cd backend && uv run langgraph dev

# Run frontend and backend concurrently
dev:
	@echo "Starting both frontend and backend development servers..."
	@make dev-frontend & make dev-backend
