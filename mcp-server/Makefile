# Makefile for Gemini Research Agent MCP Server
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-coverage lint format type-check clean run setup docs build publish

# Default target
help:
	@echo "Gemini Research Agent MCP Server - Available Commands:"
	@echo ""
	@echo "Development Commands:"
	@echo "  setup          - Complete development environment setup"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  run            - Run the MCP server"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  test           - Run all tests"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black"
	@echo "  type-check     - Run type checking with mypy"
	@echo "  check-all      - Run all code quality checks"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  build          - Build distribution packages"
	@echo "  publish        - Publish to PyPI (requires auth)"
	@echo "  clean          - Clean build artifacts"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs           - Generate documentation"
	@echo ""

# Development Environment Setup
setup: install-dev
	@echo "Setting up development environment..."
	@cp env.example .env 2>/dev/null || echo "Please create .env from env.example"
	@echo "Development environment ready!"
	@echo "Don't forget to set your GEMINI_API_KEY in .env"

install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt

install-dev: install
	@echo "Installing development dependencies..."
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy build twine
	pip install -e .

# Code Quality
test:
	@echo "Running tests..."
	pytest test_server.py -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest test_server.py --cov=server --cov-report=html --cov-report=term-missing -v
	@echo "Coverage report generated in htmlcov/"

lint:
	@echo "Running linting checks..."
	flake8 server.py test_server.py --max-line-length=100 --extend-ignore=E203,W503

format:
	@echo "Formatting code with black..."
	black server.py test_server.py setup.py --line-length=100

type-check:
	@echo "Running type checks..."
	mypy server.py --ignore-missing-imports --no-strict-optional

check-all: format lint type-check test
	@echo "All code quality checks completed!"

# Server Operations
run:
	@echo "Starting Gemini Research Agent MCP Server..."
	@if [ ! -f .env ]; then echo "Warning: .env file not found. Please copy from env.example"; fi
	python server.py

# Development Utilities
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean completed!"

# Build and Distribution
build: clean
	@echo "Building distribution packages..."
	python -m build

publish: build
	@echo "Publishing to PyPI..."
	@echo "Make sure you have configured your PyPI credentials!"
	python -m twine upload dist/*

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "# API Documentation" > docs.md
	@echo "" >> docs.md
	@echo "Auto-generated from server resource:" >> docs.md
	@echo "\`\`\`" >> docs.md
	@python -c "import asyncio; from server import get_research_documentation; print(asyncio.run(get_research_documentation()))" >> docs.md
	@echo "\`\`\`" >> docs.md
	@echo "Documentation generated in docs.md"

# Development Helpers
dev-setup: setup
	@echo "Development environment fully configured!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env and add your GEMINI_API_KEY"
	@echo "2. Run 'make test' to verify everything works"
	@echo "3. Run 'make run' to start the server"

# Continuous Integration Targets
ci-test: install-dev check-all test-coverage
	@echo "CI pipeline completed successfully!"

# Release Preparation
pre-release: clean check-all test-coverage build
	@echo "Pre-release checks completed!"
	@echo "Ready for release. Run 'make publish' to upload to PyPI."

# Quick Commands
quick-test:
	@pytest test_server.py::TestUtilityFunctions -v

quick-format:
	@black server.py --line-length=100

# Environment Information
info:
	@echo "Environment Information:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git status:"
	@git status --short || echo "Not a git repository"

# Security and Dependencies
security-check:
	@echo "Checking for security vulnerabilities..."
	@pip install safety
	@safety check

update-deps:
	@echo "Updating dependencies..."
	@pip install --upgrade pip
	@pip install --upgrade -r requirements.txt

# Docker Support (if needed)
docker-build:
	@echo "Building Docker image..."
	@if [ -f Dockerfile ]; then \
		docker build -t gemini-research-mcp .; \
	else \
		echo "Dockerfile not found. Create one for Docker support."; \
	fi

docker-run: docker-build
	@echo "Running Docker container..."
	@docker run -p 8000:8000 --env-file .env gemini-research-mcp 