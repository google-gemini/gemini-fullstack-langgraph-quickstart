# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files and install dependencies
COPY frontend/package.json ./
COPY frontend/package-lock.json ./
# If you use yarn or pnpm, adjust accordingly (e.g., copy yarn.lock or pnpm-lock.yaml and use yarn install or pnpm install)
RUN npm install

# Copy the rest of the frontend source code
COPY frontend/ ./

# Build the frontend
RUN npm run build

# Stage 2: Python Backend
FROM docker.io/langchain/langgraph-api:3.11

# -- Install UV --
# First install curl, then install UV using the standalone installer
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"
# -- End of UV installation --

# -- Copy built frontend from builder stage --
# The app.py expects the frontend build to be at ../frontend/dist relative to its own location.
# If app.py is at /deps/backend/src/agent/app.py, then ../frontend/dist resolves to /deps/frontend/dist.
COPY --from=frontend-builder /app/frontend/dist /deps/frontend/dist
# -- End of copying built frontend --

# -- Adding local package . --
ADD backend/ /deps/backend
ADD accounting_data.csv /deps/backend/accounting_data.csv
# -- End of local package . --

# -- Installing all local dependencies using UV --
# First, we need to ensure pip is available for UV to use
RUN uv pip install --system pip setuptools wheel
# Install dependencies with UV, respecting constraints
RUN cd /deps/backend && \
    PYTHONDONTWRITEBYTECODE=1 UV_SYSTEM_PYTHON=1 uv pip install --system -c /api/constraints.txt -e .
# -- End of local dependencies install --
ENV LANGGRAPH_HTTP='{"app": "/deps/backend/src/agent/app.py:app"}'
ENV LANGSERVE_GRAPHS='{"agent": "/deps/backend/src/agent/graph.py:graph"}'

# -- Ensure user deps didn't inadvertently overwrite langgraph-api
# Create all required directories that the langgraph-api package expects
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license /api/langgraph_storage && \
    touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py /api/langgraph_storage/__init__.py
# Use pip for this specific package as it has poetry-based build requirements
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing pip from the final image (but keeping UV) --
RUN uv pip uninstall --system pip setuptools wheel && \
    rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && \
    find /usr/local/bin -name "pip*" -delete
# -- End of pip removal --

WORKDIR /deps/backend
