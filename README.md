# Gemini Fullstack LangGraph Quickstart

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend agent. The agent is designed to perform comprehensive research on a user's query by dynamically generating search terms, querying the web using Google Search, reflecting on the results to identify knowledge gaps, and iteratively refining its search until it can provide a well-supported answer with citations. This application serves as an example of building research-augmented conversational AI using LangGraph and Google's Gemini models.

![Gemini Fullstack LangGraph](./app.png)

## Features

- 💬 Fullstack application with a React frontend and LangGraph backend.
- 🧠 Powered by a LangGraph agent for advanced research and conversational AI.
- 💡 **Multi-LLM Support:** Flexibility to use different LLM providers (Gemini, OpenRouter, DeepSeek).
- 🔍 Dynamic search query generation using the configured LLM.
- 🌐 Integrated web research via Google Search API.
- 🏠 **Local Network Search:** Optional capability to search within configured local domains.
- 🔄 **Flexible Search Modes:** Control whether to search internet, local network, or both, and in which order.
- 🤔 Reflective reasoning to identify knowledge gaps and refine searches.
- 📄 Generates answers with citations from gathered sources.
- 🎨 **Updated UI Theme:** Modern, light theme for improved readability and a professional look.
- 🛠️ **Configurable Tracing:** LangSmith tracing can be enabled/disabled.
- 🔄 Hot-reloading for both frontend and backend development during development.

### Upcoming Features
- Dedicated "Finance" and "HR" sections for specialized research tasks.

## Project Structure

The project is divided into two main directories:

-   `frontend/`: Contains the React application built with Vite.
-   `backend/`: Contains the LangGraph/FastAPI application, including the research agent logic.

## Getting Started: Development and Local Testing

Follow these steps to get the application running locally for development and testing.

**1. Prerequisites:**

-   Node.js and npm (or yarn/pnpm)
-   Python 3.8+
-   **API Keys & Configuration:** The backend agent requires API keys depending on the chosen LLM provider and other features. See the "Configuration" section below for details on setting up your `.env` file in the `backend/` directory.

**2. Install Dependencies:**

**Backend:**

```bash
cd backend
pip install .
```
*Note: If you plan to use the Local Network Search feature, ensure you install its dependencies:*
```bash
pip install ".[local_search]"
```
*(Or `pip install requests beautifulsoup4` if you manage dependencies manually)*

**Frontend:**

```bash
cd frontend
npm install
```

**3. Run Development Servers:**

**Backend & Frontend:**

```bash
make dev
```
This will run the backend and frontend development servers. Open your browser and navigate to the frontend development server URL (e.g., `http://localhost:5173/app`).

_Alternatively, you can run the backend and frontend development servers separately. For the backend, open a terminal in the `backend/` directory and run `langgraph dev`. The backend API will be available at `http://127.0.0.1:2024`. It will also open a browser window to the LangGraph UI. For the frontend, open a terminal in the `frontend/` directory and run `npm run dev`. The frontend will be available at `http://localhost:5173`._

## Configuration

Create a `.env` file in the `backend/` directory by copying `backend/.env.example`. Below are the available environment variables:

### Core Agent & LLM Configuration
-   `GEMINI_API_KEY`: Your Google Gemini API key. Required if using "gemini" as the LLM provider for any task or for Google Search functionality.
-   `LLM_PROVIDER`: Specifies the primary LLM provider for core agent tasks (query generation, reflection, answer synthesis).
    -   Options: `"gemini"`, `"openrouter"`, `"deepseek"`.
    -   Default: `"gemini"`.
-   `LLM_API_KEY`: The API key for the selected `LLM_PROVIDER`.
    -   Example: If `LLM_PROVIDER="openrouter"`, this should be your OpenRouter API key.
-   `OPENROUTER_MODEL_NAME`: Specify the full model string if using OpenRouter (e.g., `"anthropic/claude-3-haiku"`). This can be used by the agent if specific task models are not set.
-   `DEEPSEEK_MODEL_NAME`: Specify the model name if using DeepSeek (e.g., `"deepseek-chat"`). This can be used by the agent if specific task models are not set.
-   `QUERY_GENERATOR_MODEL`: Model used for generating search queries. Interpreted based on `LLM_PROVIDER`.
    -   Default for Gemini: `"gemini-1.5-flash"`
-   `REFLECTION_MODEL`: Model used for reflection and knowledge gap analysis. Interpreted based on `LLM_PROVIDER`.
    -   Default for Gemini: `"gemini-1.5-flash"`
-   `ANSWER_MODEL`: Model used for synthesizing the final answer. Interpreted based on `LLM_PROVIDER`.
    -   Default for Gemini: `"gemini-1.5-pro"`
-   `NUMBER_OF_INITIAL_QUERIES`: Number of initial search queries to generate. Default: `3`.
-   `MAX_RESEARCH_LOOPS`: Maximum number of research refinement loops. Default: `2`.

### LangSmith Tracing
-   `LANGSMITH_ENABLED`: Master switch to enable (`true`) or disable (`false`) LangSmith tracing for the backend. Default: `true`.
    -   If `true`, various LangSmith environment variables below should also be set.
    -   If `false`, tracing is globally disabled for the application process, and the UI toggle cannot override this.
-   `LANGCHAIN_API_KEY`: Your LangSmith API key. Required if `LANGSMITH_ENABLED` is true.
-   `LANGCHAIN_TRACING_V2`: Set to `"true"` to use the V2 tracing protocol. Usually managed by the `LANGSMITH_ENABLED` setting.
-   `LANGCHAIN_ENDPOINT`: LangSmith API endpoint. Defaults to `"https://api.smith.langchain.com"`.
-   `LANGCHAIN_PROJECT`: Name of the project in LangSmith.

### Local Network Search
-   `ENABLE_LOCAL_SEARCH`: Set to `true` to enable searching within local network domains. Default: `false`.
-   `LOCAL_SEARCH_DOMAINS`: A comma-separated list of base URLs or domains for local search.
    -   Example: `"http://intranet.mycompany.com,http://docs.internal.team"`
-   `SEARCH_MODE`: Defines the search behavior when both internet and local search capabilities might be active.
    -   `"internet_only"` (Default): Searches only the public internet.
    *   `"local_only"`: Searches only configured local domains (requires `ENABLE_LOCAL_SEARCH=true` and `LOCAL_SEARCH_DOMAINS` to be set).
    *   `"internet_then_local"`: Performs internet search first, then local search if enabled.
    *   `"local_then_internet"`: Performs local search first if enabled, then internet search.

## Frontend UI Settings

The user interface provides several controls to customize the agent's behavior for each query:

-   **Effort Level:** (Low, Medium, High) - Adjusts the number of initial queries and maximum research loops.
-   **Reasoning Model:** (Flash/Fast, Pro/Advanced) - Selects a class of model for reasoning tasks (reflection, answer synthesis). The actual model used depends on the selected LLM Provider.
-   **LLM Provider:** (Gemini, OpenRouter, DeepSeek) - Choose the primary LLM provider for the current query. Requires corresponding API keys to be configured on the backend.
-   **LangSmith Monitoring:** (Toggle Switch) - If LangSmith is enabled globally on the backend, this allows users to toggle tracing for their specific session/query.
-   **Search Scope:** (Internet Only, Local Only, Internet then Local, Local then Internet) - Defines where the agent should search for information. "Local" options require backend configuration for local search.

## How the Backend Agent Works (High-Level)

The core of the backend is a LangGraph agent defined in `backend/src/agent/graph.py`. It follows these steps:

![Agent Flow](./agent.png)

1.  **Configure:** Reads settings from environment variables and per-request UI selections.
2.  **Generate Initial Queries:** Based on your input and configured model, it generates initial search queries.
3.  **Web/Local Research:** Depending on the `SEARCH_MODE`:
    *   Performs searches using the Google Search API (for internet results).
    *   Performs searches using the custom `LocalSearchTool` against configured domains (for local results).
    *   Combines results if applicable.
4.  **Reflection & Knowledge Gap Analysis:** The agent analyzes the search results to determine if the information is sufficient or if there are knowledge gaps.
5.  **Iterative Refinement:** If gaps are found, it generates follow-up queries and repeats the research and reflection steps.
6.  **Finalize Answer:** Once research is sufficient, the agent synthesizes the information into a coherent answer with citations, using the configured answer model.

## Deployment

In production, the backend server serves the optimized static frontend build. LangGraph requires a Redis instance and a Postgres database. Redis is used as a pub-sub broker to enable streaming real time output from background runs. Postgres is used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. For more details on how to deploy the backend server, take a look at the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/). Below is an example of how to build a Docker image that includes the optimized frontend build and the backend server and run it via `docker-compose`.

_Note: For the docker-compose.yml example you need a LangSmith API key, you can get one from [LangSmith](https://smith.langchain.com/settings)._

_Note: If you are not running the docker-compose.yml example or exposing the backend server to the public internet, you update the `apiUrl` in the `frontend/src/App.tsx` file your host. Currently the `apiUrl` is set to `http://localhost:8123` for docker-compose or `http://localhost:2024` for development._

**1. Build the Docker Image:**

   Run the following command from the **project root directory**:
   ```bash
   docker build -t gemini-fullstack-langgraph -f Dockerfile .
   ```
**2. Run the Production Server:**

   Adjust the `docker-compose.yml` or your deployment environment to include all necessary environment variables as described in the "Configuration" section.
   Example:
   ```bash
   # Ensure your .env file (if used by docker-compose) or environment variables are set
   # e.g., GEMINI_API_KEY, LLM_PROVIDER, LLM_API_KEY, LANGSMITH_API_KEY (if LangSmith enabled), etc.
   docker-compose up
   ```

Open your browser and navigate to `http://localhost:8123/app/` to see the application. The API will be available at `http://localhost:8123`.

## Technologies Used

- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - For the frontend user interface.
- [Tailwind CSS](https://tailwindcss.com/) - For styling.
- [Shadcn UI](https://ui.shadcn.com/) - For components.
- [LangGraph](https://github.com/langchain-ai/langgraph) - For building the backend research agent.
- LLMs: [Google Gemini](https://ai.google.dev/models/gemini), and adaptable for others like [OpenRouter](https://openrouter.ai/), [DeepSeek](https://www.deepseek.com/).
- Search: Google Search API, Custom Local Network Search (Python `requests` & `BeautifulSoup`).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 