# AWS Bedrock Fullstack LangGraph Quickstart

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend agent. The agent leverages AWS Bedrock foundation models to perform comprehensive research on user queries, dynamically generating search terms, performing web searches, and iteratively refining its research until providing well-supported answers with citations. This application serves as an example of building research-augmented conversational AI using LangGraph and AWS Bedrock.

![AWS Bedrock Fullstack LangGraph](./app.png)

## Features

- 💬 Fullstack application with a React frontend and LangGraph backend
- 🧠 Powered by AWS Bedrock foundation models for advanced conversational AI
- 🔍 Dynamic search query generation using AWS Bedrock Claude
- 🌐 Integrated web research capabilities powered by AWS
- 🤔 Reflective reasoning using AWS Bedrock models to identify knowledge gaps
- 📄 Answer synthesis with citations using AWS Bedrock
- 🔄 Hot-reloading for both frontend and backend development

## Project Structure

The project is divided into two main directories:

-   `frontend/`: Contains the React application built with Vite
-   `backend/`: Contains the LangGraph/FastAPI application leveraging AWS Bedrock

## Getting Started: Development and Local Testing

Follow these steps to get the application running locally for development and testing.

**1. Prerequisites:**

-   Node.js and npm (or yarn/pnpm)
-   Python 3.8+
-   uv (https://docs.astral.sh/uv/)
-   **AWS Credentials:**
    1.  Navigate to the `backend/` directory
    2.  Create a file named `.env` by copying the `backend/.env.example` file
    3.  Add your AWS credentials:
        ```
        AWS_ACCESS_KEY_ID=your_access_key
        AWS_SECRET_ACCESS_KEY=your_secret_key
        AWS_REGION=your_region
        ```

**2. Install Dependencies:**

```bash
make setup
```

**3. Run Development Servers:**

**Backend & Frontend:**

```bash
make dev
```
This will run the backend and frontend development servers. Open your browser and navigate to the frontend development server URL (e.g., `http://localhost:5173/app`).

_Alternatively, you can run the backend and frontend development servers separately. For the backend, open a terminal in the `backend/` directory and run `langgraph dev`. The backend API will be available at `http://127.0.0.1:2024`. It will also open a browser window to the LangGraph UI. For the frontend, open a terminal in the `frontend/` directory and run `npm run dev`. The frontend will be available at `http://localhost:5173`._

## How the Backend Agent Works (High-Level)

The core of the backend is a LangGraph agent defined in `backend/src/agent/graph.py`. It leverages AWS Bedrock foundation models at each step:

![Agent Flow](./agent.png)

1.  **Generate Initial Queries:** Uses AWS Bedrock Claude to analyze the input and generate targeted search queries
2.  **Web Research:** Performs web searches using AWS capabilities to find relevant information
3.  **Reflection & Knowledge Gap Analysis:** Uses AWS Bedrock Claude to analyze search results and identify knowledge gaps
4.  **Iterative Refinement:** Generates follow-up queries using AWS Bedrock Claude and repeats research if needed
5.  **Answer Synthesis:** Uses AWS Bedrock Claude to create a comprehensive answer with citations

## Deployment

In production, the backend server serves the optimized static frontend build. LangGraph requires a Redis instance and a Postgres database. Redis is used as a pub-sub broker to enable streaming real time output from background runs. Postgres is used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. For more details on how to deploy the backend server, take a look at the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/). Below is an example of how to build a Docker image that includes the optimized frontend build and the backend server and run it via `docker-compose`.

_Note: For the docker-compose.yml example you need a LangSmith API key, you can get one from [LangSmith](https://smith.langchain.com/settings)._

_Note: If you are not running the docker-compose.yml example or exposing the backend server to the public internet, you update the `apiUrl` in the `frontend/src/App.tsx` file your host. Currently the `apiUrl` is set to `http://localhost:8123` for docker-compose or `http://localhost:2024` for development._

**1. Build the Docker Image:**

   Run the following command from the **project root directory**:
   ```bash
   docker build -t aws-bedrock-fullstack-langgraph -f Dockerfile .
   ```
**2. Run the Production Server:**

   ```bash
   AWS_ACCESS_KEY_ID=your_access_key AWS_SECRET_ACCESS_KEY=your_secret_key AWS_REGION=your_region LANGSMITH_API_KEY=your_langsmith_api_key docker-compose up
   ```

Open your browser and navigate to `http://localhost:8123/app/` to see the application. The API will be available at `http://localhost:8123`.

## Technologies Used

- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - For the frontend user interface
- [Tailwind CSS](https://tailwindcss.com/) - For styling
- [Shadcn UI](https://ui.shadcn.com/) - For components
- [LangGraph](https://github.com/langchain-ai/langgraph) - For building the backend research agent
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - Foundation models for query generation, reflection, and answer synthesis

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.