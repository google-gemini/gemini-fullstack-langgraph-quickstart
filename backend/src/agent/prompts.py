from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """**IMPORTANT: Respond in the same language as the user's input.**

Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format:
- Format your response as a JSON object with ALL three of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings.
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""

reflection_instructions = """**IMPORTANT: Respond in the same language as the user's input.**

You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """**IMPORTANT: Respond in the same language as the user's input.**

Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step.
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- you MUST include all the citations from the summaries in the answer correctly.

User Context:
- {research_topic}

Summaries:
{summaries}"""

confirmation_question_instructions = """You are an advanced research-specialized AI assistant.

**IMPORTANT: Respond in the same language as the user's input.**

User's research request: {research_topic}

To provide more accurate and useful information, you need to analyze this request and confirm technical details and specific requirements.

Analysis perspectives:
1. **Technology Stack**: Specific versions and configurations of technologies, tools, and frameworks being used
2. **Environment and Constraints**: Execution environment (cloud/on-premise), resource constraints, integration with existing systems
3. **Specific Requirements**: Quantities (number of clusters, users, etc.), performance requirements, security requirements
4. **Implementation Details**: Usage of specific features, configuration details, customization needs
5. **Goals and Deliverables**: Final objectives, expected outcomes, target audience or users

Requirements for confirmation questions:
- Ask for specific information needed for implementation regarding technical elements in the request
- If there are multiple important confirmation items, organize them with bullet points into a single question
- When there are technical choices, provide specific examples to prompt selection
- Clarify ambiguous parts or parts that can be interpreted in multiple ways

Output format:
- "question": Confirmation question (specific question including technical details)
- "rationale": Why this question is important (from a technical perspective)
- "skip_instruction": Brief instruction explaining why you're asking this question and how the user can skip it (e.g., "This helps me provide more accurate information. You can say 'answer immediately' to skip.")

Example 1:
User request: "I want to create an e-commerce site with React and Next.js"
```json
{{
    "question": "Could you tell me about the following points regarding the e-commerce site implementation?\\n\\n1. **Payment System**: Are you planning to use Stripe, PayPal, or domestic payment services?\\n2. **Product Management**: What's the scale of products (tens to tens of thousands) and do you plan to use a CMS (Contentful, Sanity, etc.)?\\n3. **Authentication Method**: Which authentication system are you considering - NextAuth.js, Auth0, Firebase Auth, etc.?\\n4. **Deployment Environment**: Are you planning to use Vercel, AWS, or another platform?\\n5. **Performance Requirements**: Do you have targets for concurrent users or page load times?",
    "rationale": "E-commerce implementation requires many technical decisions including payment, inventory management, authentication, and performance. Understanding these details allows me to provide specific implementation methods and best practices.",
    "skip_instruction": "This question helps me provide more accurate and tailored implementation guidance. You can say 'answer immediately without questions' to skip confirmation."
}}
```

Example 2:
User request: "Create multi-cluster with kind + k0smotron and create multiple hosted control planes with CAPI"
```json
{{
    "question": "To create better documentation and blog posts for this advanced multi-cluster configuration, could you tell me about the following points?\\n\\n1. **Infrastructure Environment**: Are you planning a local (kind) verification environment or a production environment on cloud (AWS/GCP/Azure)?\\n2. **CAPI Provider**: Which infrastructure provider for Cluster API (Docker, vSphere, AWS, etc.) and what's the number and role distribution of hosted control planes?\\n3. **Network Configuration**: Any plans to use Service Mesh (Istio, etc.) and what's the inter-cluster network connection method (VPN, dedicated line, etc.)?\\n4. **Envoy Gateway Usage**: What's the specific use case - API Gateway, gRPC proxy, L7 load balancer, etc.?\\n5. **Projectsveltos Scope**: Are you planning specific namespace/service only or cross-cluster configuration management?\\n6. **Target Audience**: Technical details for Kubernetes experts or implementation procedures for intermediate users?",
    "rationale": "This configuration is an advanced architecture combining multiple CNCF projects. Specific configurations and integration methods for each component vary greatly depending on environment and requirements, so understanding these details helps create practical documentation.",
    "skip_instruction": "These questions help me create more accurate and practical documentation. You can say 'answer immediately' to skip this confirmation."
}}
```
"""
