# Agent-to-Agent (A2A) Communication System - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Roles and Responsibilities](#component-roles-and-responsibilities)
4. [Request Flow](#request-flow)
5. [Azure AI Foundry Agent Service v2 Integration](#azure-ai-foundry-agent-service-v2-integration)
6. [Agent-to-Agent (A2A) Protocol](#agent-to-agent-a2a-protocol)
7. [SDKs and Key Constructs](#sdks-and-key-constructs)
8. [Deployment and Configuration](#deployment-and-configuration)

---

## Overview

This project implements a **distributed multi-agent system** that enables intelligent routing and delegation of tasks between specialized AI agents. The system demonstrates how to build a scalable agent architecture using:

- **Azure AI Foundry Agent Service v2** for hosting intelligent agents
- **Agent-to-Agent (A2A) Protocol** for inter-agent communication
- **FastAPI/Starlette** for HTTP endpoints and server management
- **Async/await patterns** for concurrent request handling
- **Configurable timeouts** for reliable agent-to-agent communication

### Key Capabilities

The system consists of four specialized agents:

1. **Title Agent** - Generates catchy blog post titles from topics
2. **Outline Agent** - Creates structured outlines for articles (4-6 sections)
3. **Content Agent** - Generates brief, engaging blog content (max 100 words) **with Bing Search grounding** for current, accurate information
4. **Routing Agent** - Orchestrates requests and delegates to specialized agents in parallel

### Use Case

A user sends a natural language request (e.g., "Create a blog post about Python async programming"). The Routing Agent simultaneously delegates to all three specialist agents (Title, Outline, Content) via the A2A protocol, waits for all responses, and returns a combined result with title, outline, and content grounded in current web information.

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Client                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST /message
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                     Routing Agent Server                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application (routing_agent/server.py)           │  │
│  │  - Receives user messages                                │  │
│  │  - Lifecycle management                                  │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  RoutingAgent (routing_agent/agent.py)                   │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │ Azure AI Foundry Agent Service v2                 │   │  │
│  │  │ - AIProjectClient                                 │   │  │
│  │  │ - PromptAgentDefinition with FunctionTool         │   │  │
│  │  │ - Conversation Management (multi-turn)            │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  │                                                          │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │ A2A Client Components                             │   │  │
│  │  │ - RemoteAgentConnections                          │   │  │
│  │  │ - A2AClient instances                             │   │  │
│  │  │ - AgentCard resolvers                             │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  └────────────────────────┬─────────────────────────────────┘  │
└───────────────────────────┼────────────────────────────────────┘
                            │ A2A Protocol
              ┌─────────────┴──────────────┬──────────────────────┐
              │                            │                      │
              ▼                            ▼                      ▼
┌─────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│  Title Agent Server     │   │  Outline Agent Server    │   │  Content Agent Server    │
│  (title_agent/)         │   │  (outline_agent/)        │   │  (content_agent/)        │
│                         │   │                          │   │                          │
│  ┌──────────────────┐   │   │  ┌────────────────────┐  │   │  ┌────────────────────┐  │
│  │ A2A Server       │   │   │  │ A2A Server         │  │   │  │ A2A Server         │  │
│  │ (server.py)      │   │   │  │ (server.py)        │  │   │  │ (server.py)        │  │
│  │ - AgentCard      │   │   │  │ - AgentCard        │  │   │  │ - AgentCard        │  │
│  │ - A2A Routes     │   │   │  │ - A2A Routes       │  │   │  │ - A2A Routes       │  │
│  └────────┬─────────┘   │   │  └──────────┬─────────┘  │   │  └──────────┬─────────┘  │
│           │             │   │             │            │   │             │            │
│  ┌────────▼─────────┐   │   │  ┌──────────▼─────────┐  │   │  ┌──────────▼─────────┐  │
│  │ AgentExecutor    │   │   │  │ AgentExecutor      │  │   │  │ AgentExecutor      │  │
│  │ (agent_executor) │   │   │  │ (agent_executor)   │  │   │  │ (agent_executor)   │  │
│  │ - Task handling  │   │   │  │ - Task handling    │  │   │  │ - Task handling    │  │
│  │ - Status updates │   │   │  │ - Status updates   │  │   │  │ - Status updates   │  │
│  └────────┬─────────┘   │   │  └──────────┬─────────┘  │   │  └──────────┬─────────┘  │
│           │             │   │             │            │   │             │            │
│  ┌────────▼─────────┐   │   │  ┌──────────▼─────────┐  │   │  ┌──────────▼─────────┐  │
│  │ TitleAgent       │   │   │  │ OutlineAgent       │  │   │  │ ContentAgent       │  │
│  │ (agent.py)       │   │   │  │ (agent.py)         │  │   │  │ (agent.py)         │  │
│  │ - AIProjectClient│   │   │  │ - AIProjectClient  │  │   │  │ - AIProjectClient  │  │
│  │ - Agent v2       │   │   │  │ - Agent v2         │  │   │  │ - Agent v2         │  │
│  └──────────────────┘   │   │  └────────────────────┘  │   │  │ - Bing Grounding   │  │
└─────────────────────────┘   └──────────────────────────┘   │  └────────────────────┘  │
            │                            │                    │             │            │
            └────────────┬───────────────┴────────────────────┘             │            │
                         │                                                  │            │
                         ▼                                                  ▼            │
        ┌─────────────────────────────────────┐              ┌──────────────────────────┐│
        │   Azure AI Foundry Service v2       │              │   Bing Search API        ││
        │   - Agent Hosting                   │              │   - Web Search           ││
        │   - Model Deployment                │              │   - Grounding Data       ││
        │   - Conversation Management         │              └──────────────────────────┘│
        └─────────────────────────────────────┘                                         │
                                                               ←───────────────────────────┘
```

### Architecture Layers

#### 1. **Client Layer** (`client.py`, `run_all.py`)
- **Purpose**: User interface and orchestration
- **Components**:
  - `client.py`: Simple CLI for sending messages to the Routing Agent
  - `run_all.py`: Orchestrator that starts all agent servers and the client

#### 2. **Routing Layer** (`routing_agent/`)
- **Purpose**: Intelligent request routing and delegation
- **Components**:
  - `server.py`: FastAPI application exposing HTTP endpoints
  - `agent.py`: Core routing logic with Azure AI Foundry integration

#### 3. **Specialist Agent Layer** (`title_agent/`, `outline_agent/`, `content_agent/`)
- **Purpose**: Domain-specific task execution
- **Components**:
  - `server.py`: A2A-compliant server exposing agent capabilities
  - `agent_executor.py`: Task lifecycle management
  - `agent.py`: Azure AI Foundry agent implementation (Content Agent includes Bing grounding)

#### 4. **Azure AI Foundry Layer** (Cloud Service)
- **Purpose**: AI model hosting and agent runtime
- **Services**:
  - Agent Service v2 API
  - OpenAI-compatible endpoints
  - Conversation state management

---

## Component Roles and Responsibilities

### 1. Client Components

#### `client.py` - User Interface
**Role**: Provides a simple CLI for users to interact with the agent system.

**Responsibilities**:
- Accept user input from the command line
- Send HTTP POST requests to the Routing Agent
- Display agent responses
- Handle connection errors

**Key Functions**:
```python
def send_prompt(prompt: str) -> dict
    # Sends message to routing agent HTTP endpoint
    # Returns response or error message
```

#### `run_all.py` - System Orchestrator
**Role**: Manages the lifecycle of all agent servers and coordinates system startup.

**Responsibilities**:
- Start multiple uvicorn servers (one per agent)
- Perform health checks on each server
- Stream server logs to console
- Gracefully shutdown all servers on exit
- Launch the client after all servers are ready

**Key Functions**:
```python
async def wait_for_server_ready(server, timeout=30)
    # Polls server health endpoint until ready
    # Returns True if server responds within timeout

def stream_subprocess_output(process)
    # Streams subprocess stdout to main console
    # Runs in separate thread per server

async def main()
    # Orchestrates full system lifecycle
    # 1. Start all servers as subprocesses
    # 2. Wait for each to be healthy
    # 3. Run client
    # 4. Cleanup on exit
```

**Environment Variables Required**:
- `SERVER_URL`: Host for all servers (e.g., "127.0.0.1")
- `TITLE_AGENT_PORT`: Port for Title Agent
- `OUTLINE_AGENT_PORT`: Port for Outline Agent
- `CONTENT_AGENT_PORT`: Port for Content Agent
- `ROUTING_AGENT_PORT`: Port for Routing Agent

---

### 2. Routing Agent Components

#### `routing_agent/server.py` - HTTP Server
**Role**: Exposes HTTP endpoints for client communication.

**Responsibilities**:
- Create FastAPI application with async lifecycle management
- Initialize RoutingAgent during startup
- Handle incoming `/message` POST requests
- Provide `/health` endpoint for monitoring
- Cleanup resources on shutdown

**Key Patterns**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize routing agent and connect to remote agents
    global routing_agent
    routing_agent = await RoutingAgent.create([...remote_urls...])
    await routing_agent.create_agent()
    yield
    # Shutdown: Close all connections
    await routing_agent.close()

@app.post("/message")
async def handle_message(request: Request):
    # Extract user message from request body
    # Delegate to routing agent for processing
    # Return agent response
```

#### `routing_agent/agent.py` - Core Routing Logic
**Role**: Intelligent request routing using Azure AI Foundry Agent Service v2 with function calling.

**Responsibilities**:
1. **Remote Agent Management**:
   - Discover and connect to remote A2A agents
   - Retrieve and cache AgentCard metadata
   - Maintain A2A client connections with configurable timeouts

2. **Azure AI Foundry Integration**:
   - Create and manage Azure AI agent with function tools
   - Maintain conversation state for multi-turn interactions
   - Handle function calling loop for agent delegation

3. **Request Processing**:
   - Accept user messages
   - Let Azure AI agent decide which remote agent to invoke
   - Execute function calls to delegate tasks with timeout protection
   - Return final responses to user

**Key Classes and Methods**:

```python
class RemoteAgentConnections:
    """Wrapper for A2A client connection to a remote agent"""
    
    def __init__(self, agent_card: AgentCard, agent_url: str, timeout: float = 120.0):
        # Initialize httpx client with configurable timeout
        # Default: 120 seconds for agent-to-agent communication
        # Store agent card metadata
    
    async def send_message(self, message_request: SendMessageRequest):
        # Send A2A message to remote agent
        # Returns SendMessageResponse

class RoutingAgent:
    """Main routing agent with Azure AI Foundry integration"""
    
    def __init__(self, task_callback: Optional[TaskUpdateCallback], agent_timeout: float = 120.0):
        # Initialize Azure clients:
        # - DefaultAzureCredential for authentication
        # - AIProjectClient for agent management
        # - OpenAI client for conversation API
        # - agent_timeout: Timeout in seconds for A2A communication (default: 120s)
        
    @classmethod
    async def create(cls, remote_agent_addresses: list[str], task_callback: Optional[TaskUpdateCallback] = None, agent_timeout: float = 120.0):
        # Factory method for async initialization
        # 1. Create instance with specified timeout
        # 2. Connect to all remote agents with timeout configuration
        # 3. Return initialized instance
    
    async def _async_init_components(self, remote_agent_addresses: list[str]):
        # Connect to each remote agent via A2A protocol
        # Retrieve AgentCard from each agent
        # Handle connection errors (ConnectError, TimeoutException)
        # Store connections in dictionary
    
    def list_remote_agents(self) -> str:
        # Return formatted list of available remote agents
        # Used in Azure AI agent instructions
    
    async def send_message_to_agent(self, agent_name: str, task: str) -> str:
        # Function exposed as tool to Azure AI agent
        # 1. Validate agent exists
        # 2. Create A2A message request
        # 3. Send to remote agent with timeout protection
        # 4. Return JSON-serialized response
        # Handles multiple timeout scenarios:
        #   - TimeoutException: Overall operation timeout
        #   - ReadTimeout: Response reading timeout
        #   - ConnectTimeout: Connection establishment timeout
        # Error messages include timeout duration for debugging
    
    async def create_agent(self):
        # Create Azure AI agent with function tools
        # 1. Define FunctionTool for send_message_to_agent
        # 2. Create agent with PromptAgentDefinition
        # 3. Create conversation for multi-turn chat
        # 4. Store agent and conversation IDs
    
    async def process_user_message(self, user_message: str) -> str:
        # Main processing loop using parallel execution
        # 1. Call all three specialist agents in parallel using asyncio.gather()
        #    - Title Agent: generates title
        #    - Outline Agent: generates outline
        #    - Content Agent: generates content with Bing grounding
        # 2. Wait for all agents with timeout protection (default: 120s each)
        # 3. Parse and extract responses from A2A task data
        # 4. Return combined JSON: {"title": "...", "outline": "...", "content": "..."}
        # 5. Fail completely if any agent fails or times out (all-or-nothing)
    
    async def close(self):
        # Cleanup all connections
        # - Close A2A client connections
        # - Close OpenAI client
        # - Close project client
```

**Timeout Configuration**:

The routing agent supports configurable timeouts for agent-to-agent communication:

```python
# Default timeout (120 seconds)
routing_agent = await RoutingAgent.create(
    remote_agent_addresses=[...]
)

# Custom timeout (e.g., 60 seconds for faster failure)
routing_agent = await RoutingAgent.create(
    remote_agent_addresses=[...],
    agent_timeout=60.0
)

# Extended timeout (e.g., 300 seconds for long-running tasks)
routing_agent = await RoutingAgent.create(
    remote_agent_addresses=[...],
    agent_timeout=300.0
)
```

**Timeout Error Handling**:

The system provides detailed error messages for different timeout scenarios:

- **Connection Timeout**: `"Connection timeout to {agent_name}"`
- **Read Timeout**: `"Read timeout from {agent_name} (timeout: {timeout}s)"`
- **General Timeout**: `"Timeout waiting for response from {agent_name} (timeout: {timeout}s)"`

These errors are returned as JSON responses and propagate through the system for proper error handling.

---

### 3. Specialist Agent Components (Title, Outline & Content)

All three specialist agents (Title, Outline, and Content) follow the same architectural pattern, differing only in their specific AI instructions and capabilities. The Content Agent additionally integrates **Bing Search grounding** for generating content based on current web information.

#### `{agent}/server.py` - A2A Server
**Role**: Expose agent capabilities via A2A protocol.

**Responsibilities**:
- Define AgentCard with skills and capabilities
- Create A2AStarletteApplication
- Configure request handlers and task storage
- Expose A2A protocol endpoints
- Provide health check endpoint

**Key Constructs**:
```python
# Define agent skills (what the agent can do)
skills = [
    AgentSkill(
        id='generate_blog_title',  # or 'generate_outline'
        name='Generate Blog Title',
        description='Generates a blog title based on a topic',
        tags=['title'],
        examples=['Can you give me a title for this article?'],
    ),
]

# Create agent card (agent's public profile)
agent_card = AgentCard(
    name='AI Foundry Title Agent',
    description='An intelligent title generator...',
    url=f'http://{host}:{port}/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(),
    skills=skills,
)

# Create agent executor (handles requests)
agent_executor = create_foundry_agent_executor(agent_card)

# Create request handler (A2A protocol compliance)
request_handler = DefaultRequestHandler(
    agent_executor=agent_executor,
    task_store=InMemoryTaskStore()
)

# Create A2A application (exposes protocol endpoints)
a2a_app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler
)
```

#### `{agent}/agent_executor.py` - Task Lifecycle Manager
**Role**: Bridge between A2A protocol and Azure AI Foundry agent.

**Responsibilities**:
- Implement `AgentExecutor` interface from A2A SDK
- Manage task lifecycle (submit, working, complete, failed)
- Extract user messages from A2A protocol structures
- Invoke Azure AI Foundry agent
- Update task status with progress and results
- Handle errors and cancellations

**Key Class Structure**:
```python
class FoundryAgentExecutor(AgentExecutor):
    """Executes tasks using Azure AI Foundry agent"""
    
    def __init__(self, card: AgentCard):
        # Store agent card
        # Initialize foundry agent reference (lazy load)
    
    async def _get_or_create_agent(self):
        # Lazy initialization of Azure AI agent
        # Returns TitleAgent or OutlineAgent instance
    
    async def _process_request(self, message_parts, context_id, task_updater):
        # Main processing logic:
        # 1. Extract text from A2A message parts
        # 2. Get or create Foundry agent
        # 3. Update task status to "working"
        # 4. Run agent conversation
        # 5. Stream responses as task updates
        # 6. Mark task as complete
        # 7. Handle errors with failed status
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Called by A2A framework to execute a task
        # 1. Create TaskUpdater
        # 2. Submit task
        # 3. Start work
        # 4. Process request
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        # Handle cancellation requests
        # Mark task as failed with cancellation message
```

#### `{agent}/agent.py` - Azure AI Foundry Agent Wrapper
**Role**: Encapsulate Azure AI Foundry Agent Service v2 interaction.

**Responsibilities**:
- Initialize Azure AI clients (AIProjectClient, OpenAI client)
- Create Azure AI agent with specific instructions
- Run conversations with the agent
- Handle authentication and credentials
- Manage resource cleanup

**Key Class Structure**:
```python
class TitleAgent:  # or OutlineAgent
    """Azure AI Foundry Agent Service v2 wrapper"""
    
    def __init__(self):
        # Initialize Azure credentials (DefaultAzureCredential)
        # Create AIProjectClient
        # Get OpenAI client from project client
        # Initialize agent and conversation state
    
    async def create_agent(self):
        # Create agent using v2 API
        # Define agent with:
        #   - agent_name: Unique identifier
        #   - model: Deployment name from environment
        #   - instructions: Agent's system prompt
        # Wrap sync API call in asyncio.to_thread()
        # Return created agent
    
    async def run_conversation(self, user_message: str) -> list[str]:
        # Run a stateless conversation:
        # 1. Create new conversation
        # 2. Send user message
        # 3. Get response from agent
        # 4. Return output text
        # All sync calls wrapped in asyncio.to_thread()
    
    async def close(self):
        # Close OpenAI client
        # Close project client
        # Wrapped in asyncio.to_thread()
```

#### Content Agent - Bing Search Grounding

The **Content Agent** extends the standard agent pattern with **Bing Search grounding** capability, enabling it to generate content based on current web information.

**Key Differences from Title/Outline Agents**:

1. **Bing Grounding Tool Integration** (`content_agent/agent.py`):
   - Uses `BingGroundingAgentTool` from Azure AI Projects SDK
   - Configured with `BingGroundingSearchToolParameters` and `BingGroundingSearchConfiguration`
   - Requires `BING_PROJECT_CONNECTION_ID` environment variable (Azure AI Foundry project connection)

2. **Enhanced AI Instructions**:
   - Explicitly instructs the agent: "Use the Bing search tool to gather current, accurate information to ground your content"
   - Generates content based on real-time web search results
   - Produces well-researched, factually grounded blog content (max 100 words)

3. **AgentCard Capabilities**:
   - Described as providing "well-researched blog content grounded in current information"
   - Status messages indicate "Content Agent is processing your request with Bing search..."

**Setup Requirements**:
- Azure AI Foundry project with Bing Search connection configured
- `BING_PROJECT_CONNECTION_ID` must be set to your project's Bing connection ID
- Find connection ID in Azure AI Foundry portal under project connections

**Architecture Pattern**:
```
ContentAgent
    ├─> Azure AI Foundry Agent Service v2 (standard agent creation)
    ├─> BingGroundingAgentTool (web search capability)
    └─> Bing Search API (via Azure AI Foundry connection)
```

The Content Agent follows the same `server.py` → `agent_executor.py` → `agent.py` pattern as other agents, with the addition of Bing grounding configuration in the agent creation process.

---

## Request Flow

### End-to-End Request Flow

Let's trace a complete request: **"Create a blog post about Python async programming"**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 1: User Input                                                  │
└──────────────────────────────────────────────────────────────────────┘

User (client.py)
    │
    └─► send_prompt("Create a blog post about Python async programming")
            │
            └─► HTTP POST http://127.0.0.1:8003/message
                Body: {"message": "Create a blog post about Python async programming"}

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 2: Routing Agent Receives Request                             │
└──────────────────────────────────────────────────────────────────────┘

routing_agent/server.py
    │
    ├─► @app.post("/message")
    │   async def handle_message(request: Request)
    │       │
    │       ├─► Extract: user_message = "Create a blog post..."
    │       │
    │       └─► routing_agent.process_user_message(user_message)
    │
    └─► routing_agent/agent.py: RoutingAgent.process_user_message()

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 3: Parallel Agent Execution                                   │
└──────────────────────────────────────────────────────────────────────┘

RoutingAgent.process_user_message()
    │
    ├─► Simultaneously calls all three specialist agents using asyncio.gather():
    │   
    │   title_task = self.send_message_to_agent("AI Foundry Title Agent", user_message)
    │   outline_task = self.send_message_to_agent("AI Foundry Outline Agent", user_message)
    │   content_task = self.send_message_to_agent("AI Foundry Content Agent", user_message)
    │   
    │   title_result, outline_result, content_result = await asyncio.gather(
    │       title_task, outline_task, content_task
    │   )
    │
    └─► All three agents process concurrently (not sequentially)
        - Each has 120s timeout protection
        - Fails completely if ANY agent fails (all-or-nothing)

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 4: Title Agent Processing (Concurrent with Outline & Content) │
└──────────────────────────────────────────────────────────────────────┘

RoutingAgent.send_message_to_agent("AI Foundry Title Agent", ...)
    │
    ├─► 1. Validate agent exists: "AI Foundry Title Agent" ✓
    │
    ├─► 2. Get A2A client connection
    │      client = remote_agent_connections["AI Foundry Title Agent"]
    │
    ├─► 3. Create A2A message request
    │      message_id = uuid.uuid4()
    │      payload = {
    │          "message": {
    │              "role": "user",
    │              "parts": [{"kind": "text", "text": "Create a blog post..."}],
    │              "messageId": message_id
    │          }
    │      }
    │
    └─► 4. Send via A2A protocol with timeout protection
           try:
               send_response = await client.send_message(message_request)
           except httpx.TimeoutException:
               return {"error": "Timeout waiting for response (timeout: 120s)"}
           except httpx.ReadTimeout:
               return {"error": "Read timeout from agent (timeout: 120s)"}
           except httpx.ConnectTimeout:
               return {"error": "Connection timeout to agent"}
           
           └─► HTTP POST http://127.0.0.1:8001/api/a2a/messages
               Headers: Content-Type: application/json
               Body: A2A protocol message
               Timeout: 120 seconds (configurable)

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 5: Title Agent Processing (Concurrent)                        │
└──────────────────────────────────────────────────────────────────────┘

title_agent/server.py → title_agent/agent_executor.py → title_agent/agent.py

TitleAgent.run_conversation()
    │
    ├─► 1. Create new conversation with Azure AI
    │      conversation = openai_client.conversations.create()
    │
    ├─► 2. Send message to Azure AI agent
    │      Model: gpt-4o
    │      Instructions: "Generate catchy blog post title..."
    │      Input: "Create a blog post about Python async programming"
    │      
    │      → Model generates: "Mastering Python Async: A Developer's Guide"
    │
    └─► 3. Return via A2A protocol:
           {"result": {"state": "completed", "artifacts": [{"parts": [{"text": "..."}]}]}}

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 6: Outline Agent Processing (Concurrent)                      │
└──────────────────────────────────────────────────────────────────────┘

outline_agent/server.py → outline_agent/agent_executor.py → outline_agent/agent.py

OutlineAgent.run_conversation()
    │
    ├─► Similar process as Title Agent
    │   Model: gpt-4o
    │   Instructions: "Create concise outline with 4-6 sections..."
    │   Input: "Create a blog post about Python async programming"
    │   
    │   → Model generates:
    │     "1. Introduction to Async
    │      2. Event Loop Basics
    │      3. Async/Await Syntax
    │      4. Common Patterns
    │      5. Error Handling
    │      6. Best Practices"
    │
    └─► Return via A2A protocol

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 7: Content Agent Processing (Concurrent) - WITH BING SEARCH   │
└──────────────────────────────────────────────────────────────────────┘

content_agent/server.py → content_agent/agent_executor.py → content_agent/agent.py

ContentAgent.run_conversation()
    │
    ├─► 1. Create Azure AI agent with BingGroundingAgentTool
    │      Model: gpt-4o
    │      Tool: Bing Search (via project connection)
    │      Instructions: "Generate brief blog content... Use Bing search tool..."
    │      
    ├─► 2. Agent automatically uses Bing Search
    │      → Searches web for "Python async programming"
    │      → Retrieves current information, articles, best practices
    │      → Grounds content generation in search results
    │
    ├─► 3. Model generates content based on Bing results:
    │      "Python's async programming enables efficient handling of I/O-bound 
    │       operations. Using async/await syntax, developers can write concurrent 
    │       code that doesn't block execution. Modern frameworks like FastAPI and 
    │       aiohttp leverage async for high-performance applications..."
    │      (Max 100 words, grounded in current web information)
    │
    └─► 4. Return via A2A protocol

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 8: Combine Results and Return to User                         │
└──────────────────────────────────────────────────────────────────────┘

RoutingAgent.process_user_message()
    │
    ├─► All three agents have completed (via asyncio.gather)
    │   title_result = {... Title Agent response ...}
    │   outline_result = {... Outline Agent response ...}
    │   content_result = {... Content Agent response ...}
    │
    ├─► Parse responses from A2A task data
    │   title = extract_text(title_result)
    │   outline = extract_text(outline_result)
    │   content = extract_text(content_result)
    │
    ├─► Combine into structured JSON
    │   response = {
    │       "title": "Mastering Python Async: A Developer's Guide",
    │       "outline": "1. Introduction...\n2. Event Loop...",
    │       "content": "Python's async programming enables..."
    │   }
    │
    └─► Return combined result

routing_agent/server.py
    │
    └─► return {"response": json.dumps(response)}

client.py
    │
    └─► Display to user:
        {
            "title": "Mastering Python Async: A Developer's Guide",
            "outline": "1. Introduction to Async\n2. Event Loop Basics...",
            "content": "Python's async programming enables efficient..."
        }
```

### Key Flow Observations

1. **Parallel Batch Processing Architecture**:
   - Routing Agent calls **all three specialist agents simultaneously**
   - Uses `asyncio.gather()` for concurrent execution
   - All-or-nothing: fails completely if any agent fails
   - No intelligent routing decision - always processes all three agents

2. **Content Agent with Bing Grounding**:
   - Automatically searches web for current information
   - Generates content based on real-time search results
   - Provides factually grounded, up-to-date blog content
   - Requires Bing Search connection in Azure AI Foundry project

3. **Structured Output Format**:
   - System always returns JSON with three fields: `title`, `outline`, `content`
   - Consistent format regardless of user input
   - Each field contains specialist agent's response

4. **Protocol Separation**:
   - User ↔ Routing Agent: Simple HTTP JSON
   - Routing Agent ↔ Specialist Agents: A2A protocol (standardized agent communication)
   - All Agents ↔ Azure AI: OpenAI-compatible API
   - Content Agent ↔ Bing Search: Azure AI Foundry connection

5. **Asynchronous Processing**:
   - All I/O operations use async/await
   - Three agents queried concurrently (not sequentially)
   - Non-blocking server implementations
   - **Timeout protection** prevents indefinite waiting (default: 120s per agent)

6. **State Management**:
   - Routing Agent: Maintains conversation state across user turns (not used in current implementation)
   - Specialist Agents: Stateless per A2A request (new conversation each time)
   - Azure AI: Manages conversation history in cloud
   - **Timeout configuration** persists across agent lifecycle
           "I've generated a title for your article: 
            'Mastering Python Async: A Developer's Guide'"

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 8: Return to User                                             │
└──────────────────────────────────────────────────────────────────────┘

routing_agent/server.py
    │
    └─► return {"response": final_response}

client.py
    │
    └─► Display to user:

---

## Azure AI Foundry Agent Service v2 Integration

### What is Azure AI Foundry Agent Service v2?

Azure AI Foundry Agent Service v2 is Microsoft's managed platform for building, deploying, and hosting AI agents. It provides:

- **Managed Agent Hosting**: Serverless execution of AI agents
- **Model Access**: Direct integration with Azure OpenAI models
- **Conversation Management**: Built-in multi-turn conversation support
- **Function Calling**: Native tool/function invocation capabilities
- **Enterprise Features**: Authentication, monitoring, governance

### Key Differences from v1

| Feature | v1 (Legacy) | v2 (Current) |
|---------|-------------|--------------|
| Client SDK | `AgentsClient` | `AIProjectClient` |
| Agent Creation | `create_agent()` | `create_version()` with versioning |
| Agent Definition | Inline parameters | `PromptAgentDefinition` model |
| Conversations | Manual management | Built-in conversation API |
| Tool Definition | `ToolSet` with decorators | `FunctionTool` declarative model |
| Response API | Thread/message pattern | Direct response API with `previous_response_id` |

### SDK Architecture

#### Core Components

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool

# 1. Authentication
credential = DefaultAzureCredential(
    exclude_environment_credential=True,
    exclude_managed_identity_credential=True
)
# Uses Azure CLI credentials or managed identity

# 2. Project Client (main entry point)
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],  # e.g., https://xxx.services.ai.azure.com/api/projects/xxx
    credential=credential
)

# 3. OpenAI Client (for conversations and responses)
openai_client = project_client.get_openai_client()
# Returns OpenAI-compatible client with Azure authentication
```

### Agent Creation Pattern

#### Step 1: Define Function Tools

Function tools allow agents to call Python functions during conversation processing.

```python
send_message_tool = FunctionTool(
    name="send_message_to_agent",
    description="Send a task to a remote agent via A2A protocol. Use this to delegate tasks to specialized agents.",
    parameters={
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "The name of the remote agent to send the message to"
            },
            "task": {
                "type": "string",
                "description": "The task or message to send to the agent"
            }
        },
        "required": ["agent_name", "task"]
    }
)
```

**Key Aspects**:
- `name`: Function identifier (must match actual Python function)
- `description`: Helps Azure AI agent decide when to use the tool
- `parameters`: JSON Schema describing function arguments
- Follows OpenAI function calling schema

#### Step 2: Create Agent Definition

```python
definition = PromptAgentDefinition(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],  # e.g., "gpt-4o"
    instructions="""
        You are an expert Routing Delegator that helps users with requests.
        
        Your role:
        - Delegate user inquiries to appropriate specialized remote agents
        - Provide clear and helpful responses to users
        
        Available Agents: {list_of_agents}
        
        Always be helpful and route requests to the most appropriate agent.
        Use the send_message_to_agent function to communicate with remote agents.
    """,
    tools=[send_message_tool],  # List of FunctionTool objects
)
```

**Key Aspects**:
- `model`: Specifies which Azure OpenAI deployment to use
- `instructions`: System prompt defining agent behavior
- `tools`: List of available function tools

#### Step 3: Create Versioned Agent

```python
agent = project_client.agents.create_version(
    agent_name="routing-agent-v2",  # Unique identifier
    definition=definition
)
```

**Important**: 
- v2 uses versioning system - each `create_version()` creates a new immutable version
- `agent_name` identifies the agent family
- Returns agent object with `name` field like `"routing-agent-v2:v1"`

### Conversation Management

#### Creating Conversations

```python
conversation = openai_client.conversations.create()
conversation_id = conversation.id
```

**Purpose**: Conversations track multi-turn interactions and maintain context.

#### Sending Messages and Getting Responses

##### First Message in Conversation

```python
response = openai_client.responses.create(
    conversation=conversation_id,  # Reference to conversation
    input=user_message,  # User's text input
    extra_body={
        "agent": {
            "name": agent.name,  # e.g., "routing-agent-v2:v1"
            "type": "agent_reference"
        }
    }
)
```

##### Subsequent Messages (Multi-Turn)

```python
response = openai_client.responses.create(
    input=next_user_message,
    previous_response_id=response.id,  # Links to previous response
    extra_body={
        "agent": {
            "name": agent.name,
            "type": "agent_reference"
        }
    }
)
```

**Key Pattern**: Use `previous_response_id` to maintain conversation continuity.

#### Response Structure

```python
response.id  # Unique response identifier
response.output  # List of output items (text, function_calls, etc.)
response.output_text  # Convenience property for text-only output
```

**Output Types**:
- `text`: Text content from the agent
- `function_call`: Request to execute a function tool

### Function Calling Loop

The function calling loop is a critical pattern in v2:

```python
async def process_user_message(self, user_message: str) -> str:
    # Send initial message
    response = await asyncio.to_thread(
        lambda: openai_client.responses.create(
            conversation=conversation_id,
            input=user_message,
            extra_body={"agent": {...}}
        )
    )
    
    # Process function calls iteratively
    while True:
        # Check for function calls in response
        function_calls = [
            item for item in response.output 
            if item.type == "function_call"
        ]
        
        if not function_calls:
            # No more function calls, return final text
            return response.output_text
        
        # Execute each function call
        tool_outputs = []
        for call in function_calls:
            if call.name == "send_message_to_agent":
                args = json.loads(call.arguments)
                result = await self.send_message_to_agent(
                    agent_name=args["agent_name"],
                    task=args["task"]
                )
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result  # Must be string (JSON)
                })
        
        # Send tool outputs back to agent
        response = await asyncio.to_thread(
            lambda: openai_client.responses.create(
                input=tool_outputs,
                previous_response_id=response.id,
                extra_body={"agent": {...}}
            )
        )
        # Loop continues until no more function calls
```

**Loop Phases**:
1. Send user message or tool outputs
2. Receive response with potential function calls
3. Execute function calls and collect outputs
4. Send outputs back and repeat
5. Exit when response contains only text

### Async Compatibility

The v2 SDK is **synchronous**, but this project requires async operations. The pattern used:

```python
import asyncio

# Wrap sync calls in asyncio.to_thread()
async def create_agent(self):
    def _create():
        return self.project_client.agents.create_version(...)
    
    self.agent = await asyncio.to_thread(_create)

async def run_conversation(self, user_message: str):
    def _run_sync():
        conversation = self.openai_client.conversations.create()
        response = self.openai_client.responses.create(...)
        return response.output_text
    
    result = await asyncio.to_thread(_run_sync)
    return [result]
```

**Why This Matters**:
- FastAPI/Starlette require async handlers
- A2A protocol operations are async
- Prevents blocking the event loop
- Maintains concurrent request handling

### Resource Management

```python
async def close(self):
    """Close all client connections"""
    def _close():
        self.openai_client.close()
        self.project_client.close()
    
    await asyncio.to_thread(_close)
```

**Best Practice**: Always close clients in cleanup handlers (FastAPI `lifespan`, `finally` blocks).

---

## Agent-to-Agent (A2A) Protocol

### What is A2A?

The **Agent-to-Agent (A2A) Protocol** is a standardized communication protocol for inter-agent messaging. It provides:

- **Standardized Message Format**: Consistent structure across agents
- **Task Lifecycle Management**: Submit, working, completed, failed states
- **Capability Discovery**: AgentCard metadata for agent discovery
- **Streaming Support**: Real-time task progress updates
- **Error Handling**: Structured error responses

### A2A Protocol Components

#### 1. AgentCard - Agent Metadata

The `AgentCard` is an agent's public profile, describing its capabilities and location.

```python
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

agent_card = AgentCard(
    name='AI Foundry Title Agent',
    description='An intelligent title generator agent powered by Foundry.',
    url='http://127.0.0.1:8001/',  # Agent's base URL
    version='1.0.0',
    default_input_modes=['text'],  # Supported input types
    default_output_modes=['text'],  # Supported output types
    capabilities=AgentCapabilities(
        streaming=False  # Whether agent supports streaming responses
    ),
    skills=[  # List of agent skills
        AgentSkill(
            id='generate_blog_title',
            name='Generate Blog Title',
            description='Generates a blog title based on a topic',
            tags=['title'],
            examples=['Can you give me a title for this article?'],
        )
    ],
)
```

**Purpose**: 
- Service discovery: Routing agents can list available agents
- Capability negotiation: Clients know what the agent can do
- Documentation: Self-describing agent endpoints

**Retrieval**: Exposed at `{agent_url}/api/a2a/agent-card`

#### 2. Message Structure

A2A messages follow a structured format:

```python
from a2a.types import SendMessageRequest, MessageSendParams

message_request = SendMessageRequest(
    id=message_id,  # Unique message identifier
    params=MessageSendParams(
        message={
            'role': 'user',  # Message sender role
            'parts': [  # Message content parts
                {
                    'kind': 'text',
                    'text': 'Generate a title for Python async'
                }
            ],
            'messageId': message_id
        }
    )
)
```

**Key Fields**:
- `id`: UUID for tracking the request
- `role`: `'user'` or `'agent'`
- `parts`: List of content parts (text, images, etc.)
- `messageId`: Links messages in a conversation

#### 3. Task Lifecycle

A2A uses a task-based model with state transitions:

```
┌─────────┐
│submitted│  Initial state when task is received
└────┬────┘
     │
     ▼
┌─────────┐
│ working │  Agent is processing the request
└────┬────┘
     │
     ├──────► ┌──────────┐
     │        │completed │  Task finished successfully
     │        └──────────┘
     │
     └──────► ┌────────┐
              │ failed │  Task encountered an error
              └────────┘
```

**State Updates via TaskUpdater**:

```python
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

# Initialize updater
updater = TaskUpdater(event_queue, task_id, context_id)

# Submit task
await updater.submit()

# Start work
await updater.start_work()

# Update progress
await updater.update_status(
    TaskState.working,
    message=new_agent_text_message('Processing...', context_id)
)

# Complete task
await updater.complete(
    message=new_agent_text_message('Result text', context_id)
)

# Or mark as failed
await updater.failed(
    message=new_agent_text_message('Error occurred', context_id)
)
```

#### 4. Response Structure

Successful A2A responses contain task information:

```python
from a2a.types import SendMessageResponse, Task

send_response: SendMessageResponse = await client.send_message(message_request)

# Check response type
if isinstance(send_response.root, SendMessageSuccessResponse):
    # Extract task
    task: Task = send_response.root.result
    
    # Access task fields
    task.id  # Task identifier
    task.state  # TaskState (submitted, working, completed, failed)
    task.artifacts  # List of result artifacts
    task.message  # Current status message
```

### A2A Client Components

#### A2ACardResolver - Agent Discovery

```python
from a2a.client import A2ACardResolver
import httpx

# Configure timeout for card resolution
timeout_config = httpx.Timeout(timeout=120.0, connect=60.0)
async with httpx.AsyncClient(timeout=timeout_config) as client:
    card_resolver = A2ACardResolver(client, agent_url)
    agent_card = await card_resolver.get_agent_card()
    # Returns AgentCard from {agent_url}/api/a2a/agent-card
```

#### A2AClient - Message Sending

```python
from a2a.client import A2AClient
import httpx

# Configure timeout for message sending
timeout_config = httpx.Timeout(timeout=120.0, connect=60.0)
httpx_client = httpx.AsyncClient(timeout=timeout_config)
a2a_client = A2AClient(httpx_client, agent_card, url=agent_url)

# Send message (will timeout after configured duration)
try:
    response = await a2a_client.send_message(message_request)
except httpx.TimeoutException:
    # Handle timeout appropriately
    print(f"Request timed out after {timeout_config.timeout}s")
```

**Connection Pattern in This Project**:

```python
class RemoteAgentConnections:
    """Wrapper for A2A client connection"""
    
    def __init__(self, agent_card: AgentCard, agent_url: str, timeout: float = 120.0):
        # Configure timeout for all HTTP operations
        timeout_config = httpx.Timeout(timeout=timeout, connect=60.0)
        self._httpx_client = httpx.AsyncClient(timeout=timeout_config)
        self.agent_client = A2AClient(
            self._httpx_client,
            agent_card,
            url=agent_url
        )
        self.card = agent_card
    
    async def send_message(self, message_request):
        # Will timeout based on configured duration
        return await self.agent_client.send_message(message_request)
    
    async def close(self):
        await self._httpx_client.aclose()
```

### A2A Server Components

#### A2AStarletteApplication - Protocol Server

```python
from a2a.server.apps import A2AStarletteApplication
from starlette.applications import Starlette

# Create A2A application
a2a_app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler
)

# Get A2A routes
routes = a2a_app.routes()
# Returns routes for:
# - GET /api/a2a/agent-card
# - POST /api/a2a/messages
# - POST /api/a2a/cancel

# Add custom routes (e.g., health check)
routes.append(Route('/health', endpoint=health_check, methods=['GET']))

# Create Starlette app
app = Starlette(routes=routes)
```

#### DefaultRequestHandler - Request Processing

```python
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

request_handler = DefaultRequestHandler(
    agent_executor=agent_executor,  # Your custom AgentExecutor
    task_store=InMemoryTaskStore()  # In-memory task storage
)
```

**Purpose**:
- Receives incoming A2A messages
- Validates request structure
- Creates task in task store
- Invokes AgentExecutor to process request
- Manages event queue for task updates

#### AgentExecutor Interface

Custom agent executors must implement this interface:

```python
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue

class MyAgentExecutor(AgentExecutor):
    
    async def execute(
        self,
        context: RequestContext,  # Request context with message and IDs
        event_queue: EventQueue   # Queue for sending status updates
    ):
        # 1. Create TaskUpdater
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # 2. Submit task
        await updater.submit()
        
        # 3. Start work
        await updater.start_work()
        
        # 4. Process request (call your AI agent)
        # ...
        
        # 5. Update status/complete task
        await updater.complete(message=result_message)
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ):
        # Handle cancellation requests
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.failed(message=cancellation_message)
```

**RequestContext Fields**:
- `task_id`: Unique task identifier
- `context_id`: Conversation context identifier
- `message`: Incoming message with parts

### A2A Protocol Endpoints

Each A2A-compliant agent exposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/a2a/agent-card` | GET | Retrieve agent metadata |
| `/api/a2a/messages` | POST | Send message to agent |
| `/api/a2a/cancel` | POST | Cancel active task |

---

## SDKs and Key Constructs

### 1. Azure AI Projects SDK (`azure-ai-projects`)

**Purpose**: Interact with Azure AI Foundry Agent Service v2.

**Key Classes**:

#### AIProjectClient
Main client for agent and project operations.

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint="https://xxx.services.ai.azure.com/api/projects/xxx",
    credential=credential
)

# Agent operations
agent = project_client.agents.create_version(...)
agents_list = project_client.agents.list()

# Get OpenAI client
openai_client = project_client.get_openai_client()
```

#### PromptAgentDefinition
Defines agent configuration.

```python
from azure.ai.projects.models import PromptAgentDefinition

definition = PromptAgentDefinition(
    model="gpt-4o",
    instructions="You are a helpful assistant...",
    tools=[...],  # List of FunctionTool objects
    temperature=0.7,  # Optional parameters
    top_p=1.0
)
```

#### FunctionTool
Declares a function that the agent can call.

```python
from azure.ai.projects.models import FunctionTool

tool = FunctionTool(
    name="function_name",
    description="What this function does",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "number", "description": "..."}
        },
        "required": ["param1"]
    }
)
```

### 2. Azure Identity SDK (`azure-identity`)

**Purpose**: Authenticate to Azure services.

```python
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential(
    exclude_environment_credential=True,  # Don't use env vars
    exclude_managed_identity_credential=True  # Don't use managed identity
)
# Falls back to Azure CLI credentials
```

**Credential Chain** (in order):
1. Environment variables (excluded in this project)
2. Managed Identity (excluded in this project)
3. Azure CLI credentials ✓ (used in this project)
4. Azure PowerShell credentials
5. Interactive browser authentication

### 3. A2A SDK (`a2a`)

**Purpose**: Implement Agent-to-Agent protocol.

**Key Modules**:

#### Client Side (`a2a.client`)

```python
from a2a.client import A2AClient, A2ACardResolver

# Discover agent
resolver = A2ACardResolver(httpx_client, agent_url)
card = await resolver.get_agent_card()

# Create client
client = A2AClient(httpx_client, card, url=agent_url)

# Send message
response = await client.send_message(message_request)
```

#### Server Side (`a2a.server`)

```python
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor

# Create application
app = A2AStarletteApplication(
    agent_card=card,
    http_handler=DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore()
    )
)
```

#### Types (`a2a.types`)

```python
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    SendMessageRequest,
    SendMessageResponse,
    MessageSendParams,
    Task,
    TaskState,
    Part
)
```

#### Utilities (`a2a.utils`)

```python
from a2a.utils import new_agent_text_message

message = new_agent_text_message(
    text="Response text",
    context_id=context_id
)
```

### 4. FastAPI/Starlette

**Purpose**: HTTP server framework with async support.

#### FastAPI with Lifespan

```python
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("Starting up...")
    await initialize_resources()
    yield
    # Shutdown code
    print("Shutting down...")
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)

@app.post("/endpoint")
async def handler(request: Request):
    data = await request.json()
    return {"result": "..."}
```

#### Starlette for A2A Servers

```python
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import PlainTextResponse

async def health(request):
    return PlainTextResponse("Healthy")

routes = [
    Route("/health", endpoint=health, methods=["GET"]),
    *a2a_app.routes()  # Add A2A routes
]

app = Starlette(routes=routes)
```

### 5. httpx

**Purpose**: Async HTTP client for making requests.

```python
import httpx

# Async context manager with timeout
timeout_config = httpx.Timeout(timeout=120.0, connect=60.0)
async with httpx.AsyncClient(timeout=timeout_config) as client:
    response = await client.get("http://example.com")
    response = await client.post(
        "http://example.com/api",
        json={"key": "value"}
    )

# Long-lived client with custom timeout
timeout_config = httpx.Timeout(timeout=120.0, connect=60.0)
client = httpx.AsyncClient(timeout=timeout_config)
try:
    response = await client.get("...")
finally:
    await client.aclose()
```

**Timeout Configuration**:

httpx supports granular timeout control:

```python
# Simple timeout (applies to all operations)
httpx.Timeout(timeout=30.0)

# Granular timeout control
httpx.Timeout(
    timeout=120.0,  # Total operation timeout
    connect=60.0,   # Connection establishment timeout
    read=90.0,      # Reading response timeout
    write=30.0,     # Writing request timeout
    pool=5.0        # Acquiring connection from pool timeout
)
```

In this project, we use:
- **Total timeout**: 120 seconds (default, configurable)
- **Connect timeout**: 60 seconds (connection establishment)

### 6. uvicorn

**Purpose**: ASGI server for running FastAPI/Starlette applications.

```python
import uvicorn

# Run programmatically
uvicorn.run(
    app,  # FastAPI/Starlette app
    host="127.0.0.1",
    port=8000,
    log_level="info"
)

# Run as subprocess (run_all.py pattern)
subprocess.Popen([
    sys.executable,
    "-m", "uvicorn",
    "module.path:app",
    "--host", "127.0.0.1",
    "--port", "8000",
    "--log-level", "info"
])
```

### 7. python-dotenv

**Purpose**: Load environment variables from `.env` files.

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env from current directory

# Access variables
project_endpoint = os.environ["PROJECT_ENDPOINT"]
model_name = os.environ["MODEL_DEPLOYMENT_NAME"]
```

**Expected `.env` Format**:
```bash
PROJECT_ENDPOINT=https://xxx.services.ai.azure.com/api/projects/xxx
MODEL_DEPLOYMENT_NAME=gpt-4o
BING_PROJECT_CONNECTION_ID=connection-abc123  # Required for Content Agent with Bing grounding
SERVER_URL=127.0.0.1
TITLE_AGENT_PORT=8001
OUTLINE_AGENT_PORT=8002
CONTENT_AGENT_PORT=8004
ROUTING_AGENT_PORT=8003
AGENT_TIMEOUT=120  # Optional: timeout in seconds for A2A communication (default: 120)
```

---

## Deployment and Configuration

### Local Development Setup

#### Prerequisites

1. **Python 3.9+**
2. **Azure CLI** with authenticated session (`az login`)
3. **Azure AI Foundry Project**:
   - Project endpoint URL
   - Deployed AI model (e.g., GPT-4o)
4. **Required Python Packages** (see `requirements.txt`)

#### Configuration Steps

1. **Create `.env` file** in `python/` directory:
   ```bash
   PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
   MODEL_DEPLOYMENT_NAME=your-model-deployment-name
   BING_PROJECT_CONNECTION_ID=your-bing-connection-id
   SERVER_URL=127.0.0.1
   TITLE_AGENT_PORT=8001
   OUTLINE_AGENT_PORT=8002
   CONTENT_AGENT_PORT=8004
   ROUTING_AGENT_PORT=8003
   AGENT_TIMEOUT=120  # Optional: timeout in seconds for A2A communication (default: 120)
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate with Azure**:
   ```bash
   az login
   ```

4. **Run the system**:
   ```bash
   python run_all.py
   ```

### Running Individual Agents

#### Title Agent
```bash
uvicorn title_agent.server:app --host 127.0.0.1 --port 8001
```

#### Outline Agent
```bash
uvicorn outline_agent.server:app --host 127.0.0.1 --port 8002
```

#### Content Agent
```bash
uvicorn content_agent.server:app --host 127.0.0.1 --port 8004
```

#### Routing Agent
```bash
uvicorn routing_agent.server:app --host 127.0.0.1 --port 8003
```

#### Client (after agents are running)
```bash
python client.py
```

### Health Checks

Each agent exposes a health endpoint:

```bash
# Title Agent
curl http://127.0.0.1:8001/health

# Outline Agent
# Outline Agent
curl http://127.0.0.1:8002/health

# Content Agent
curl http://127.0.0.1:8004/health

# Routing Agent
curl http://127.0.0.1:8003/health
```

### Testing A2A Protocol

#### Get Agent Card
```bash
curl http://127.0.0.1:8001/api/a2a/agent-card
```

#### Send Message
```bash
curl -X POST http://127.0.0.1:8001/api/a2a/messages \
  -H "Content-Type: application/json" \
  -d '{
    "id": "msg-123",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Generate a title for AI agents"}],
        "messageId": "msg-123"
      }
    }
  }'
```

### Production Considerations

#### Azure AI Foundry Hosting

The agents themselves (TitleAgent, OutlineAgent, RoutingAgent) are **not deployed to Azure** in this architecture. Instead:

- **Azure AI Foundry Agent Service v2** hosts the **AI models and agent logic** (the LLM-powered decision-making)
- **Local/container servers** host the **A2A protocol endpoints and orchestration logic**

**Deployment Architecture**:

```
┌─────────────────────────────────────────────┐
│  Your Infrastructure (VM/Container/K8s)     │
│                                             │
│  ┌───────────────────────────────────────┐ │
│  │  Python Servers (FastAPI/Starlette)   │ │
│  │  - A2A protocol endpoints             │ │
│  │  - Request routing                    │ │
│  │  - Task orchestration                 │ │
│  └───────────────┬───────────────────────┘ │
└──────────────────┼─────────────────────────┘
                   │
                   │ HTTPS/Azure SDK
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Azure AI Foundry (Managed Service)          │
│  - AI Models (GPT-4o, etc.)                  │
│  - Agent Definitions                         │
│  - Conversation Management                   │
│  - Authentication/Authorization              │
└──────────────────────────────────────────────┘
```

#### Scaling Strategies

1. **Containerization**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "routing_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Kubernetes Deployment**:
   - Deploy each agent as a separate pod
   - Use Service resources for service discovery
   - Configure environment variables via ConfigMaps
   - Store secrets in Azure Key Vault

3. **Azure Container Apps**:
   - Serverless container platform
   - Auto-scaling based on HTTP requests
   - Built-in ingress and TLS

#### Security Best Practices

1. **Authentication**:
   - Use Managed Identity in Azure environments
   - Rotate credentials regularly
   - Use Azure Key Vault for secrets

2. **Network Security**:
   - Place agents in private VNet
   - Use Application Gateway for external access
   - Implement rate limiting

3. **Agent-to-Agent Communication**:
   - Use mTLS for A2A protocol
   - Implement agent authentication tokens
   - Validate AgentCard signatures

4. **Azure AI Foundry Security**:
   - Use Azure RBAC for access control
   - Enable audit logging
   - Implement content filtering policies

#### Monitoring and Observability

1. **Application Insights Integration**:
   ```python
   from opencensus.ext.azure.log_exporter import AzureLogHandler
   import logging
   
   logger = logging.getLogger(__name__)
   logger.addHandler(AzureLogHandler(
       connection_string='InstrumentationKey=...'
   ))
   ```

2. **Metrics to Track**:
   - Request latency (client → routing → specialist agent)
   - Agent response times
   - Azure AI Foundry API latency
   - Error rates and types
   - A2A message success/failure rates
   - **Timeout occurrences** by agent and timeout type
   - **Average response times** compared to timeout thresholds

3. **Logging Best Practices**:
   - Log all A2A message exchanges
   - Log Azure AI agent function calls
   - Include correlation IDs across services
   - Sanitize sensitive data from logs
   - **Log timeout events** with agent name and timeout duration
   - **Log slow requests** that approach timeout threshold

### Environment Variables Reference

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| `PROJECT_ENDPOINT` | Azure AI Foundry project URL | `https://xxx.services.ai.azure.com/api/projects/xxx` | *Required* |
| `MODEL_DEPLOYMENT_NAME` | Azure OpenAI model deployment | `gpt-4o` | *Required* |
| `BING_PROJECT_CONNECTION_ID` | Azure AI Foundry project connection ID for Bing Search (used by Content Agent) | `connection-abc123` | *Required for Content Agent* |
| `SERVER_URL` | Host for local servers | `127.0.0.1` or `0.0.0.0` | *Required* |
| `TITLE_AGENT_PORT` | Title Agent server port | `8001` | *Required* |
| `OUTLINE_AGENT_PORT` | Outline Agent server port | `8002` | *Required* |
| `CONTENT_AGENT_PORT` | Content Agent server port | `8004` | *Required* |
| `ROUTING_AGENT_PORT` | Routing Agent server port | `8003` | *Required* |
| `AGENT_TIMEOUT` | Timeout (seconds) for agent-to-agent communication | `120` | `120` |
| `HEALTH_CHECK_TIMEOUT` | Timeout (seconds) for waiting for agents to become ready | `60` | `60` |

**Timeout Configuration Guidelines**:

- **Default (120s)**: Suitable for most scenarios with typical LLM response times
- **Short (30-60s)**: For fast-failing systems that need quick error detection
- **Extended (300-600s)**: For complex tasks with long-running agent processing
- **Connection timeout**: Set to ~50% of total timeout (e.g., 60s for 120s total)

**Tuning Recommendations**:

1. Monitor actual agent response times in production
2. Set timeout to P95 response time + buffer (e.g., 2x P95)
3. Consider separate timeouts for different agent types if response times vary significantly
4. Log timeout events to identify agents that need optimization

---

## Summary

This system demonstrates a **production-ready architecture** for building distributed AI agent systems with:

1. **Clear separation of concerns**: Routing vs. specialist agents (Title, Outline, Content)
2. **Standardized communication**: A2A protocol for inter-agent messaging
3. **Cloud-native AI**: Azure AI Foundry Agent Service v2 for hosting intelligence
4. **Bing Search grounding**: Content Agent integrates real-time web search for factually grounded content
5. **Parallel batch processing**: All specialist agents execute concurrently via asyncio.gather()
6. **Async processing**: Non-blocking, concurrent request handling
7. **Extensibility**: Easy to add new specialist agents
8. **Observable**: Built-in health checks and task status tracking
9. **Reliable**: Configurable timeout protection for all agent-to-agent communication

The architecture is designed to scale horizontally, support multiple agent types, integrate seamlessly with Azure's managed AI services (including Bing Search), and provide robust error handling through configurable timeout mechanisms while maintaining flexibility for custom deployment scenarios. The system returns structured JSON output combining title, outline, and web-grounded content.
