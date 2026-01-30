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

### Key Capabilities

The system consists of three specialized agents:

1. **Title Agent** - Generates catchy blog post titles from topics
2. **Outline Agent** - Creates structured outlines for articles
3. **Routing Agent** - Intelligently delegates user requests to specialized agents

### Use Case

A user sends a natural language request (e.g., "Create a title for an article about Python async programming"). The Routing Agent analyzes the request, determines which specialized agent should handle it, delegates the task via the A2A protocol, and returns the result to the user.

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
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│  Title Agent Server     │   │  Outline Agent Server    │
│  (title_agent/)         │   │  (outline_agent/)        │
│                         │   │                          │
│  ┌──────────────────┐   │   │  ┌────────────────────┐  │
│  │ A2A Server       │   │   │  │ A2A Server         │  │
│  │ (server.py)      │   │   │  │ (server.py)        │  │
│  │ - AgentCard      │   │   │  │ - AgentCard        │  │
│  │ - A2A Routes     │   │   │  │ - A2A Routes       │  │
│  └────────┬─────────┘   │   │  └──────────┬─────────┘  │
│           │             │   │             │            │
│  ┌────────▼─────────┐   │   │  ┌──────────▼─────────┐  │
│  │ AgentExecutor    │   │   │  │ AgentExecutor      │  │
│  │ (agent_executor) │   │   │  │ (agent_executor)   │  │
│  │ - Task handling  │   │   │  │ - Task handling    │  │
│  │ - Status updates │   │   │  │ - Status updates   │  │
│  └────────┬─────────┘   │   │  └──────────┬─────────┘  │
│           │             │   │             │            │
│  ┌────────▼─────────┐   │   │  ┌──────────▼─────────┐  │
│  │ TitleAgent       │   │   │  │ OutlineAgent       │  │
│  │ (agent.py)       │   │   │  │ (agent.py)         │  │
│  │ - AIProjectClient│   │   │  │ - AIProjectClient  │  │
│  │ - Agent v2       │   │   │  │ - Agent v2         │  │
│  └──────────────────┘   │   │  └────────────────────┘  │
└─────────────────────────┘   └──────────────────────────┘
            │                            │
            └────────────┬───────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │   Azure AI Foundry Service v2       │
        │   - Agent Hosting                   │
        │   - Model Deployment                │
        │   - Conversation Management         │
        └─────────────────────────────────────┘
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

#### 3. **Specialist Agent Layer** (`title_agent/`, `outline_agent/`)
- **Purpose**: Domain-specific task execution
- **Components**:
  - `server.py`: A2A-compliant server exposing agent capabilities
  - `agent_executor.py`: Task lifecycle management
  - `agent.py`: Azure AI Foundry agent implementation

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
   - Maintain A2A client connections

2. **Azure AI Foundry Integration**:
   - Create and manage Azure AI agent with function tools
   - Maintain conversation state for multi-turn interactions
   - Handle function calling loop for agent delegation

3. **Request Processing**:
   - Accept user messages
   - Let Azure AI agent decide which remote agent to invoke
   - Execute function calls to delegate tasks
   - Return final responses to user

**Key Classes and Methods**:

```python
class RemoteAgentConnections:
    """Wrapper for A2A client connection to a remote agent"""
    
    def __init__(self, agent_card: AgentCard, agent_url: str)
        # Initialize httpx client and A2AClient
        # Store agent card metadata
    
    async def send_message(self, message_request: SendMessageRequest)
        # Send A2A message to remote agent
        # Returns SendMessageResponse

class RoutingAgent:
    """Main routing agent with Azure AI Foundry integration"""
    
    def __init__(self, task_callback: Optional[TaskUpdateCallback])
        # Initialize Azure clients:
        # - DefaultAzureCredential for authentication
        # - AIProjectClient for agent management
        # - OpenAI client for conversation API
        
    @classmethod
    async def create(cls, remote_agent_addresses: list[str])
        # Factory method for async initialization
        # 1. Create instance
        # 2. Connect to all remote agents
        # 3. Return initialized instance
    
    async def _async_init_components(self, remote_agent_addresses: list[str])
        # Connect to each remote agent via A2A protocol
        # Retrieve AgentCard from each agent
        # Store connections in dictionary
    
    def list_remote_agents(self) -> str
        # Return formatted list of available remote agents
        # Used in Azure AI agent instructions
    
    async def send_message_to_agent(self, agent_name: str, task: str) -> str
        # Function exposed as tool to Azure AI agent
        # 1. Validate agent exists
        # 2. Create A2A message request
        # 3. Send to remote agent
        # 4. Return JSON-serialized response
    
    async def create_agent(self)
        # Create Azure AI agent with function tools
        # 1. Define FunctionTool for send_message_to_agent
        # 2. Create agent with PromptAgentDefinition
        # 3. Create conversation for multi-turn chat
        # 4. Store agent and conversation IDs
    
    async def process_user_message(self, user_message: str) -> str
        # Main processing loop
        # 1. Send user message to Azure AI agent
        # 2. Check for function calls in response
        # 3. Execute function calls (delegate to remote agents)
        # 4. Send tool outputs back to Azure AI agent
        # 5. Repeat until no more function calls
        # 6. Return final text response
    
    async def close(self)
        # Cleanup all connections
        # - Close A2A client connections
        # - Close OpenAI client
        # - Close project client
```

---

### 3. Specialist Agent Components (Title & Outline)

Both Title Agent and Outline Agent follow the same architectural pattern, differing only in their specific AI instructions.

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
    
    def __init__(self, card: AgentCard)
        # Store agent card
        # Initialize foundry agent reference (lazy load)
    
    async def _get_or_create_agent(self)
        # Lazy initialization of Azure AI agent
        # Returns TitleAgent or OutlineAgent instance
    
    async def _process_request(self, message_parts, context_id, task_updater)
        # Main processing logic:
        # 1. Extract text from A2A message parts
        # 2. Get or create Foundry agent
        # 3. Update task status to "working"
        # 4. Run agent conversation
        # 5. Stream responses as task updates
        # 6. Mark task as complete
        # 7. Handle errors with failed status
    
    async def execute(self, context: RequestContext, event_queue: EventQueue)
        # Called by A2A framework to execute a task
        # 1. Create TaskUpdater
        # 2. Submit task
        # 3. Start work
        # 4. Process request
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue)
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
    
    def __init__(self)
        # Initialize Azure credentials (DefaultAzureCredential)
        # Create AIProjectClient
        # Get OpenAI client from project client
        # Initialize agent and conversation state
    
    async def create_agent(self)
        # Create agent using v2 API
        # Define agent with:
        #   - agent_name: Unique identifier
        #   - model: Deployment name from environment
        #   - instructions: Agent's system prompt
        # Wrap sync API call in asyncio.to_thread()
        # Return created agent
    
    async def run_conversation(self, user_message: str) -> list[str]
        # Run a stateless conversation:
        # 1. Create new conversation
        # 2. Send user message
        # 3. Get response from agent
        # 4. Return output text
        # All sync calls wrapped in asyncio.to_thread()
    
    async def close(self)
        # Close OpenAI client
        # Close project client
        # Wrapped in asyncio.to_thread()
```

---

## Request Flow

### End-to-End Request Flow

Let's trace a complete request: **"Generate a title for an article about Python async programming"**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 1: User Input                                                  │
└──────────────────────────────────────────────────────────────────────┘

User (client.py)
    │
    └─► send_prompt("Generate a title for Python async programming")
            │
            └─► HTTP POST http://127.0.0.1:8003/message
                Body: {"message": "Generate a title for Python async programming"}

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 2: Routing Agent Receives Request                             │
└──────────────────────────────────────────────────────────────────────┘

routing_agent/server.py
    │
    ├─► @app.post("/message")
    │   async def handle_message(request: Request)
    │       │
    │       ├─► Extract: user_message = "Generate a title..."
    │       │
    │       └─► routing_agent.process_user_message(user_message)
    │
    └─► routing_agent/agent.py: RoutingAgent.process_user_message()

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 3: Azure AI Agent Processing                                  │
└──────────────────────────────────────────────────────────────────────┘

RoutingAgent.process_user_message()
    │
    ├─► 1. Send message to Azure AI agent via OpenAI API
    │      openai_client.responses.create(
    │          conversation=self.conversation_id,
    │          input=user_message,
    │          extra_body={"agent": {...}}
    │      )
    │
    ├─► 2. Azure AI Agent analyzes request
    │      Instructions: "Delegate to appropriate agent..."
    │      Available tools: send_message_to_agent
    │      Available agents: Title Agent, Outline Agent
    │      
    │      Decision: "This is a title generation task"
    │      → Returns function_call: send_message_to_agent
    │
    └─► 3. Response contains function call:
           {
               "type": "function_call",
               "name": "send_message_to_agent",
               "call_id": "call_abc123",
               "arguments": {
                   "agent_name": "AI Foundry Title Agent",
                   "task": "Generate a title for Python async programming"
               }
           }

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 4: Function Call Execution                                    │
└──────────────────────────────────────────────────────────────────────┘

RoutingAgent.send_message_to_agent()
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
    │              "parts": [{"kind": "text", "text": "Generate a title..."}],
    │              "messageId": message_id
    │          }
    │      }
    │
    └─► 4. Send via A2A protocol
           send_response = await client.send_message(message_request)
           
           └─► HTTP POST http://127.0.0.1:8001/api/a2a/messages
               Headers: Content-Type: application/json
               Body: A2A protocol message

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 5: Title Agent Processes Request                              │
└──────────────────────────────────────────────────────────────────────┘

title_agent/server.py
    │
    └─► A2AStarletteApplication receives message
            │
            └─► DefaultRequestHandler.handle()
                    │
                    └─► FoundryAgentExecutor.execute()

title_agent/agent_executor.py: FoundryAgentExecutor.execute()
    │
    ├─► 1. Create TaskUpdater (for A2A status updates)
    │      updater = TaskUpdater(event_queue, task_id, context_id)
    │      await updater.submit()
    │      await updater.start_work()
    │
    ├─► 2. Extract user message from A2A parts
    │      user_message = message_parts[0].root.text
    │
    ├─► 3. Get TitleAgent instance
    │      agent = await self._get_or_create_agent()
    │
    ├─► 4. Update task status
    │      await updater.update_status(
    │          TaskState.working,
    │          message="Title Agent is processing..."
    │      )
    │
    └─► 5. Run agent conversation
           responses = await agent.run_conversation(user_message)

title_agent/agent.py: TitleAgent.run_conversation()
    │
    ├─► 1. Create new conversation with Azure AI
    │      conversation = openai_client.conversations.create()
    │
    ├─► 2. Send message to Azure AI agent
    │      response = openai_client.responses.create(
    │          conversation=conversation.id,
    │          input=user_message,
    │          extra_body={"agent": {"name": "title-agent-v2", ...}}
    │      )
    │
    ├─► 3. Azure AI Model processes request
    │      Model: gpt-4o (or configured model)
    │      Instructions: "Generate catchy blog post title..."
    │      Input: "Generate a title for Python async programming"
    │      
    │      → Model generates: "Mastering Python Async: A Developer's Guide"
    │
    └─► 4. Return response text

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 6: Title Agent Completes Task                                 │
└──────────────────────────────────────────────────────────────────────┘

FoundryAgentExecutor._process_request()
    │
    ├─► responses = ["Mastering Python Async: A Developer's Guide"]
    │
    ├─► Update task with response
    │   await updater.update_status(
    │       TaskState.working,
    │       message="Mastering Python Async: A Developer's Guide"
    │   )
    │
    └─► Complete task
        await updater.complete(
            message="Mastering Python Async: A Developer's Guide"
        )

A2A Response sent back to Routing Agent:
    {
        "result": {
            "id": "task_123",
            "state": "completed",
            "artifacts": [{
                "parts": [{"text": "Mastering Python Async: A Developer's Guide"}]
            }]
        }
    }

┌──────────────────────────────────────────────────────────────────────┐
│ Phase 7: Routing Agent Sends Tool Output Back                       │
└──────────────────────────────────────────────────────────────────────┘

RoutingAgent.process_user_message()
    │
    ├─► Parse Title Agent response
    │   result = json.loads(response)
    │   
    ├─► Create tool output
    │   tool_outputs = [{
    │       "type": "function_call_output",
    │       "call_id": "call_abc123",
    │       "output": "{...task result...}"
    │   }]
    │
    ├─► Send tool outputs back to Azure AI agent
    │   response = openai_client.responses.create(
    │       input=tool_outputs,
    │       previous_response_id=self.last_response_id,
    │       extra_body={"agent": {...}}
    │   )
    │
    └─► Azure AI agent processes tool output
        → Generates final user-facing response:
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
        "Agent: I've generated a title for your article: 
         'Mastering Python Async: A Developer's Guide'"
```

### Key Flow Observations

1. **Two-Level Agent Hierarchy**:
   - Routing Agent: High-level orchestration with Azure AI decision-making
   - Specialist Agents: Domain-specific execution with Azure AI capabilities

2. **Protocol Separation**:
   - User ↔ Routing Agent: Simple HTTP JSON
   - Routing Agent ↔ Specialist Agents: A2A protocol (standardized agent communication)
   - All Agents ↔ Azure AI: OpenAI-compatible API

3. **Asynchronous Processing**:
   - All I/O operations use async/await
   - Multiple agents can be queried concurrently
   - Non-blocking server implementations

4. **State Management**:
   - Routing Agent: Maintains conversation state across user turns
   - Specialist Agents: Stateless per A2A request
   - Azure AI: Manages conversation history in cloud

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

async with httpx.AsyncClient() as client:
    card_resolver = A2ACardResolver(client, agent_url)
    agent_card = await card_resolver.get_agent_card()
    # Returns AgentCard from {agent_url}/api/a2a/agent-card
```

#### A2AClient - Message Sending

```python
from a2a.client import A2AClient
import httpx

httpx_client = httpx.AsyncClient(timeout=30)
a2a_client = A2AClient(httpx_client, agent_card, url=agent_url)

# Send message
response = await a2a_client.send_message(message_request)
# POSTs to {agent_url}/api/a2a/messages
```

**Connection Pattern in This Project**:

```python
class RemoteAgentConnections:
    """Wrapper for A2A client connection"""
    
    def __init__(self, agent_card: AgentCard, agent_url: str):
        self._httpx_client = httpx.AsyncClient(timeout=30)
        self.agent_client = A2AClient(
            self._httpx_client,
            agent_card,
            url=agent_url
        )
        self.card = agent_card
    
    async def send_message(self, message_request):
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

# Async context manager
async with httpx.AsyncClient(timeout=30) as client:
    response = await client.get("http://example.com")
    response = await client.post(
        "http://example.com/api",
        json={"key": "value"}
    )

# Long-lived client
client = httpx.AsyncClient(timeout=30)
try:
    response = await client.get("...")
finally:
    await client.aclose()
```

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
SERVER_URL=127.0.0.1
TITLE_AGENT_PORT=8001
OUTLINE_AGENT_PORT=8002
ROUTING_AGENT_PORT=8003
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
   SERVER_URL=127.0.0.1
   TITLE_AGENT_PORT=8001
   OUTLINE_AGENT_PORT=8002
   ROUTING_AGENT_PORT=8003
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
curl http://127.0.0.1:8002/health

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

3. **Logging Best Practices**:
   - Log all A2A message exchanges
   - Log Azure AI agent function calls
   - Include correlation IDs across services
   - Sanitize sensitive data from logs

### Environment Variables Reference

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| `PROJECT_ENDPOINT` | Azure AI Foundry project URL | `https://xxx.services.ai.azure.com/api/projects/xxx` | *Required* |
| `MODEL_DEPLOYMENT_NAME` | Azure OpenAI model deployment | `gpt-4o` | *Required* |
| `SERVER_URL` | Host for local servers | `127.0.0.1` or `0.0.0.0` | *Required* |
| `TITLE_AGENT_PORT` | Title Agent server port | `8001` | *Required* |
| `OUTLINE_AGENT_PORT` | Outline Agent server port | `8002` | *Required* |
| `ROUTING_AGENT_PORT` | Routing Agent server port | `8003` | *Required* |
| `AGENT_TIMEOUT` | Timeout (seconds) for agent-to-agent communication | `120` | `120` |
| `HEALTH_CHECK_TIMEOUT` | Timeout (seconds) for waiting for agents to become ready | `60` | `60` |

---

## Summary

This system demonstrates a **production-ready architecture** for building distributed AI agent systems with:

1. **Clear separation of concerns**: Routing vs. specialist agents
2. **Standardized communication**: A2A protocol for inter-agent messaging
3. **Cloud-native AI**: Azure AI Foundry Agent Service v2 for hosting intelligence
4. **Async processing**: Non-blocking, concurrent request handling
5. **Extensibility**: Easy to add new specialist agents
6. **Observable**: Built-in health checks and task status tracking

The architecture is designed to scale horizontally, support multiple agent types, and integrate seamlessly with Azure's managed AI services while maintaining flexibility for custom deployment scenarios.
