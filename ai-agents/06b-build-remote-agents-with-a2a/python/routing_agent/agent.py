""" Azure AI Foundry Agent Service v2 - Routing Agent (using AIProjectClient with function tools) """

import asyncio
import json
import os
import uuid
import httpx

from typing import Any, Callable
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool
from collections.abc import Callable
from dotenv import load_dotenv
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str, timeout: float = 120.0):
        timeout_config = httpx.Timeout(timeout=timeout, connect=60.0)
        self._httpx_client = httpx.AsyncClient(timeout=timeout_config)
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message_request: SendMessageRequest) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)

    async def close(self):
        await self._httpx_client.aclose()


class RoutingAgent:
    """Routing Agent using Azure AI Foundry Agent Service v2 (AIProjectClient) with function tools."""

    def __init__(self, task_callback: TaskUpdateCallback | None = None, agent_timeout: float = 120.0):
        """Initialize the routing agent.
        
        Args:
            task_callback: Optional callback for task updates
            agent_timeout: Timeout in seconds for agent-to-agent communication (default: 120s)
        """
        self.task_callback = task_callback
        self.agent_timeout = agent_timeout
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''

        # Initialize sync Azure AI Projects client (v2 SDK)
        self.credential = DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True
        )
        self.project_client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=self.credential
        )
        self.openai_client = self.project_client.get_openai_client()

        self.azure_agent = None
        self.conversation_id: str | None = None
        self.last_response_id: str | None = None


    @classmethod
    async def create(cls, remote_agent_addresses: list[str], task_callback: TaskUpdateCallback | None = None, agent_timeout: float = 120.0) -> 'RoutingAgent':
        """Create and asynchronously initialize an instance of the RoutingAgent.
        
        Args:
            remote_agent_addresses: List of agent URLs to connect to
            task_callback: Optional callback for task updates
            agent_timeout: Timeout in seconds for agent-to-agent communication (default: 120s)
        """
        instance = cls(task_callback, agent_timeout=agent_timeout)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def list_remote_agents(self) -> str:
        """List all connected remote agents."""
        if not self.remote_agent_connections:
            return "[]"

        lines = []
        for card in self.cards.values():
            lines.append(f"{card.name}: {card.description}")

        return "[\n  " + ",\n  ".join(lines) + "\n]"
    

    async def _async_init_components(self, remote_agent_addresses: list[str]) -> None:
        """Asynchronous part of initialization - connect to remote agents."""
        timeout_config = httpx.Timeout(timeout=self.agent_timeout, connect=60.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, 
                        agent_url=address, 
                        timeout=self.agent_timeout
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f'ERROR: Failed to get agent card from {address}: {e}')
                except httpx.TimeoutException as e:
                    print(f'ERROR: Timeout connecting to {address}: {e}')
                except Exception as e:
                    print(f'ERROR: Failed to initialize connection for {address}: {e}')
            print(f"Found remote agents: {self.list_remote_agents()}")

    async def send_message_to_agent(self, agent_name: str, task: str) -> str:
        """
        Send a task to a remote agent via A2A protocol.
        This function is exposed as a tool to the Azure AI Agent.
        """
        if agent_name not in self.remote_agent_connections:
            return json.dumps({"error": f"Agent '{agent_name}' not found"})

        client = self.remote_agent_connections[agent_name]
        if not client:
            return json.dumps({"error": f"Client not available for {agent_name}"})

        message_id = str(uuid.uuid4())

        # Construct the A2A payload
        payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': task}],
                'messageId': message_id,
            },
        }

        # Wrap in SendMessageRequest
        message_request = SendMessageRequest(
            id=message_id,
            params=MessageSendParams.model_validate(payload)
        )

        try:
            # Send via A2A client
            send_response: SendMessageResponse = await client.send_message(message_request=message_request)

            if not isinstance(send_response.root, SendMessageSuccessResponse):
                return json.dumps({"error": "Received non-success response"})

            if not isinstance(send_response.root.result, Task):
                return json.dumps({"error": "Received non-task response"})

            # Return the task result as JSON string
            return json.dumps(send_response.root.result.model_dump())
        except httpx.TimeoutException as e:
            error_msg = f"Timeout waiting for response from {agent_name} (timeout: {self.agent_timeout}s)"
            print(f"ERROR: {error_msg}")
            return json.dumps({"error": error_msg})
        except httpx.ReadTimeout as e:
            error_msg = f"Read timeout from {agent_name} (timeout: {self.agent_timeout}s)"
            print(f"ERROR: {error_msg}")
            return json.dumps({"error": error_msg})
        except httpx.ConnectTimeout as e:
            error_msg = f"Connection timeout to {agent_name}"
            print(f"ERROR: {error_msg}")
            return json.dumps({"error": error_msg})
        except Exception as e:
            return json.dumps({"error": str(e)})


    async def create_agent(self):
        """
        Create an Azure AI Agent with function tools for A2A communication.
        Uses v2 SDK with FunctionTool for tool definition.
        """
        try:
            # Define the send_message_to_agent function as a tool
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

            # Create Azure AI Agent with function tool (v2 pattern)
            def _create():
                return self.project_client.agents.create_version(
                    agent_name="routing-agent-v2",
                    definition=PromptAgentDefinition(
                        model=os.environ["MODEL_DEPLOYMENT_NAME"],
                        instructions=f"""
                        You are an expert Routing Delegator that helps users with requests.

                        Your role:
                        - Delegate user inquiries to appropriate specialized remote agents
                        - Provide clear and helpful responses to users

                        Available Agents: {self.list_remote_agents()}

                        Always be helpful and route requests to the most appropriate agent.
                        Use the send_message_to_agent function to communicate with remote agents.""",
                        tools=[send_message_tool],
                    ),
                )

            self.azure_agent = await asyncio.to_thread(_create)

            # Create a conversation for multi-turn interactions
            def _create_conversation():
                return self.openai_client.conversations.create()
            
            conversation = await asyncio.to_thread(_create_conversation)
            self.conversation_id = conversation.id

            return self.azure_agent

        except Exception as e:
            print(f"Error creating Azure AI agent: {e}")
            raise

    async def process_user_message(self, user_message: str) -> str:
        """
        Process a user message by calling all three agents in parallel.
        Returns JSON with title, outline, and content.
        Fails completely if any agent fails.
        """
        if not self.azure_agent:
            return json.dumps({"error": "Azure AI Agent not initialized. Please ensure the agent is properly created."})

        if not self.conversation_id:
            return json.dumps({"error": "Azure AI Conversation not initialized. Please ensure the agent is properly created."})

        try:
            # Call all three agents in parallel
            title_task = self.send_message_to_agent("AI Foundry Title Agent", user_message)
            outline_task = self.send_message_to_agent("AI Foundry Outline Agent", user_message)
            content_task = self.send_message_to_agent("AI Foundry Content Agent", user_message)

            # Wait for all agents to complete (will raise exception if any fails)
            title_result, outline_result, content_result = await asyncio.gather(
                title_task, outline_task, content_task
            )
            
            # Parse results from A2A responses
            title_data = json.loads(title_result)
            outline_data = json.loads(outline_result)
            content_data = json.loads(content_result)

            # Safe print function that handles Unicode characters
            def safe_print(message):
                try:
                    print(message)
                except UnicodeEncodeError:
                    # Fallback: print with ASCII encoding and replace problematic characters
                    print(message.encode('ascii', errors='replace').decode('ascii'))

            #safe_print(f"Title Agent - Sending to Title Agent: {user_message}")
            #safe_print(f"Title Agent - Getting from Title Agent: {title_data}")
            #safe_print(f"Outline Agent - Sending to Outline Agent: {user_message}")
            #safe_print(f"Outline Agent - Getting from Outline Agent: {outline_data}")
            #safe_print(f"Content Agent - Sending to Content Agent: {user_message}")
            #safe_print(f"Content Agent - Getting from Content Agent: {content_data}")
            
            # Check for errors in any response
            if "error" in title_data:
                raise Exception(f"Title Agent error: {title_data['error']}")
            if "error" in outline_data:
                raise Exception(f"Outline Agent error: {outline_data['error']}")
            if "error" in content_data:
                raise Exception(f"Content Agent error: {content_data['error']}")
            
            # Extract the actual content from the task response
            def extract_content(task_data):
                # First, try to get from status.message.parts (this is where the final response is)
                if "status" in task_data and task_data["status"] is not None:
                    status = task_data["status"]
                    if "message" in status and status["message"] is not None:
                        message = status["message"]
                        if "parts" in message and message["parts"] is not None and len(message["parts"]) > 0:
                            if "text" in message["parts"][0]:
                                return message["parts"][0]["text"]
                
                # Fallback: try to extract from artifacts
                if "artifacts" in task_data and task_data["artifacts"] is not None and len(task_data["artifacts"]) > 0:
                    artifacts = task_data["artifacts"]
                    if "parts" in artifacts[0] and artifacts[0]["parts"] is not None and len(artifacts[0]["parts"]) > 0:
                        parts = artifacts[0]["parts"]
                        if "text" in parts[0]:
                            return parts[0]["text"]
                
                # Another fallback: try to get from history (last agent message)
                if "history" in task_data and task_data["history"] is not None and len(task_data["history"]) > 0:
                    for message in reversed(task_data["history"]):
                        if message.get("role") == "agent" and "parts" in message and message["parts"] is not None:
                            parts = message["parts"]
                            if len(parts) > 0 and "text" in parts[0]:
                                return parts[0]["text"]
                
                return "No content available"
            
            title_text = extract_content(title_data)
            outline_text = extract_content(outline_data)
            content_text = extract_content(content_data)
            
            # Return combined JSON result with Unicode characters preserved
            result = {
                "title": title_text,
                "outline": outline_text,
                "content": content_text
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            error_msg = f"Routing-Agent - Error in process_user_message: {e}"
            print(error_msg)
            raise  # Re-raise to ensure failure propagates

    async def close(self):
        """Close all client connections."""
        for conn in self.remote_agent_connections.values():
            await conn.close()
        
        def _close():
            self.openai_client.close()
            self.project_client.close()
        await asyncio.to_thread(_close)


async def get_initialized_routing_agent() -> RoutingAgent:
    """Create and initialize a RoutingAgent asynchronously."""
    routing_agent_instance = await RoutingAgent.create(
        remote_agent_addresses=[
            f"http://{os.environ['SERVER_URL']}:{os.environ['TITLE_AGENT_PORT']}",
            f"http://{os.environ['SERVER_URL']}:{os.environ['OUTLINE_AGENT_PORT']}",
            f"http://{os.environ['SERVER_URL']}:{os.environ['CONTENT_AGENT_PORT']}",
        ]
    )
    # Create the Azure AI agent
    await routing_agent_instance.create_agent()
    return routing_agent_instance
