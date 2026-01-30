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

    def __init__(self, agent_card: AgentCard, agent_url: str):
        self._httpx_client = httpx.AsyncClient(timeout=30)
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

    def __init__(self, task_callback: TaskUpdateCallback | None = None):
        self.task_callback = task_callback
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
    async def create(cls, remote_agent_addresses: list[str], task_callback: TaskUpdateCallback | None = None) -> 'RoutingAgent':
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback)
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
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(agent_card=card, agent_url=address)
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f'ERROR: Failed to get agent card from {address}: {e}')
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
        Process a user message using v2 SDK with function calling.
        Handles tool calls manually and uses previous_response_id for multi-turn.
        """
        if not self.azure_agent:
            return "Azure AI Agent not initialized. Please ensure the agent is properly created."

        if not self.conversation_id:
            return "Azure AI Conversation not initialized. Please ensure the agent is properly created."

        try:
            def _send_message():
                # Build the request with previous_response_id for multi-turn
                extra_body = {"agent": {"name": self.azure_agent.name, "type": "agent_reference"}}
                
                if self.last_response_id:
                    # Continue the conversation with previous_response_id
                    return self.openai_client.responses.create(
                        input=user_message,
                        previous_response_id=self.last_response_id,
                        extra_body=extra_body,
                    )
                else:
                    # First message in the conversation
                    return self.openai_client.responses.create(
                        conversation=self.conversation_id,
                        input=user_message,
                        extra_body=extra_body,
                    )

            response = await asyncio.to_thread(_send_message)
            self.last_response_id = response.id

            # Process function calls if any
            while True:
                function_calls = [item for item in response.output if item.type == "function_call"]
                
                if not function_calls:
                    # No more function calls, return the final text response
                    break

                # Process each function call
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
                            "output": result
                        })

                # Send the tool outputs back and get next response
                def _send_tool_outputs():
                    return self.openai_client.responses.create(
                        input=tool_outputs,
                        previous_response_id=self.last_response_id,
                        extra_body={"agent": {"name": self.azure_agent.name, "type": "agent_reference"}},
                    )

                response = await asyncio.to_thread(_send_tool_outputs)
                self.last_response_id = response.id

            return response.output_text if response.output_text else "No response received from agent."

        except Exception as e:
            error_msg = f"Error in process_user_message: {e}"
            print(error_msg)
            return "An error occurred while processing your message."

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
        ]
    )
    # Create the Azure AI agent
    await routing_agent_instance.create_agent()
    return routing_agent_instance
