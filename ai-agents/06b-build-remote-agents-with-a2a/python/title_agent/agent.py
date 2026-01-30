""" Azure AI Foundry Agent Service v2 - Title Agent (using AIProjectClient) """

import asyncio
import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition


class TitleAgent:
    """Title Agent using Azure AI Foundry Agent Service v2 (AIProjectClient)."""

    def __init__(self):
        # Create the sync clients (v2 SDK)
        self.credential = DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True
        )
        self.project_client = AIProjectClient(
            endpoint=os.environ['PROJECT_ENDPOINT'],
            credential=self.credential
        )
        self.openai_client = self.project_client.get_openai_client()
        self.agent = None
        self.conversation_id: str | None = None
        self.last_response_id: str | None = None

    async def create_agent(self):
        """Create the title agent."""
        if self.agent:
            return self.agent

        # Create the title agent using v2 API (sync call wrapped for async compatibility)
        def _create():
            return self.project_client.agents.create_version(
                agent_name='title-agent-v2',
                definition=PromptAgentDefinition(
                    model=os.environ['MODEL_DEPLOYMENT_NAME'],
                    instructions="""
                    You are a helpful writing assistant.
                    Given a topic the user wants to write about, suggest a single clear and catchy blog post title.
                    """,
                ),
            )
        
        self.agent = await asyncio.to_thread(_create)
        return self.agent

    async def run_conversation(self, user_message: str) -> list[str]:
        """Run a conversation with the agent."""
        if not self.agent:
            await self.create_agent()

        def _run_sync():
            # Create a new conversation for each request (stateless per A2A call)
            conversation = self.openai_client.conversations.create()
            
            # Send user message and get response
            response = self.openai_client.responses.create(
                conversation=conversation.id,
                input=user_message,
                extra_body={"agent": {"name": self.agent.name, "type": "agent_reference"}},
            )
            
            return response.output_text if response.output_text else 'No response received'

        result = await asyncio.to_thread(_run_sync)
        return [result]

    async def close(self):
        """Close the client connections."""
        def _close():
            self.openai_client.close()
            self.project_client.close()
        await asyncio.to_thread(_close)


async def create_foundry_title_agent() -> TitleAgent:
    """Factory function to create and initialize a TitleAgent."""
    agent = TitleAgent()
    await agent.create_agent()
    return agent
