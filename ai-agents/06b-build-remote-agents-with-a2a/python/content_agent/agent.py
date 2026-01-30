""" Azure AI Foundry Agent Service v2 - Content Agent with Bing Search (using AIProjectClient) """

import asyncio
import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, BingGroundingAgentTool, BingGroundingSearchConfiguration, BingGroundingSearchToolParameters

class ContentAgent:
    """Content Agent using Azure AI Foundry Agent Service v2 (AIProjectClient) with Bing Search."""

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

        connection_id = os.environ.get('BING_PROJECT_CONNECTION_ID')

    async def create_agent(self):
        """Create the content agent with Bing search tool."""
        if self.agent:
            return self.agent
        
        connection_id = os.environ["BING_PROJECT_CONNECTION_ID"]
        print(f"Grounding with Bing Search connection ID: {connection_id}")

        # Create the content agent using v2 API with Bing search tool
        def _create():
            return self.project_client.agents.create_version(
                agent_name='content-agent-v2',
                definition=PromptAgentDefinition(
                    model=os.environ['MODEL_DEPLOYMENT_NAME'],
                    instructions="""
                    You are a helpful writing assistant specialized in creating blog content.
                    
                    Given a topic, title, or outline, generate brief, engaging blog content.
                    Your content should be:
                    - Maximum 100 words
                    - Well-structured with clear paragraphs
                    - Informative and engaging
                    - Based on current and accurate information
                    
                    Use the Bing search tool to gather current, accurate information to ground your content.
                    Focus on quality over quantity - keep it concise and valuable.
                    """,
                    tools=[
                        BingGroundingAgentTool(
                            bing_grounding=BingGroundingSearchToolParameters(
                                search_configurations=[
                                    BingGroundingSearchConfiguration(
                                        project_connection_id=connection_id
                                    )
                                ]
                            )
                        )
                    ],
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


async def create_foundry_content_agent() -> ContentAgent:
    """Factory function to create and initialize a ContentAgent."""
    agent = ContentAgent()
    await agent.create_agent()
    return agent
