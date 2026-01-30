from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

with (
    DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    ) as credential,
    AIProjectClient(
        endpoint="https://aif-swcv2cu.services.ai.azure.com/api/projects/projectv2cu",
        credential=credential
    ) as project_client,
):
    # List all agents in the project
    agents = project_client.agents.list()
    for agent in agents:
        print(f"Agent: {agent.name} (ID: {agent.id})")