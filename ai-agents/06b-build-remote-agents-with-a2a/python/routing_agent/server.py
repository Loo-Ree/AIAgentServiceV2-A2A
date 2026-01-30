import os
import asyncio
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from routing_agent.agent import RoutingAgent  

load_dotenv()

routing_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global routing_agent
    print("Starting up: Initializing routing agent...")
    
    # Get configurable timeout from environment (default: 120 seconds)
    agent_timeout = float(os.getenv('AGENT_TIMEOUT', '120'))
    print(f"Using agent communication timeout: {agent_timeout}s")
    
    routing_agent = await RoutingAgent.create(
        remote_agent_addresses=[
            f"http://{os.environ['SERVER_URL']}:{os.environ['TITLE_AGENT_PORT']}",
            f"http://{os.environ['SERVER_URL']}:{os.environ['OUTLINE_AGENT_PORT']}",
            f"http://{os.environ['SERVER_URL']}:{os.environ['CONTENT_AGENT_PORT']}",
        ],
        agent_timeout=agent_timeout
    )
    await routing_agent.create_agent()  # Now async in v2
    print("Routing agent initialized.")
    yield
    # Cleanup on shutdown
    if routing_agent:
        await routing_agent.close()

app = FastAPI(lifespan=lifespan)

@app.post("/message")
async def handle_message(request: Request):
    print("Agent: Processing request, please wait.")

    data = await request.json()
    user_message = data.get("message")

    if not user_message:
        return {"error": "No message provided."}
    
    try:
        response = await routing_agent.process_user_message(user_message)

    except Exception as e:
        return {"error": f"Failed to process message: {str(e)}"}
    
    return {"response": response}

@app.get("/health")
async def health_check():
    return {"status": "Routing agent is running!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv["ROUTING_AGENT_PORT"])
    uvicorn.run("routing_main:app", host="127.0.0.1", port=port, reload=True)
