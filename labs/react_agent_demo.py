import asyncio
import logging
import operator
import os
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


# SECTION 1: CONFIGURATION & VERSIONING
# Key point: Version EVERYTHING - prompts, models, tools, configs
class Environment(str, Enum):
    """Deployment environments with different configurations."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AgentConfig(BaseModel):
    """
    Versioned configuration for the agent.
    In production: pin exact model versions, not just 'gpt-4'.
    """
    version: str = "1.0.0"
    model_name: str = "gpt-4o"# Pin exact version in prod: "gpt-4-0613"
    temperature: float = 0.0  # 0 for deterministic production behavior
    max_iterations: int = 10  # Prevent infinite loops
    recursion_limit: int = 25  # LangGraph hard stop
    timeout_seconds: int = 60  # Global timeout
    environment: Environment = Environment.DEVELOPMENT
    prompt_version: str = "v1.0.0"

    # Feature flags for gradual rollout / instant rollback
    verbose_logging: bool = True


# Load config based on environment
def load_config(env: Environment = Environment.DEVELOPMENT) -> AgentConfig:
    """Load environment-specific configuration."""
    configs = {
        Environment.DEVELOPMENT: AgentConfig(
            temperature=0.7,
            verbose_logging=True,
            environment=Environment.DEVELOPMENT,
        ),
        Environment.STAGING: AgentConfig(
            temperature=0.3,
            verbose_logging=True,
            environment=Environment.STAGING,
        ),
        Environment.PRODUCTION: AgentConfig(
            temperature=0.0,
            verbose_logging=False,  # Structured logging only
            environment=Environment.PRODUCTION,
        ),
    }
    return configs.get(env, configs[Environment.DEVELOPMENT])


CONFIG = load_config()

# SECTION 2: LOGGING & OBSERVABILITY
# Key point: Log EVERYTHING - thoughts, actions, observations for debugging
logging.basicConfig(
    level=logging.DEBUG if CONFIG.verbose_logging else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class AgentTrace:
    """Simple tracing for observability - in prod use LangSmith or similar."""

    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.start_time = datetime.now()
        self.tokens_used = 0

    def log_step(self, step_type: str, content: str, metadata: dict | None = None):
        step = {
            "timestamp": datetime.now().isoformat(),
            "step_number": len(self.steps) + 1,
            "type": step_type,
            "content": content[:500],  # Truncate for logging
            "metadata": metadata or {},
        }
        self.steps.append(step)
        logger.info(f"[{step_type.upper()}] {content[:100]}...")

    def get_summary(self) -> dict:
        return {
            "total_steps": len(self.steps),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "steps": self.steps,
        }


# SECTION 3: STATE MANAGEMENT WITH PYDANTIC
# Key point: State flows between nodes, use Annotated for accumulation
from typing import TypedDict


class AgentState(TypedDict):
    """
    State schema for LangGraph.

    Key patterns:
    - Annotated[list, operator.add]: Messages ACCUMULATE across nodes
    - Regular fields: Get REPLACED each time
    """
    # Messages accumulate - crucial for conversation history
    messages: Annotated[list[BaseMessage], operator.add]

    # Tracking fields - replaced each update
    current_step: str
    iteration_count: int

    # Loop detection
    last_actions: list[str]  # Track recent actions to detect loops

    # Results
    final_answer: str | None

    # Observability
    trace: dict | None


# SECTION 4: TOOL DEFINITIONS
# Key point: Tools are functions the agent can call. Use @tool decorator.
@tool
def search_database(query: str) -> str:
    """
    Search the company database for information.
    Use this when you need to find data about products, customers, or sales.
    """
    # Simulated database search
    logger.debug(f"Searching database for: {query}")

    mock_data = {
        "revenue": "Q3 2024 revenue was $4.2M, up 15% YoY",
        "products": "Top products: Widget Pro ($1.2M), Service Plus ($800K)",
        "customers": "Total customers: 1,247. Enterprise: 89, SMB: 1,158",
        "employees": "Current headcount: 156 employees across 3 offices",
    }

    query_lower = query.lower()
    for key, value in mock_data.items():
        if key in query_lower:
            return value

    return f"Found 3 results for '{query}': [Summary data available]"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use this for any calculations like addition, multiplication, percentages.
    Examples: '25 * 4', '1000 * 1.15', '500 / 100'
    """
    logger.debug(f"Calculating: {expression}")
    try:
        # Safe eval - only allow math operations
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_current_date() -> str:
    """
    Get the current date and time.
    Use this when the user asks about today's date or current time.
    """
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def send_notification(recipient: str, message: str) -> str:
    """
    Send a notification to a user.
    Use this when you need to alert or notify someone.
    """
    # In production: integrate with email/Slack/Teams
    logger.info(f"Notification to {recipient}: {message}")

    return f"Notification sent to {recipient}: '{message[:50]}...'"


# All available tools
TOOLS = [search_database, calculate, get_current_date, send_notification]


# SECTION 5: LANGGRAPH NODES
# Key point: Nodes are functions that receive state and return partial updates
def create_llm() -> ChatOpenAI:
    """Create LLM with tool binding."""
    llm = ChatOpenAI(
        model=CONFIG.model_name,
        temperature=CONFIG.temperature,
        api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
    )
    return llm.bind_tools(TOOLS)


# System prompt - versioned!
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Your approach (ReAct pattern):
1. THINK: Analyze the user's request and plan your approach
2. ACT: Use tools when needed to gather information or perform actions
3. OBSERVE: Review tool outputs and determine next steps
4. REPEAT: Continue until you can provide a complete answer

Available tools:
- search_database: Find company data (revenue, products, customers)
- calculate: Perform mathematical calculations
- get_current_date: Get current date/time
- send_notification: Send alerts to users

Guidelines:
- Be concise and direct
- Use tools when you need real data, don't make up numbers
- If you have enough information, provide the final answer
- Always explain your reasoning briefly

Prompt version: {prompt_version}
"""


def agent_node(state: AgentState) -> dict:
    """
    Agent reasoning node - the 'brain' of the ReAct loop.
    Receives state, calls LLM, returns message update.
    """
    logger.info(f"Agent node - iteration {state.get('iteration_count', 0) + 1}")

    # Prepare messages with system prompt
    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(prompt_version=CONFIG.prompt_version))
    messages = [system_msg] + state["messages"]

    # Call LLM
    llm = create_llm()
    response = llm.invoke(messages)

    logger.debug(f"Agent response: {response.content[:100] if response.content else 'Tool call'}...")

    # Return partial state update - messages will be APPENDED due to operator.add
    return {
        "messages": [response],
        "current_step": "reasoning",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tool_node(state: AgentState) -> dict:
    """
    Tool execution node.
    Uses LangGraph's ToolNode for automatic tool invocation.
    """
    logger.info("Executing tools...")

    # Get last message (should have tool calls)
    last_message = state["messages"][-1]

    # Track actions for loop detection
    action_names = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        action_names = [tc["name"] for tc in last_message.tool_calls]

    # Execute tools using LangGraph's ToolNode
    tool_executor = ToolNode(TOOLS)
    result = tool_executor.invoke(state)

    # Update last_actions for loop detection
    current_actions = state.get("last_actions", [])[-5:]  # Keep last 5
    current_actions.extend(action_names)

    return {
        "messages": result["messages"],
        "current_step": "tool_execution",
        "last_actions": current_actions[-6:],  # Keep last 6 for comparison
    }


# SECTION 6: LOOP PREVENTION
# Key point: Multiple strategies - max iterations, recursion limit, detection
def detect_loop(state: AgentState) -> bool:
    """
    Detect if agent is stuck in a loop.
    Strategy: Check if same action repeated 3+ times consecutively.
    """
    actions = state.get("last_actions", [])
    if len(actions) >= 3:
        # Check last 3 actions
        last_three = actions[-3:]
        if len(set(last_three)) == 1:
            logger.warning(f"Loop detected! Action '{last_three[0]}' repeated 3 times")
            return True
    return False


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Routing function - decides next step.
    This is where we implement loop prevention!
    """
    # Strategy 1: Max iterations check
    if state.get("iteration_count", 0) >= CONFIG.max_iterations:
        logger.warning(f"Max iterations ({CONFIG.max_iterations}) reached - forcing end")
        return "end"

    # Strategy 2: Loop detection
    if detect_loop(state):
        logger.warning("Loop detected - forcing end")
        return "end"

    # Check if last message has tool calls
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.debug("Has tool calls - routing to tools")
        return "tools"

    # No tool calls = agent is done
    logger.debug("No tool calls - ending")
    return "end"


# SECTION 7: BUILD THE GRAPH
# Key point: StateGraph with nodes, edges, and conditional routing
def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph agent.

    Graph structure:
    [START] -> [agent] -> {tools} -> {END or REPEAT}
    """
    # 1. Create graph with state schema
    workflow = StateGraph(AgentState)

    # 2. Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 3. Set entry point
    workflow.set_entry_point("agent")

    # 4. Add conditional edges (the ReAct loop!)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # 5. After tools, always go back to agent (the loop)
    workflow.add_edge("tools", "agent")

    return workflow


# Compile the graph
agent_graph = build_agent_graph()
agent_app = agent_graph.compile()


# SECTION 8: RUNTIME WITH TIMEOUT & ERROR HANDLING
# Key point: Wrap execution with timeout, handle errors gracefully
async def run_agent_async(
        query: str,
        timeout: int | None = None,
        session_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the agent with timeout protection.

    Args:
        query: User's question
        timeout: Max execution time in seconds
        session_id: For conversation tracking (checkpointing)

    Returns:
        Agent response with metadata
    """
    timeout = timeout or CONFIG.timeout_seconds
    trace = AgentTrace()

    trace.log_step("input", query)

    # Prepare initial state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "current_step": "start",
        "iteration_count": 0,
        "last_actions": [],
        "final_answer": None,
        "trace": None,
    }

    try:
        # Run with timeout (Strategy 4 for loop prevention)
        result = await asyncio.wait_for(
            agent_app.ainvoke(
                initial_state,
                {"recursion_limit": CONFIG.recursion_limit}  # Strategy 2
            ),
            timeout=timeout
        )

        # Extract final answer
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else str(final_message)

        trace.log_step("output", answer)

        return {
            "success": True,
            "answer": answer,
            "iterations": result.get("iteration_count", 0),
            "trace": trace.get_summary(),
            "config_version": CONFIG.version,
        }

    except asyncio.TimeoutError:
        logger.error(f"Agent timed out after {timeout}s")
        trace.log_step("error", f"Timeout after {timeout}s")
        return {
            "success": False,
            "error": f"Agent timed out after {timeout} seconds",
            "answer": None,
            "trace": trace.get_summary(),
        }
    except Exception as e:
        logger.exception("Agent execution failed")
        trace.log_step("error", str(e))
        return {
            "success": False,
            "error": str(e),
            "answer": None,
            "trace": trace.get_summary(),
        }


def run_agent(query: str) -> dict[str, Any]:
    """Synchronous wrapper for the async agent."""
    return asyncio.run(run_agent_async(query))


# SECTION 9: FASTAPI DEPLOYMENT
# Key point: Production-ready API with proper request/response models
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def create_api() -> "FastAPI":
    """Create FastAPI application for agent deployment."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="ReAct Agent API",
        description="LangGraph-based ReAct agent with compliance features",
        version=CONFIG.version,
    )

    # CORS for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production!
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request/Response models
    class AgentRequest(BaseModel):
        query: str = Field(..., description="User's question or task")
        session_id: str | None = Field(None, description="Session ID for conversation tracking")
        timeout: int | None = Field(None, description="Custom timeout in seconds")

    class AgentResponse(BaseModel):
        success: bool
        answer: str | None
        error: str | None = None
        iterations: int | None = None
        config_version: str

    @app.post("/agent/invoke", response_model=AgentResponse)
    async def invoke_agent(request: AgentRequest):
        """Invoke the ReAct agent with a query."""
        try:
            result = await run_agent_async(
                query=request.query,
                timeout=request.timeout,
                session_id=request.session_id,
            )
            return AgentResponse(
                success=result["success"],
                answer=result.get("answer"),
                error=result.get("error"),
                iterations=result.get("iterations"),
                config_version=CONFIG.version,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health_check():
        """Health check endpoint for load balancers."""
        return {
            "status": "healthy",
            "version": CONFIG.version,
            "environment": CONFIG.environment.value,
        }

    @app.get("/config")
    def get_config():
        """Get current configuration (non-sensitive fields)."""
        return {
            "version": CONFIG.version,
            "environment": CONFIG.environment.value,
            "prompt_version": CONFIG.prompt_version,
            "max_iterations": CONFIG.max_iterations,
            "model": CONFIG.model_name,
        }

    return app


# SECTION 10: MULTI-AGENT EXAMPLE (Supervisor Pattern)
# Key point: Supervisor routes tasks to specialized workers
class MultiAgentState(TypedDict):
    """State for multi-agent system."""
    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str
    task_type: str
    results: dict


def supervisor_node(state: MultiAgentState) -> dict:
    """
    Supervisor agent - routes to specialists.
    In production: Use LLM to make routing decisions.
    """
    last_msg = state["messages"][-1].content.lower()

    # Simple keyword-based routing (use LLM in production)
    if any(word in last_msg for word in ["calculate", "math", "number"]):
        return {"next_agent": "calculator", "task_type": "calculation"}
    elif any(word in last_msg for word in ["search", "find", "data", "revenue"]):
        return {"next_agent": "researcher", "task_type": "research"}
    else:
        return {"next_agent": "generalist", "task_type": "general"}


def build_multi_agent_graph() -> StateGraph:
    """
    Build multi-agent supervisor graph.

    Structure:
    [supervisor] -> {calculator | researcher | generalist} -> [END]
    """
    workflow = StateGraph(MultiAgentState)

    # Add supervisor
    workflow.add_node("supervisor", supervisor_node)

    # Add worker nodes (simplified - use full agents in production)
    workflow.add_node("calculator", lambda s: {"results": {"calc": "done"}})
    workflow.add_node("researcher", lambda s: {"results": {"research": "done"}})
    workflow.add_node("generalist", lambda s: {"results": {"general": "done"}})

    workflow.set_entry_point("supervisor")

    # Route based on supervisor decision
    workflow.add_conditional_edges(
        "supervisor",
        lambda s: s["next_agent"],
        {
            "calculator": "calculator",
            "researcher": "researcher",
            "generalist": "generalist",
        }
    )

    # All workers end
    workflow.add_edge("calculator", END)
    workflow.add_edge("researcher", END)
    workflow.add_edge("generalist", END)

    return workflow


# MAIN
def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}\n")


def demo():
    """Run demonstration of all features."""
    print_section("ReAct Agent Demo")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("NOTE: OPENAI_API_KEY not set. Using mock responses.\n")
        print("Set the key to see real agent behavior:")
        print("export OPENAI_API_KEY='your-key-here'\n")

    # Show configuration
    print_section("1. Configuration & Versioning")
    print(f"Agent Version: {CONFIG.version}")
    print(f"Prompt Version: {CONFIG.prompt_version}")
    print(f"Model: {CONFIG.model_name}")
    print(f"Environment: {CONFIG.environment.value}")
    print(f"Max Iterations: {CONFIG.max_iterations}")
    print(f"Temperature: {CONFIG.temperature}")

    # Show graph structure
    print_section("2. LangGraph Structure")
    print("Nodes: agent, tools")
    print("Entry: agent")
    print("Flow: agent -> (tools -> agent)* -> END")
    print("Loop prevention: max_iterations, recursion_limit, detection, timeout")

    # Show state schema
    print_section("3. State Management")
    print("AgentState fields:")
    print("- messages: Annotated[list, operator.add]  # Accumulates!")
    print("- iteration_count: int  # Replaced each update")
    print("- last_actions: list[str]  # For loop detection")
    print("- final_answer: str | None")

    # Show tools
    print_section("4. Available Tools")
    for tool in TOOLS:
        print(f"- {tool.name}: {tool.description[:60]}...")

    # Run agent if API key is available
    if os.getenv("OPENAI_API_KEY"):
        print_section("6. Running Agent")

        queries = [
            "What was the Q3 revenue and calculate 15% growth on it?",
            "What is today's date?",
            "Search for customer count and calculate average revenue per customer if revenue is 4.2M",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            print("-" * 40)
            result = run_agent(query)
            if result["success"]:
                print(f"Answer: {result['answer']}")
                print(f"Iterations: {result['iterations']}")
            else:
                print(f"Error: {result['error']}")
    else:
        print_section("5. Example Queries (requires API key)")
        print("Example queries the agent can handle:")
        print("- 'What was the Q3 revenue and calculate 15% growth?'")
        print("- 'What is today's date?'")
        print("- 'Search for customer count and calculate average revenue'")

    # FastAPI info
    print_section("6. FastAPI Deployment")
    if FASTAPI_AVAILABLE:
        print("FastAPI is available!")
        print("To run the API server:")
        print("uvicorn react_agent_demo:api --reload")
        print("\nEndpoints:")
        print("POST /agent/invoke - Run the agent")
        print("GET /health - Health check")
        print("GET /config - Current configuration")
    else:
        print("Install FastAPI: pip install fastapi uvicorn")

    print_section("Demo Complete!")
    print("This demo demonstrates:")
    print("- ReAct pattern (Thought -> Action -> Observation)")
    print("- LangChain tools with @tool decorator")
    print("- LangGraph state machines with TypedDict")
    print("- State passing with Annotated[list, operator.add]")
    print("- Loop prevention (4 strategies)")
    print("- Multi-agent supervisor pattern")
    print("- Configuration versioning")
    print("- FastAPI deployment ready")


# Create API instance for uvicorn
if FASTAPI_AVAILABLE:
    api = create_api()

if __name__ == "__main__":
    demo()
