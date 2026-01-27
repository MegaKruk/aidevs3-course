"""
ReAct Agent Showcase - Python AI Engineer Interview Demo
=========================================================
Demonstrates: LangChain, LangGraph, ReAct pattern, state management,
loop prevention, multi-agent design, versioning, compliance, FastAPI deployment,
and LightRAG graph-enhanced retrieval.

Author: MegaKruk
Python: 3.12

Setup:
    1. Create .env file with: OPENAI_API_KEY=your-key-here
    2. pip install python-dotenv langchain langchain-openai langgraph pydantic fastapi uvicorn lightrag-hku pypdf
    3. Place documents in ./documents/ folder for indexing
    4. python react_agent_demo.py
"""

import asyncio
import json
import logging
import operator
import os
import re
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from pydantic import BaseModel, Field
from pypdf import PdfReader

# Load environment variables from .env file
load_dotenv()


# CONFIGURATION AND VERSIONING
class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AgentConfig(BaseModel):
    """Versioned configuration for the agent."""
    version: str = "2.0.0"  # Updated for LightRAG integration
    model_name: str = "gpt-4o"
    indexing_model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_iterations: int = 10
    recursion_limit: int = 25
    timeout_seconds: int = 120  # Increased for LightRAG operations
    environment: Environment = Environment.DEVELOPMENT
    prompt_version: str = "v2.0.0"

    # Feature flags
    enable_lightrag: bool = True  # Toggle LightRAG features
    verbose_logging: bool = True

    # LightRAG settings
    lightrag_working_dir: str = "./lightrag_data"
    documents_dir: str = "./documents"
    chunk_token_size: int = 500
    chunk_overlap_token_size: int = 100


def load_config(env: Environment = Environment.DEVELOPMENT) -> AgentConfig:
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
            verbose_logging=False,
            environment=Environment.PRODUCTION,
        ),
    }
    return configs.get(env, configs[Environment.DEVELOPMENT])


CONFIG = load_config()


# LOGGING AND OBSERVABILITY
logging.basicConfig(
    level=logging.DEBUG if CONFIG.verbose_logging else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class AgentTrace:
    """Simple tracing for observability."""

    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.start_time = datetime.now()
        self.tokens_used = 0

    def log_step(self, step_type: str, content: str, metadata: dict | None = None):
        step = {
            "timestamp": datetime.now().isoformat(),
            "step_number": len(self.steps) + 1,
            "type": step_type,
            "content": content[:500],
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


# LIGHTRAG INTEGRATION
def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text content from a PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


class LightRAGManager:
    """
    Manager for LightRAG knowledge base operations.

    LightRAG provides graph-enhanced RAG with:
    - Knowledge graph extraction from documents
    - Dual-level retrieval (local entities + global relationships)
    - Incremental document updates
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.rag: LightRAG | None = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize LightRAG instance."""
        if self._initialized:
            return True

        try:
            # Create working directory
            Path(self.config.lightrag_working_dir).mkdir(parents=True, exist_ok=True)

            # Initialize LightRAG with OpenAI
            self.rag = LightRAG(
                working_dir=self.config.lightrag_working_dir,
                llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs:
                openai_complete_if_cache(
                    # Use gpt-4o-mini for indexing tasks to save tokens/limit
                    self.config.indexing_model_name if "keyword" in str(prompt).lower() or "entities" in str(
                        prompt).lower() else self.config.model_name,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    **kwargs
                ),
                embedding_func=EmbeddingFunc(
                    embedding_dim=1536,
                    func=lambda texts: openai_embed(
                        texts,
                        model="text-embedding-3-small",
                        api_key=os.getenv("OPENAI_API_KEY"),
                    )
                ),
                chunk_token_size=self.config.chunk_token_size,
                chunk_overlap_token_size=self.config.chunk_overlap_token_size,
                llm_model_max_async=1,  # Process only 1 chunk at a time
                embedding_func_max_async=1,  # Process only 1 embedding batch at a time
            )

            # Critical: Initialize storage backends
            await self.rag.initialize_storages()

            self._initialized = True
            logger.info("LightRAG initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            return False

    async def index_document(self, content: str, doc_id: str | None = None) -> dict:
        """Index a document into the knowledge graph."""
        if not self._initialized:
            await self.initialize()

        try:
            if doc_id:
                await self.rag.ainsert(content, ids=[doc_id])
            else:
                await self.rag.ainsert(content)

            return {"success": True, "message": "Document indexed successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def index_documents_from_folder(self, folder_path: str | None = None) -> dict:
        """Index all supported files from a folder."""
        folder = Path(folder_path or self.config.documents_dir)

        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            return {"success": False, "message": f"Created empty folder: {folder}. Add documents and retry."}

        results = {"indexed": [], "failed": [], "skipped": []}

        # Support multiple file types including PDF
        patterns = ["*.txt", "*.md", "*.json", "*.pdf"]
        files = []
        for pattern in patterns:
            files.extend(folder.glob(pattern))

        if not files:
            return {"success": False, "message": f"No documents found in {folder}"}

        for file_path in files:
            try:
                # Handle PDF files separately
                if file_path.suffix.lower() == ".pdf":
                    content = extract_text_from_pdf(file_path)
                else:
                    content = file_path.read_text(encoding="utf-8")

                if len(content.strip()) < 100:
                    results["skipped"].append(str(file_path))
                    continue

                result = await self.index_document(content, doc_id=file_path.stem)
                if result["success"]:
                    results["indexed"].append(str(file_path))
                else:
                    results["failed"].append({"file": str(file_path), "error": result.get("error")})
            except Exception as e:
                results["failed"].append({"file": str(file_path), "error": str(e)})

        return {
            "success": len(results["indexed"]) > 0,
            "total_files": len(files),
            "indexed": len(results["indexed"]),
            "failed": len(results["failed"]),
            "skipped": len(results["skipped"]),
            "details": results
        }

    async def query(
            self,
            question: str,
            mode: Literal["naive", "local", "global", "hybrid", "mix"] = "hybrid"
    ) -> dict:
        """Query the knowledge base using different retrieval modes."""
        if not self._initialized:
            await self.initialize()

        try:
            param = QueryParam(
                mode=mode,
                top_k=60,
                max_total_tokens=30000,
            )

            result = await self.rag.aquery(question, param=param)

            return {
                "success": True,
                "answer": result,
                "mode": mode,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "mode": mode}

    async def get_context_only(self, question: str, mode: str = "hybrid") -> dict:
        """Get only the retrieved context without generating an answer."""
        if not self._initialized:
            await self.initialize()

        try:
            param = QueryParam(
                mode=mode,
                only_need_context=True,
            )

            context = await self.rag.aquery(question, param=param)
            return {"success": True, "context": context, "mode": mode}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """Cleanup LightRAG resources."""
        if self.rag and self._initialized:
            try:
                await self.rag.finalize_storages()
                logger.info("LightRAG resources cleaned up")
            except Exception as e:
                logger.error(f"Error during LightRAG cleanup: {e}")


# Global LightRAG manager instance
lightrag_manager: LightRAGManager | None = None


async def get_lightrag_manager() -> LightRAGManager:
    """Get or create LightRAG manager instance."""
    global lightrag_manager
    if lightrag_manager is None:
        lightrag_manager = LightRAGManager(CONFIG)
        await lightrag_manager.initialize()
    return lightrag_manager


# STATE MANAGEMENT WITH PYDANTIC
class AgentState(TypedDict):
    """State schema for LangGraph."""
    messages: Annotated[list[BaseMessage], operator.add]
    current_step: str
    iteration_count: int
    last_actions: list[str]
    final_answer: str | None
    trace: dict | None


# TOOL DEFINITIONS (sync tools only - async tools handled separately)
@tool
def search_database(query: str) -> str:
    """
    Search the company database for information.
    Use this when you need to find data about products, customers, or sales.
    """
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
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        result = round(result, 10)
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


# Tool metadata for LLM (descriptions for tool binding)
TOOL_DESCRIPTIONS = {
    "search_knowledge_base": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base using graph-enhanced retrieval (LightRAG).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "mode": {
                    "type": "string",
                    "default": "hybrid",
                    "enum": ["naive", "local", "global", "hybrid", "mix"],
                    "description": "Search mode: naive (basic), local (entity-focused), global (relationship-focused), hybrid (balanced), mix (KG+vector)"
                }
            },
            "required": ["query"]
        }
    },
    "index_document": {
        "name": "index_document",
        "description": "Index a new document into the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The document content to index"},
                "doc_id": {"type": "string", "default": "", "description": "Optional document identifier"}
            },
            "required": ["content"]
        }
    },
    "index_folder": {
        "name": "index_folder",
        "description": "Index all documents within a specific folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "default": "", "description": "Path to the folder containing documents"}
            },
            "required": []
        }
    }
}

# Sync tools list
SYNC_TOOLS = [search_database, calculate, get_current_date]


# LANGGRAPH NODES
def create_llm() -> ChatOpenAI:
    """Create LLM with tool binding."""
    llm = ChatOpenAI(
        model=CONFIG.model_name,
        temperature=CONFIG.temperature,
        api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
    )

    # Bind sync tools normally
    tools_for_binding = SYNC_TOOLS.copy()

    # Add async tool definitions manually as function schemas
    async_tool_schemas = [
        {"type": "function", "function": TOOL_DESCRIPTIONS["search_knowledge_base"]},
        {"type": "function", "function": TOOL_DESCRIPTIONS["index_document"]},
        {"type": "function", "function": TOOL_DESCRIPTIONS["index_folder"]},
    ]

    # Bind sync tools first, then add async schemas
    llm_with_tools = llm.bind_tools(tools_for_binding)

    # Override with all tools including async ones
    return llm.bind(tools=[
        *[{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.args_schema.schema() if hasattr(t, 'args_schema') and t.args_schema else {"type": "object", "properties": {}}}} for t in SYNC_TOOLS],
        *async_tool_schemas
    ])


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
- search_knowledge_base: Search indexed documents using graph-enhanced RAG (LightRAG)
  * Use mode='local' for specific entity queries ("Who is X?", "What is Y?")
  * Use mode='global' for relationship queries ("How does X relate to Y?")
  * Use mode='hybrid' for general queries (default, balanced approach)
  * Use mode='mix' for complex multi-hop queries
- index_document: Add new documents to the knowledge base

Guidelines:
- Be concise and direct
- Use tools when you need real data
- For document/knowledge queries, prefer search_knowledge_base over search_database
- Always explain your reasoning briefly

Prompt version: {prompt_version}
"""


async def agent_node(state: AgentState) -> dict:
    """Agent reasoning node."""
    logger.info(f"Agent node - iteration {state.get('iteration_count', 0) + 1}")

    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(prompt_version=CONFIG.prompt_version))
    messages = [system_msg] + state["messages"]

    llm = create_llm()
    response = await llm.ainvoke(messages)

    logger.debug(f"Agent response: {response.content[:100] if response.content else 'Tool call'}...")

    return {
        "messages": [response],
        "current_step": "reasoning",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def tool_node(state: AgentState) -> dict:
    """
    Tool execution node - handles both sync and async tools.

    This is the key fix: we execute tools directly in the async context
    instead of using ToolNode which runs sync tools in a thread pool.
    """
    logger.info("Executing tools...")

    last_message = state["messages"][-1]
    action_names = []
    tool_messages = []

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            action_names.append(tool_name)

            logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")

            try:
                # Handle async LightRAG tools
                if tool_name == "search_knowledge_base":
                    query = tool_args.get("query", "")
                    mode = tool_args.get("mode", "hybrid")
                    logger.debug(f"LightRAG search: {query} (mode={mode})")
                    manager = await get_lightrag_manager()
                    result = await manager.query(query, mode=mode)
                    if result["success"]:
                        output = f"[Knowledge Base - {mode} mode]\n{result['answer']}"
                    else:
                        output = f"Search failed: {result.get('error', 'Unknown error')}"

                elif tool_name == "index_document":
                    content = tool_args.get("content", "")
                    doc_id = tool_args.get("doc_id", "")
                    manager = await get_lightrag_manager()
                    result = await manager.index_document(content, doc_id or None)
                    if result["success"]:
                        output = f"Document '{doc_id or 'unnamed'}' indexed successfully."
                    else:
                        output = f"Indexing failed: {result.get('error', 'Unknown error')}"

                elif tool_name == "index_folder":
                    folder_path = tool_args.get("folder_path", "")
                    manager = await get_lightrag_manager()
                    result = await manager.index_documents_from_folder(folder_path or None)
                    if result["success"]:
                        output = f"Success! Indexed: {result['indexed']}, Failed: {result['failed']}"
                    else:
                        output = f"Failed: {result.get('message')}"

                # Handle sync tools
                elif tool_name == "search_database":
                    output = search_database.invoke(tool_args)

                elif tool_name == "calculate":
                    output = calculate.invoke(tool_args)

                elif tool_name == "get_current_date":
                    output = get_current_date.invoke({})

                else:
                    output = f"Unknown tool: {tool_name}"

            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                output = f"Error executing {tool_name}: {str(e)}"

            # Create tool message with result
            tool_messages.append(ToolMessage(content=output, tool_call_id=tool_id))

    current_actions = state.get("last_actions", [])[-5:]
    current_actions.extend(action_names)

    return {
        "messages": tool_messages,
        "current_step": "tool_execution",
        "last_actions": current_actions[-6:],
    }


# LOOP PREVENTION
def detect_loop(state: AgentState) -> bool:
    """Detect if agent is stuck in a loop."""
    actions = state.get("last_actions", [])
    if len(actions) >= 3:
        last_three = actions[-3:]
        if len(set(last_three)) == 1:
            logger.warning(f"Loop detected! Action '{last_three[0]}' repeated 3 times")
            return True
    return False


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Routing function with loop prevention."""
    if state.get("iteration_count", 0) >= CONFIG.max_iterations:
        logger.warning(f"Max iterations ({CONFIG.max_iterations}) reached")
        return "end"

    if detect_loop(state):
        return "end"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# BUILD THE GRAPH
def build_agent_graph() -> StateGraph:
    """Build the LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )

    workflow.add_edge("tools", "agent")

    return workflow


agent_graph = build_agent_graph()
agent_app = agent_graph.compile()


# RUNTIME
async def run_agent_async(
        query: str,
        timeout: int | None = None,
        session_id: str | None = None,
) -> dict[str, Any]:
    """Run the agent with timeout protection."""
    timeout = timeout or CONFIG.timeout_seconds
    trace = AgentTrace()

    trace.log_step("input", query)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "current_step": "start",
        "iteration_count": 0,
        "last_actions": [],
        "final_answer": None,
        "trace": None,
    }

    try:
        result = await asyncio.wait_for(
            agent_app.ainvoke(
                initial_state,
                {"recursion_limit": CONFIG.recursion_limit}
            ),
            timeout=timeout
        )

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
        return {"success": False, "error": f"Timeout after {timeout}s", "answer": None}
    except Exception as e:
        logger.exception("Agent execution failed")
        return {"success": False, "error": str(e), "answer": None}


def run_agent(query: str) -> dict[str, Any]:
    """Synchronous wrapper."""
    return asyncio.run(run_agent_async(query))


# FASTAPI DEPLOYMENT
def create_api() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="ReAct Agent API with LightRAG",
        description="LangGraph-based ReAct agent with graph-enhanced RAG",
        version=CONFIG.version,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class AgentRequest(BaseModel):
        query: str = Field(..., description="User's question or task")
        session_id: str | None = Field(None, description="Session ID")
        timeout: int | None = Field(None, description="Timeout in seconds")

    class AgentResponse(BaseModel):
        success: bool
        answer: str | None
        error: str | None = None
        iterations: int | None = None
        config_version: str

    class IndexRequest(BaseModel):
        content: str = Field(..., description="Document content to index")
        doc_id: str | None = Field(None, description="Document identifier")

    class IndexFolderRequest(BaseModel):
        folder_path: str | None = Field(None, description="Folder path to index")

    class KnowledgeQueryRequest(BaseModel):
        query: str = Field(..., description="Search query")
        mode: str = Field("hybrid", description="Search mode: naive, local, global, hybrid, mix")

    @app.post("/agent/invoke", response_model=AgentResponse)
    async def invoke_agent(request: AgentRequest):
        """Invoke the ReAct agent."""
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

    @app.post("/knowledge/index")
    async def index_document_api(request: IndexRequest):
        """Index a document into the knowledge base."""
        manager = await get_lightrag_manager()
        result = await manager.index_document(request.content, request.doc_id)
        return result

    @app.post("/knowledge/index-folder")
    async def index_folder_api(request: IndexFolderRequest):
        """Index all documents from a folder."""
        manager = await get_lightrag_manager()
        result = await manager.index_documents_from_folder(request.folder_path)
        return result

    @app.post("/knowledge/query")
    async def query_knowledge_api(request: KnowledgeQueryRequest):
        """Query the knowledge base directly."""
        manager = await get_lightrag_manager()
        result = await manager.query(request.query, mode=request.mode)
        return result

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "version": CONFIG.version,
            "lightrag_enabled": CONFIG.enable_lightrag,
        }

    @app.get("/config")
    def get_config():
        return {
            "version": CONFIG.version,
            "environment": CONFIG.environment.value,
            "lightrag_enabled": CONFIG.enable_lightrag,
            "model": CONFIG.model_name,
        }

    return app


# MAIN
def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}\n")


async def demo_async():
    """Run demonstration of all features including LightRAG."""
    print_section("ReAct Agent + LightRAG Showcase Demo")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("NOTE: OPENAI_API_KEY not set.")
        print("Set the key in .env file to see real behavior.\n")
        return

    # Show configuration
    print_section("1. Configuration")
    print(f"Agent Version: {CONFIG.version}")
    print(f"Model: {CONFIG.model_name}")
    print(f"LightRAG Enabled: {CONFIG.enable_lightrag}")

    # Show tools
    print_section("2. Available Tools")
    for t in SYNC_TOOLS:
        print(f"- {t.name}: {t.description}...")
    for name in TOOL_DESCRIPTIONS:
        print(f"- {name}: {TOOL_DESCRIPTIONS[name]['description']}...")

    # LightRAG demo
    if CONFIG.enable_lightrag:
        print_section("3. LightRAG Knowledge Base")

        manager = await get_lightrag_manager()

        # Check for documents to index
        docs_dir = Path(CONFIG.documents_dir)
        if docs_dir.exists() and list(docs_dir.glob("*")):
            print(f"Found documents in {docs_dir}, indexing...")
            result = await manager.index_documents_from_folder()
            print(f"Indexing result: {result['indexed']} documents indexed")
        else:
            print(f"No documents in {docs_dir}")

        # Demo queries
        print_section("4. LightRAG Query Demos")

        demo_queries = [
            ("What are Filip's technical skills?", "local"),
            ("How does Filip's experience relate to AI projects?", "global"),
            ("Tell me about Filip's favorite dish", "hybrid"),
        ]

        for query, mode in demo_queries:
            print(f"\nQuery: {query}")
            print(f"Mode: {mode}")
            print("-" * 40)
            result = await manager.query(query, mode=mode)
            if result["success"]:
                # Truncate long answers for demo
                answer = result["answer"]
                if len(answer) > 300:
                    answer = answer[:300] + "..."
                print(f"Answer: {answer}")
            else:
                print(f"Error: {result.get('error')}")

    # Run agent demo
    print_section("5. Agent Demo")

    queries = [
        "What skills does Filip have related to AI?",
        "Calculate 4.2 million times 1.15 for revenue growth",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = await run_agent_async(query)
        if result["success"]:
            answer = result["answer"]
            if len(answer) > 300:
                answer = answer[:300] + "..."
            print(f"Answer: {answer}")
            print(f"Iterations: {result['iterations']}")
        else:
            print(f"Error: {result['error']}")

    # Cleanup
    if lightrag_manager:
        await lightrag_manager.cleanup()

    print_section("Demo Complete!")
    print("Features demonstrated:")
    print("- LightRAG graph-enhanced retrieval")
    print("- Dual-level search (local/global/hybrid)")
    print("- Document indexing with entity extraction")
    print("- ReAct agent with knowledge base tool")
    print("- FastAPI endpoints for all operations")


def demo():
    """Sync wrapper for demo."""
    asyncio.run(demo_async())


# Create API instance
api = create_api()

if __name__ == "__main__":
    demo()
