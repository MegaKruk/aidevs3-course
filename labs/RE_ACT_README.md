# ReAct Agent with LightRAG

A Python AI agent demo showcasing LangChain, LangGraph, and LightRAG for graph-enhanced retrieval-augmented generation.

## What This Does

- **ReAct Agent**: Reasoning + Acting pattern using LangGraph state machines
- **LightRAG Integration**: Graph-enhanced RAG that extracts entities and relationships from documents
- **Multiple Search Modes**: Local (entity-focused), global (relationship-focused), hybrid, and mix
- **FastAPI Server**: REST API for agent invocation and knowledge base operations
- **Compliance Features**: PII detection, content safety, audit logging

## Quick Start

### 1. Install Dependencies

```bash
pip install python-dotenv langchain langchain-openai langgraph pydantic fastapi uvicorn lightrag-hku pypdf
```

### 2. Configure Environment

Create `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

### 3. Add Documents (Optional)

Place files in `./documents/` folder. Supported formats: `.txt`, `.md`, `.json`, `.pdf`

Example documents to add:
- Your CV/resume
- Technical documentation
- Product brochures
- Book chapters

### 4. Run the Demo

```bash
python react_agent_showcase.py
```

### 5. Run as API Server

```bash
uvicorn react_agent_showcase:api --reload --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent/invoke` | POST | Run the ReAct agent with a query |
| `/knowledge/index` | POST | Index a single document |
| `/knowledge/index-folder` | POST | Index all documents from folder |
| `/knowledge/query` | POST | Query knowledge base directly |
| `/health` | GET | Health check |
| `/config` | GET | Current configuration |

### Example Requests

**Invoke Agent:**
```json
POST /agent/invoke
{
    "query": "What skills does Filip have related to AI?",
    "timeout": 60
}
```

**Index Document:**
```json
POST /knowledge/index
{
    "content": "Your document text here...",
    "doc_id": "my-document"
}
```

**Query Knowledge Base:**
```json
POST /knowledge/query
{
    "query": "How does X relate to Y?",
    "mode": "hybrid"
}
```

## LightRAG Search Modes

| Mode | Use Case | Description |
|------|----------|-------------|
| `naive` | Simple facts | Basic vector search like traditional RAG |
| `local` | Entity queries | "Who is X?", "What is Y?" - focuses on specific entities |
| `global` | Relationship queries | "How does X relate to Y?" - captures themes and connections |
| `hybrid` | General purpose | Combines local and global (recommended default) |
| `mix` | Complex queries | Knowledge graph + vector retrieval combined |

## Project Structure

```
.
|-- react_agent_showcase.py   # Main application
|-- .env                      # Environment variables (create this)
|-- documents/                # Place documents here for indexing
|-- lightrag_data/            # LightRAG storage (auto-created)
```

## Available Tools

The agent has access to these tools:

1. **search_database** - Mock company database (revenue, products, customers)
2. **calculate** - Mathematical calculations
3. **get_current_date** - Current date/time
4. **search_knowledge_base** - LightRAG graph-enhanced search
5. **index_document** - Add documents to knowledge base

## How LightRAG Works

1. **Document Ingestion**: Text is chunked into segments
2. **Entity Extraction**: LLM identifies entities (people, places, concepts)
3. **Relationship Extraction**: LLM finds connections between entities
4. **Knowledge Graph**: Entities and relationships form a graph
5. **Dual-Level Retrieval**: Queries search both entities (local) and relationships (global)

This provides better context understanding than flat vector search, especially for:
- Multi-hop queries ("What projects did X work on that involved Y?")
- Relationship discovery ("How are these concepts connected?")
- Complex document collections

## Configuration

Key settings in `AgentConfig`:

```python
model_name: str = "gpt-4o-mini"      # LLM model
temperature: float = 0.0             # 0 for deterministic
max_iterations: int = 10             # Loop prevention
timeout_seconds: int = 120           # Request timeout
enable_lightrag: bool = True         # Toggle LightRAG
chunk_token_size: int = 1200         # Document chunking
```

## Requirements

- Python 3.10+
- OpenAI API key
- Packages: langchain, langgraph, lightrag-hku, fastapi, pydantic, pypdf

## License

MIT
