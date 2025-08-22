## Available Models

### OpenAI Models
- **GPT-5**: Latest generation models (if available in your account)
  - `gpt-5`
  - `gpt-5-turbo`
- **GPT-4o**: Optimized GPT-4 variants
  - `gpt-4o`
  - `gpt-4o-mini`
- **GPT-4 Turbo**: Best overall performance for both analysis and coding
  - `gpt-4-turbo-preview`
  - `gpt-4-1106-preview`
- **GPT-4**: Excellent for complex reasoning
  - `gpt-4`
  - `gpt-4-32k`
- **GPT-3.5 Turbo**: Faster and more cost-effective
  - `gpt-3.5-turbo`
  - `gpt-3.5-turbo-16k`

### Ollama Models
- **Qwen 2.5**: Recommended for local execution
  - `qwen2.5:14b-instruct` (decision-making)
  - `qwen2.5-coder:14b` (code generation)
- **Llama 3**: Alternative option
  - `llama3.2:latest`
- **CodeLlama**: Specialized for coding
  - `codellama:13b`

## Cost Considerations

### OpenAI
- GPT-5: Pricing varies (check OpenAI platform)
- GPT-4o: ~$0.005 per 1K input tokens, ~$0.015 per 1K output tokens
- GPT-4o-mini: ~$0.002 per 1K input tokens, ~$0.008 per 1K output tokens
- GPT-4 Turbo: ~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens
- GPT-3.5 Turbo: ~$0.0005 per 1K input tokens, ~$0.0015 per 1K output tokens
- Typical homework solving: $0.10 - $0.50 per lesson

### Ollama
- Free (runs locally)
- Requires ~16GB RAM for 14B models# Meta Agent - Autonomous LLM-Powered Homework Solver

## Overview

Meta Agent is an autonomous AI system that reads course lessons, extracts homework tasks, and attempts to solve them using LLM-generated Python code. It features safe code execution, automatic error recovery, and comprehensive logging.

## Key Features

Meta Agent supports multiple LLM providers and uses a dual-model approach for optimal performance:

### Supported Providers:
- **OpenAI** (Recommended): GPT-4 models for superior accuracy
- **Ollama**: Local models for privacy and cost-effectiveness

### Dual-Model Architecture:
- **Decision Model**: Used for analysis, summarization, strategy planning, and decision-making
  - OpenAI: `gpt-4-turbo-preview` (default)
  - Ollama: `qwen2.5:14b-instruct` (default)
- **Coding Model**: Specialized model for generating Python code solutions
  - OpenAI: `gpt-4-turbo-preview` (default)
  - Ollama: `qwen2.5-coder:14b` (default)

This separation ensures that each task is handled by the most appropriate model, improving both accuracy and efficiency.

## Installation

### Prerequisites

- Python 3.10 or higher
- Ollama installed and running locally
- (Optional) Docker for enhanced code isolation

### Setup

1. Clone the repository:
```bash
cd /home/megakruk/workspace/python/aidevs3-course/meta-agent-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key (optional)
nano .env
```

4. Set up your LLM provider:

#### Option A: OpenAI (Recommended)
```bash
# Add your API key to .env file:
OPENAI_API_KEY=sk-your-key-here
```

#### Option B: Ollama (Local)
```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull qwen2.5:14b-instruct    # For decision-making
ollama pull qwen2.5-coder:14b        # For code generation
```

## Project Structure

```
meta-agent-project/
├── meta_agent_main.py          # Main entry point
├── core/
│   ├── __init__.py
│   ├── agent.py                # Main agent logic
│   ├── llm_client.py           # Ollama client
│   ├── context_manager.py      # Context management
│   └── strategy.py             # Execution planning
├── processors/
│   ├── __init__.py
│   ├── lesson_processor.py     # Lesson summarization
│   ├── homework_extractor.py   # Task extraction
│   └── code_generator.py       # Code generation
├── executors/
│   ├── __init__.py
│   ├── safe_executor.py        # Safe code execution
│   └── result_analyzer.py      # Result analysis
├── tools/
│   ├── __init__.py
│   └── web_fetcher.py          # Web resource fetching
├── utils/
│   ├── __init__.py
│   └── logger.py               # Logging utilities
├── lessons/                     # Lesson files directory
├── output/                      # Generated outputs
└── requirements.txt
```

## Usage

### Basic Usage

```bash
python meta_agent_main.py lessons/S01E02-custom-data-1730734993.md
```

## Usage

### Basic Usage

```bash
# Auto-detect provider (uses OpenAI if API key is set, otherwise Ollama)
python meta_agent_main.py lessons/S01E02-custom-data-1730734993.md

# Force OpenAI provider
python meta_agent_main.py lessons/lesson.md --provider openai

# Force Ollama provider
python meta_agent_main.py lessons/lesson.md --provider ollama
```

### With Options

```bash
# Using OpenAI with custom models
python meta_agent_main.py lessons/lesson.md \
    --provider openai \
    --openai-decision-model gpt-4o \
    --openai-coding-model gpt-4o \
    --max-attempts 5 \
    --output-dir ./output \
    --verbose

# Using GPT-4o-mini for cost efficiency
python meta_agent_main.py lessons/lesson.md \
    --provider openai \
    --openai-decision-model gpt-4o-mini \
    --openai-coding-model gpt-4o-mini \
    --max-attempts 5

# Using GPT-5 (if available in your account)
python meta_agent_main.py lessons/lesson.md \
    --provider openai \
    --openai-decision-model gpt-5-turbo \
    --openai-coding-model gpt-5-turbo \
    --max-attempts 3

# Using Ollama with custom models
python meta_agent_main.py lessons/lesson.md \
    --provider ollama \
    --decision-model qwen2.5:14b-instruct \
    --coding-model qwen2.5-coder:14b \
    --max-attempts 5 \
    --output-dir ./output \
    --verbose
```

### Command Line Arguments

- `lesson_file`: Path to the lesson file (required)
- `--provider`: LLM provider to use: "auto", "ollama", "openai" (default: auto)
- `--decision-model`: Ollama model for analysis and decisions (default: qwen2.5:14b-instruct)
- `--coding-model`: Ollama model for code generation (default: qwen2.5-coder:14b)
- `--openai-decision-model`: OpenAI model for decisions (default: gpt-4-turbo-preview)
- `--openai-coding-model`: OpenAI model for coding (default: gpt-4-turbo-preview)
- `--max-attempts`: Maximum solution attempts (default: 5)
- `--output-dir`: Output directory (default: ./output)
- `--verbose`: Enable verbose logging

## How It Works

1. **Lesson Reading**: The agent reads the provided lesson file
2. **Content Processing**: Summarizes content and extracts key information
3. **Homework Extraction**: Identifies and extracts the homework task
4. **Resource Fetching**: Downloads any referenced external resources
5. **Code Generation**: Uses LLM to generate Python solution code
6. **Safe Execution**: Runs code in isolated environment
7. **Result Analysis**: Checks for success (FLG:XXXXX format)
8. **Error Recovery**: If failed, analyzes errors and regenerates code
9. **Report Generation**: Creates comprehensive execution report

## Output Files

The agent generates several output files:

### In Output Directory (`./output/`)
- `summary_[lesson_name].md`: Lesson summary and key takeaways
- `homework_[lesson_name].md`: Extracted homework task
- `solution_attempt_[n].py`: Generated Python code for each attempt
- `report_[lesson_name].md`: Final execution report

### In Logs Directory (`./logs/`)
- `meta_agent_[timestamp].log`: Complete execution log with:
  - All generated code for each attempt
  - Full execution output and errors
  - Detailed execution metadata
  - Step-by-step progress tracking
  - Success/failure status for each attempt
  - Found flags (if any)

## Logging Features

The Meta Agent provides comprehensive logging:

### Console Output
- Real-time progress updates
- Key milestones and results
- Error summaries

### File Logging
- **Complete Execution Trace**: Every action is logged with timestamps
- **Code Archive**: All generated code attempts are preserved
- **Output Capture**: Full stdout/stderr from code execution
- **Error Analysis**: Detailed error messages and stack traces
- **Performance Metrics**: Execution times and attempt counts

### Log Levels
- **INFO**: General progress and milestones
- **DEBUG**: Detailed execution information (in log file)
- **WARNING**: Non-critical issues
- **ERROR**: Execution failures and exceptions

Example log entry:
```
================================================================================
EXECUTION ATTEMPT #1 - 2024-11-04 10:30:45
================================================================================

STATUS: SUCCESS
FLAG: FLG:EXAMPLE123

--- GENERATED CODE (456 chars) ---
[Full Python code here]

--- EXECUTION OUTPUT ---
[Program output here]

--- ERRORS ---
(No errors)

--- EXECUTION METADATA ---
{
  "timestamp": "2024-11-04T10:30:45",
  "attempt": 1,
  "success": true,
  "flag_found": "FLG:EXAMPLE123",
  "execution_time": 1.23
}
```

## Success Criteria

The agent considers a task successfully solved when:
- The generated code executes without fatal errors
- The output contains a flag in format: `FLG:XXXXX`

## Safety Features

- **Code Validation**: Checks for potentially dangerous patterns
- **Subprocess Isolation**: Runs code in separate process with restrictions
- **Docker Support**: Optional Docker container isolation
- **Resource Limits**: Memory and CPU limits for execution
- **Timeout Protection**: Prevents infinite loops

## Troubleshooting

### OpenAI Issues

#### API Key Not Working
Ensure your API key is correctly set:
```bash
# Check if environment variable is set
echo $OPENAI_API_KEY

# Or verify in Python
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

#### Rate Limits
If you encounter rate limits, the agent will automatically retry with exponential backoff.

### Ollama Connection Issues

Ensure Ollama is running:
```bash
ollama serve
```

### Model Not Found

For Ollama:
```bash
ollama pull qwen2.5:14b-instruct
ollama pull qwen2.5-coder:14b
```

For OpenAI, ensure you have access to the specified models in your account.

### Provider Auto-Detection

The agent automatically selects the best available provider:
1. If `OPENAI_API_KEY` is set and valid → Uses OpenAI
2. If Ollama is running and accessible → Uses Ollama
3. Otherwise → Raises an error

You can force a specific provider with the `--provider` flag.

## Example Lesson Format

Lessons should be in Markdown format with clear homework sections:

```markdown
# Lesson Title

## Content
...

## Homework

Your task is to...

URLs:
- https://example.com/data.txt

Requirements:
1. Step one
2. Step two

Expected output: FLG:XXXXX
```

## Development

### Adding New Models

Edit `meta_agent_main.py` to change the default model or add new model options.

### Customizing Safety Restrictions

Modify `SafeCodeExecutor` in `executors/safe_executor.py` to adjust execution restrictions.

### Extending Functionality

The modular design allows easy extension:
- Add new processors in `processors/`
- Add new tools in `tools/`
- Modify execution strategy in `core/strategy.py`

## License

This project is for educational purposes as part of the AI_devs course.

## Support

For issues or questions, please refer to the course materials or create an issue in the project repository.
