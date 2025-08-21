# Meta Agent - Autonomous LLM-Powered Homework Solver

## Overview

Meta Agent is an autonomous AI system that reads course lessons, extracts homework tasks, and attempts to solve them using LLM-generated Python code. It features safe code execution, automatic error recovery, and comprehensive logging.

## Key Features

Meta Agent uses a dual-model approach for optimal performance:

- **Decision Model** (`qwen2.5:14b-instruct`): Used for analysis, summarization, strategy planning, and decision-making
- **Coding Model** (`qwen2.5-coder:14b`): Specialized model for generating Python code solutions

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

3. Ensure Ollama is running:
```bash
ollama serve
```

4. Pull the required models:
```bash
# For decision-making and analysis
ollama pull qwen2.5:14b-instruct

# For code generation
ollama pull qwen2.5-coder:14b
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

### With Options

```bash
python meta_agent_main.py lessons/lesson.md \
    --decision-model qwen2.5:14b-instruct \
    --coding-model qwen2.5-coder:14b \
    --max-attempts 5 \
    --output-dir ./output \
    --verbose
```

### Command Line Arguments

- `lesson_file`: Path to the lesson file (required)
- `--decision-model`: Ollama model for analysis and decisions (default: qwen2.5:14b-instruct)
- `--coding-model`: Ollama model for code generation (default: qwen2.5-coder:14b)
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

The agent generates several output files in the specified output directory:

- `summary_[lesson_name].md`: Lesson summary and key takeaways
- `homework_[lesson_name].md`: Extracted homework task
- `solution_attempt_[n].py`: Generated Python code for each attempt
- `report_[lesson_name].md`: Final execution report

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

### Ollama Connection Issues

Ensure Ollama is running:
```bash
ollama serve
```

### Model Not Found

Pull the required model:
```bash
ollama pull llama3.2:latest
```

### Permission Errors

Ensure write permissions for output directory:
```bash
chmod 755 output/
```

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
