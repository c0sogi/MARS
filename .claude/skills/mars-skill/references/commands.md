# MARS Command Reference

## mars init

```bash
uv run mars init <name> [--template TYPE] [--dir PATH]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `name` | *(required)* | Task name (creates `tasks/<name>/`) |
| `--template` | `generic` | `generic`, `classification`, `regression`, `nlp`, `vision` |
| `--dir` | `tasks` | Parent directory for tasks |

## mars run

```bash
uv run mars run <task> [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `task` | *(required)* | Task name or full path |
| `--budget` | `86400` (24h) | Time budget in seconds |
| `--model` | `google_genai:gemini-3-pro-preview` | `provider:model` format |
| `--temperature` | `1.0` | LLM sampling temperature |
| `--script-timeout` | `14400` (4h) | Per-script timeout in seconds |
| `--max-lessons` | `30` | Max lessons stored per category |
| `--max-debug` | `10` | Max debug attempts per solution |
| `--verbose` / `-v` | off | Debug logging |
| `--dir` | `tasks` | Tasks directory |

## mars list

```bash
uv run mars list [--dir PATH]
```

Shows all tasks with status, description, input, workspace, and best metric.

## Model Options

```bash
# Google
--model google_genai:gemini-3-pro-preview
--model google_genai:gemini-3-flash-preview

# OpenAI
--model openai:gpt-4o
--model openai:o3-mini

# Anthropic
--model anthropic:claude-sonnet-4-20250514

# OpenRouter (any model via single key)
--model openrouter:google/gemini-3-pro-preview
```

## Provider Setup

```bash
# Install provider
uv sync --extra google     # or: openai, anthropic, all

# .env file
GOOGLE_API_KEY=your-key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```
