# MARS Command Reference

All commands are run from the **workspace directory** (where `pyproject.toml` lives).

## Installation

**If the workspace IS the MARS repo** (developing MARS itself):
```bash
uv sync --extra google   # or: openai, anthropic, all
```

**If the workspace is a separate project** (MARS as dependency):
```bash
# From GitHub (standard use)
uv add "mars[google] @ git+https://github.com/c0sogi/MARS.git"

# From local clone (editable, for development)
uv add --editable "<PATH-TO-MARS>[google]"
```

Replace `[google]` with the desired provider extra: `[google]`, `[openai]`, `[anthropic]`, or `[all]`.

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

| Extra | Provider | Model String |
|-------|----------|-------------|
| `[google]` | Google Gemini | `google_genai:gemini-3-pro-preview`, `google_genai:gemini-3-flash-preview` |
| `[openai]` | OpenAI | `openai:gpt-4o`, `openai:o3-mini` |
| `[anthropic]` | Anthropic | `anthropic:claude-sonnet-4-20250514` |
| *(any)* | OpenRouter | `openrouter:google/gemini-3-pro-preview` (any model via single key) |

## Provider API Keys

Set in `.env` at the workspace root:

```bash
GOOGLE_API_KEY=your-key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```
