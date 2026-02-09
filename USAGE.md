# MARS Usage Guide

## Quick Start

```bash
# 1. Install MARS with a provider
uv sync --extra google  # or: openai, anthropic, all

# 2. Set up API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# 3. Create and run a task
mars init my-task --template classification
# edit tasks/my-task/description.md, copy data to input/
mars run my-task --budget 1800
```

---

## CLI Commands

MARS provides three subcommands: `init`, `run`, and `list`.

### `mars init` — Create a new task

```bash
mars init <name> [--template TYPE] [--dir PATH]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `name` | *(required)* | Task name (creates `tasks/<name>/`) |
| `--template` | `generic` | Template: `generic`, `classification`, `regression`, `nlp`, `vision` |
| `--dir` | `tasks` | Parent directory for tasks |

Creates the task directory with `description.md` and `input/`:

```bash
$ mars init titanic --template classification
Created tasks/titanic/
  tasks/titanic/description.md   <- edit this
  tasks/titanic/input/           <- put data here

Next: edit description.md, copy data to input/, then run:
  mars run titanic --budget 1800
```

### `mars run` — Run MARS on a task

```bash
mars run <task> [--budget N] [--model M] [--temperature T] ...
```

| Argument | Default | Description |
|----------|---------|-------------|
| `task` | *(required)* | Task name (resolved to `tasks/<name>/`) or full path |
| `--budget` | `86400` | Time budget in seconds (86400 = 24h) |
| `--model` | `google_genai:gemini-2.5-pro` | LLM model (`provider:model` format) |
| `--temperature` | `1.0` | LLM sampling temperature |
| `--script-timeout` | `14400` | Per-script timeout in seconds (14400 = 4h) |
| `--max-lessons` | `30` | Max lessons stored per category |
| `--max-debug` | `10` | Max debug attempts per solution |
| `--verbose` / `-v` | off | Enable debug logging |
| `--dir` | `tasks` | Tasks directory |

Examples:

```bash
# Quick test (30 minutes)
mars run my-task --budget 1800

# Full run (24 hours, default)
mars run my-task

# Different providers
mars run my-task --model google_genai:gemini-2.5-pro
mars run my-task --model openai:gpt-4o
mars run my-task --model anthropic:claude-sonnet-4-20250514

# OpenRouter
mars run my-task --model openrouter:google/gemini-2.5-pro

# Run by full path instead of name
mars run ./my-custom-path/task --budget 3600

# Verbose logging with lower temperature
mars run my-task --budget 1800 -v --temperature 0.7
```

### `mars list` — List available tasks

```bash
mars list [--dir PATH]
```

Shows all tasks with their status:

```bash
$ mars list
TASK                 DESCRIPTION   INPUT    WORKSPACE   BEST METRIC
titanic              yes           yes      no          -
house-prices         yes           yes      yes         0.142
```

---

## Step 1: Prepare Your Task

You can create a task automatically with `mars init`, or manually:

```
tasks/my-task/
├── description.md          # Task description (required)
└── input/                  # Data files (required)
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

### Writing `description.md`

Include these sections for best results:

```markdown
# Task Name

Brief description of the problem.

## Metric
Accuracy (higher is better)

## Data
- train.csv: training data with labels
- test.csv: test data without labels
- sample_submission.csv: expected output format

## Goal
Build a model that predicts [target]. Output predictions to submission.csv.
Print the final validation score as: Final Validation Metric: <value>
```

> **Important:** MARS looks for `Final Validation Metric: <value>` in stdout
> to track performance. Make sure your description mentions this output format.

---

## Step 2: Install Provider & Set Up API Key

MARS supports multiple LLM providers. Install only the one you need:

```bash
uv sync --extra google     # Google Gemini
uv sync --extra openai     # OpenAI
uv sync --extra anthropic  # Anthropic Claude
uv sync --extra all        # All providers
```

Create a `.env` file with the API key for your chosen provider:

```bash
# Google Gemini (default)
GOOGLE_API_KEY=your-gemini-api-key

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter (access 100+ models via single key)
OPENROUTER_API_KEY=sk-or-...
```

Get your keys from:
- Gemini: [Google AI Studio](https://aistudio.google.com/apikey)
- OpenAI: [OpenAI Platform](https://platform.openai.com/api-keys)
- Anthropic: [Anthropic Console](https://console.anthropic.com/)
- OpenRouter: [OpenRouter](https://openrouter.ai/keys)

---

## Step 3: Run MARS

See the [`mars run` command](#mars-run--run-mars-on-a-task) above.

The legacy invocation still works:

```bash
uv run python -m mars.run --task ./tasks/my-task --budget 1800
```

---

## Step 4: Check Results

After running, check `tasks/my-task/workspace/`:

```
workspace/
├── logs/
│   └── mars.log                    # Full execution log
├── tree.txt                        # MCTS search tree visualization
├── best_solution/                  # Best performing solution
│   ├── library/                    #   Module files
│   │   ├── data_processing.py
│   │   ├── feature_engineering.py
│   │   └── model.py
│   └── main.py                     #   Main entry script
├── node_00001/                     # Each MCTS node's solution
│   ├── library/
│   └── main.py
├── saved_solution_lessons/         # Learned solution strategies
│   └── lessons.json
└── saved_debug_lessons/            # Learned debugging patterns
    └── lessons.json
```

### Reading `tree.txt`

```
Root
├── [node_00001] draft | metric=0.82 | time=120s | R=0.75
│   ├── [node_00002] improve | metric=0.85 | time=95s | R=0.82  ★ best
│   └── [node_00003] improve | metric=0.83 | time=110s | R=0.78
└── [node_00004] draft | metric=0.79 | time=200s | R=0.68
```

- **draft** = new idea from scratch
- **improve** = improved from parent solution
- **R** = reward score (higher is better)

### Running the Best Solution

```bash
cd tasks/my-task/workspace/best_solution
python main.py
```

---

## How MARS Works

MARS uses Monte Carlo Tree Search (MCTS) to explore different ML solutions:

```
1. PREPARATION
   Parse metric → Generate metadata → Run EDA → Search architectures

2. MCTS LOOP (repeats until budget runs out)
   ┌─ SELECT node (UCT formula)
   ├─ DRAFT new idea  or  IMPROVE existing solution
   ├─ DECOMPOSE into modules → IMPLEMENT each → WRITE main script
   ├─ EXECUTE and check for errors
   ├─ If buggy → DEBUG (up to 10 attempts)
   ├─ REVIEW results → extract metric
   ├─ LEARN lessons from experience
   └─ UPDATE search tree with reward

3. OUTPUT best solution
```

Each iteration tries a different approach and learns from both successes and failures.

---

## Troubleshooting

### "No description file found"
Make sure your task directory has `description.md` (or `description.txt`, `README.md`, `task.md`).

### "Warning: Input directory not found"
Create an `input/` folder inside your task directory with data files.

### Script execution timeout
Increase `--script-timeout` or reduce data size. Default is 4 hours per script.

### API rate limits
MARS makes many LLM calls. If you hit rate limits, try:
- Use a faster model: `--model google_genai:gemini-2.5-flash`
- Switch providers: `--model openrouter:google/gemini-2.5-flash`
- Reduce `--max-debug` to fewer debugging attempts

### No metric extracted
Your generated scripts must print `Final Validation Metric: <number>` to stdout. MARS uses this to track performance. If missing, the LLM reviewer will try to extract it from the output, but the explicit format is more reliable.
