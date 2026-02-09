# MARS Usage Guide

## Quick Start

```bash
# 1. Install a provider
uv sync --extra google  # or: openai, anthropic, all

# 2. Set up API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# 3. Run MARS on a task
uv run python -m mars.run --task ./tasks/my-task --budget 1800
```

---

## Step 1: Prepare Your Task

Create a task directory with this structure:

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

### Basic Run (30-minute test)

```bash
uv run python -m mars.run --task ./tasks/my-task --budget 1800
```

### Full Run (24 hours)

```bash
uv run python -m mars.run --task ./tasks/my-task
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | *(required)* | Path to task directory |
| `--budget` | `86400` | Time budget in seconds (86400 = 24h) |
| `--model` | `google_genai:gemini-2.5-pro` | LLM model (`provider:model` format) |
| `--temperature` | `1.0` | LLM sampling temperature |
| `--script-timeout` | `14400` | Per-script timeout in seconds (14400 = 4h) |
| `--max-lessons` | `30` | Max lessons stored per category |
| `--max-debug` | `10` | Max debug attempts per solution |
| `--verbose` / `-v` | off | Enable debug logging |

### Examples

```bash
# Quick test with verbose logging
uv run python -m mars.run --task ./tasks/titanic --budget 1800 -v

# Different providers
uv run python -m mars.run --task ./tasks/titanic --model google_genai:gemini-2.5-pro
uv run python -m mars.run --task ./tasks/titanic --model openai:gpt-4o
uv run python -m mars.run --task ./tasks/titanic --model anthropic:claude-sonnet-4-20250514

# OpenRouter (auto-remaps to OpenAI backend with OpenRouter base URL)
uv run python -m mars.run --task ./tasks/titanic --model openrouter:google/gemini-2.5-pro
uv run python -m mars.run --task ./tasks/titanic --model openrouter:meta-llama/llama-3-70b

# Lower temperature for more deterministic output
uv run python -m mars.run --task ./tasks/titanic --temperature 0.7

# Short script timeout for quick iterations
uv run python -m mars.run --task ./tasks/titanic --budget 3600 --script-timeout 600
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
