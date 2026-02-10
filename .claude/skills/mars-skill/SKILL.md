---
name: mars-skill
description: Run MARS (Modular Agent with Reflective Search) to automatically build ML/data science solutions using Monte Carlo Tree Search. Use this skill whenever the user wants to set up an automated ML pipeline on a local directory or GitHub repo.
---

# MARS Skill

MARS uses budget-aware MCTS to explore different ML solutions — drafting new ideas,
improving valid solutions, and debugging failures — all within a time budget. It learns
from each iteration via reflective memory, transferring insights across search branches.

## Prerequisites

- MARS installed at project root (editable install via `uv`)
  - MARS repo: https://github.com/c0sogi/mars
- Provider installed: `uv sync --extra google` (or `openai`, `anthropic`, `all`)
- API key in `.env` at MARS project root (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY`)
- Python 3.10+

**Important**: All `uv run mars` commands must be executed from the **MARS project root directory** (where `pyproject.toml` is located), not from the task directory.

## Core Workflow

Every MARS task follows these steps regardless of data source:

```
UNDERSTAND → INIT → PREPARE DATA → WRITE description.md → RUN → MONITOR → CHECK RESULTS
```

### Step 1: Understand the Project

Read all available documentation (README, data descriptions, challenge pages).
Identify these four things before doing anything else:

1. **Target variable** — what are we predicting?
2. **Evaluation metric** — how is success measured? (and direction: higher/lower is better)
3. **Data format** — CSV, parquet, images, time-series, etc.
4. **Domain context** — what domain knowledge helps the model?

### Step 2: Initialize the Task

```bash
cd /path/to/mars  # MARS project root (where pyproject.toml lives)
uv run mars init <task-name> --template <type>
# Templates: generic | classification | regression | nlp | vision
```

This creates `tasks/<task-name>/` with `description.md` and `input/`.

### Step 3: Prepare Input Data

Copy or symlink data files into `tasks/<task-name>/input/`.

For **tabular data** (CSV/parquet) — copy directly:
```bash
cp /path/to/train.csv tasks/<task-name>/input/
```

For **large/raw data** (images, time-series, archives) — extract features first:
```python
# prepare_for_mars.py
import pandas as pd
raw = load_raw_data("/path/to/raw")
features = extract_features(raw)  # → DataFrame
features.to_parquet("tasks/<task-name>/input/train_features.parquet", index=False)
```

For **GitHub projects with custom loaders** — use the project's own data utilities:
```python
import sys
sys.path.insert(0, "/path/to/cloned-repo")
from project_package import load_dataset
data = load_dataset()
# Convert to tabular → save as parquet in input/
```

Keep `input/` under 500MB. MARS iterates many times, so smaller is faster.

**How data paths work**: MARS executes generated scripts in `workspace/<idea>/<node>/` subdirectories. It automatically creates a symlink (or Windows junction) from `./input` in each node directory back to your `tasks/<task-name>/input/` directory. Generated scripts reference `./input/` as a relative path. Similarly, `library/` modules are made importable via PYTHONPATH injection.

### Step 4: Write description.md

This is the most critical step. The quality of `description.md` directly determines
the quality of MARS's generated code. Read `references/description-guide.md` for the
full template and tips. The essential sections are:

1. **Problem Overview** — what the problem is and why it's challenging
2. **Metric** — exact metric name, direction (higher/lower is better)
3. **Data** — file names, shapes, column descriptions, statistics
4. **Domain Knowledge** — 5-10 insights about feature-target relationships
5. **Validation Strategy** — how to split data without leakage
6. **Goal** — what the model should do, ending with `Final Validation Metric: <value>`

The `Final Validation Metric: <value>` print format in stdout is how MARS tracks
performance. Without it, metric extraction becomes unreliable.

### Step 5: Run MARS (Background)

**MARS runs for minutes to hours.** Always run it as a background process so you can
continue working and monitor progress.

```bash
# Run from MARS project root, in background
cd /path/to/mars
uv run mars run <task-name> --model <provider:model> --budget <seconds> &
```

For LLM agents using Claude Code or similar tools, use the tool's background execution
capability (e.g., `run_in_background: true` for Bash tool).

Recommended starting parameters:

| Scenario | Budget | Model | Notes |
|----------|--------|-------|-------|
| Quick test | `1800` (30m) | `gemini-3-flash-preview` | Fast iteration, 1-3 ideas |
| Standard run | `3600` (1h) | `gemini-3-pro-preview` | Good balance, 2-5 ideas |
| Full exploration | `86400` (24h) | `gemini-3-pro-preview` | Maximum quality |

Additional useful flags:
- `--max-debug 5` — reduce debug attempts for faster iteration (default: 10)
- `--temperature 0.7` — lower temperature for more consistent code
- `--script-timeout 600` — reduce per-script timeout for small datasets
- `-v` — verbose logging

See `references/commands.md` for the full command reference and model options.

### Step 6: Monitor Progress

While MARS is running, periodically check progress:

```bash
# Check the search tree (updates after each node)
cat tasks/<task-name>/workspace/tree.txt

# Tail the log for live progress
tail -30 <output-file-or-log>

# Look for key events in output
grep -E "New best|MARS complete|Tree has|Draft Phase|Improve Phase" <log>
```

**What to watch for:**
- `Draft Phase (idea N)` — MARS is trying a new approach
- `Improve Phase` — MARS is improving an existing valid solution
- `Debug attempt K/10` — MARS is fixing bugs in generated code
- `New best node: <id> (metric=X)` — a new best solution was found
- `Tree has N nodes, M valid` — progress summary
- `MARS complete` — run finished

**Warning signs during execution:**
- Debug attempts reaching 8-10/10 repeatedly → description.md needs improvement
- Only `bug` nodes in tree after many iterations → data path or description issue
- Same error type in debug lessons → structural problem with generated code

### Step 7: Check Results

```bash
# Search tree overview
cat tasks/<task-name>/workspace/tree.txt

# Best solution code
cat tasks/<task-name>/workspace/best_solution/runfile.py
ls tasks/<task-name>/workspace/best_solution/library/

# Run the best solution independently
cd tasks/<task-name>/workspace/best_solution && uv run python runfile.py
```

Reading `tree.txt`:
- `●` with a metric value = valid solution (has metric)
- `◍ bug` = failed solution (execution error or no valid metric)
- `(best)` = current best solution
- Indentation = parent-child relationship (improve/debug chain)

Example:
```
Solution tree
● -1.000000 (ID: node_00000)              ← Root (no solution)
    ◍ bug (ID: node_00001)                ← Failed (debug child follows)
        ● 561.012 (ID: node_00003)        ← First valid solution
            ● 545.120 (best) (ID: node_00004)  ← Best solution ★
    ◍ bug (ID: node_00008)                ← Second idea, failed
        ◍ bug (ID: node_00009)            ← Debug attempts also failed
```

Result directory structure:
```
workspace/
├── tree.txt                  # Read this first
├── best_solution/            # Winning code (runfile.py + library/)
├── idea_<N>/                 # Each explored idea
│   ├── idea.txt              #   Idea description
│   └── node_<NNNNN>/        #   Solution attempts
├── saved_lessons/            # What MARS learned
│   ├── solution_lesson.json
│   └── node_debug_lesson.json
├── metadata_generation/      # Auto-generated data analysis
└── eda/                      # Exploratory data analysis output
```

---

## Improving Results (Re-run Strategy)

If the first run produces poor results:

| Action | When | Expected Impact |
|--------|------|-----------------|
| **Improve description.md** | Tree has many `◍ bug` nodes | High — better code generation |
| **Add more domain knowledge** | Valid solutions but bad metric | High — smarter features/modeling |
| **Increase budget** | Few nodes explored (< 5) | Medium — more exploration |
| **Change model** | Code quality issues | Medium — better LLM = better code |
| **Reduce data size** | Scripts timing out | Medium — faster iterations |
| **Lower `--max-debug`** | Too much time on bugs | Low — faster exploration |

Before re-running:
```bash
# Option A: Clean workspace and start fresh
rm -rf tasks/<task-name>/workspace

# Option B: Keep workspace (MARS resumes if nodes exist — but may cause conflicts)
# Safer to clean and restart
```

---

## GitHub Project Workflow

When the user provides a GitHub URL:

```
1. git clone <url> <local-path>
2. Read the repo's README, challenge descriptions, data docs
3. Follow the repo's data setup instructions (download, extract, etc.)
4. If data is large/raw → write a feature extraction script
5. Continue from Step 2 of Core Workflow (init → description.md → run)
```

---

## Troubleshooting

For common issues (model not found, encoding errors, Windows-specific problems,
no metric extracted, budget overruns), read `references/troubleshooting.md`.

Key quick fixes:
- **No metric extracted** → Ensure `Final Validation Metric: <value>` is printed to stdout
- **Same error repeating** → Reduce `--max-debug`, improve description.md
- **Budget exceeded (2x+)** → Use `--max-debug 5` (syntax validation and error dedup are built-in)
- **Windows encoding** → All `open()` calls need `encoding="utf-8"`
- **All nodes are bugs** → Check that `input/` contains the correct files with correct names
