---
name: mars-skill
description: Interactive workflow for running MARS (Modular Agent with Reflective Search) ML tasks. Guides through workspace setup, MARS dependency installation, task preparation, and background execution management.
---

# MARS Skill

MARS uses budget-aware MCTS to explore different ML solutions — drafting new ideas,
improving valid solutions, and debugging failures — all within a time budget. It learns
from each iteration via reflective memory, transferring insights across search branches.

## Agent Workflow Overview

```
[FAST-PATH CHECK] → WORKSPACE SETUP → MARS DEPENDENCY → TASK PREPARATION → EXECUTION & MANAGEMENT
```

Follow these phases sequentially. Each phase includes **interactive checkpoints**
where you MUST query the user before proceeding.

---

### Fast-Path: Auto-Detection

Before starting the full workflow, check if the workspace is already set up:

1. **Check cwd for `pyproject.toml`** — does it exist?
2. **Check if `mars` is available** — run `uv run mars list` (suppress errors)
3. **Check for existing tasks** — does `tasks/` have any subdirectories with `description.md` and `input/`?

**If all three pass**: Skip to Phase 2 Step 3 (task selection) and present the user
with existing tasks. Ask: "Workspace is ready and has N tasks. Which task do you want
to run, or create a new one?"

**If partially set up**: Jump to the first incomplete phase.

**If nothing is set up**: Start from Phase 1.

---

### Phase 1: Workspace Setup

#### Step 1: Determine Workspace Directory

**Action**: Ask the user if the current working directory is the target project workspace.

The workspace is the project directory where:
- `pyproject.toml` exists (or will be created)
- `tasks/` directory will contain MARS tasks
- `uv run mars ...` commands will be executed
- `.env` file holds API keys

> **Important**: The workspace does NOT have to be the MARS repository itself.
> MARS can be installed as a dependency in any Python project.

If the user specifies a different directory, **`cd` to that directory** before running
any `mars` commands. The `tasks/` directory is resolved relative to cwd, so all
`uv run mars` commands must execute from the workspace root.

If the workspace has no `pyproject.toml` yet, initialize one first:
```bash
uv init
```

#### Step 2: Add MARS as Dependency

**Action**: Check if MARS is already installed in the workspace. If not, ask the user how to install it.

**First, detect if the workspace IS the MARS repo itself:**
- Read `pyproject.toml` and check if `name = "mars"` and `mars.cli:main` is in `[project.scripts]`
- If yes: MARS is the project itself, not a dependency. **Skip dependency installation.**
  Just ensure provider extras are installed:
  ```bash
  uv sync --extra google   # or: openai, anthropic, all
  ```
  Then proceed to Step 2.4 (API key verification).

**If the workspace is NOT the MARS repo:**

1. **Check existing dependencies**: Look at `{WORKSPACE}/pyproject.toml` for `mars` in `[project.dependencies]`
2. **If not installed**, ask the user:
   - **Is MARS already cloned locally?**
     - Yes → `uv add --editable "<PATH-TO-MARS>[google]"` (for development / local changes)
     - No → `uv add "mars[google] @ git+https://github.com/c0sogi/MARS.git"` (for standard use)
   - Replace `[google]` with the desired provider extra: `[google]`, `[openai]`, `[anthropic]`, or `[all]`
3. **If already installed but missing provider**, add with extras:
   ```bash
   # Re-add with extras (uv will update the dependency)
   uv add "mars[google] @ git+https://github.com/c0sogi/MARS.git"
   ```

**Provider extras and their model strings:**

| Extra | Provider | Model String Example |
|-------|----------|---------------------|
| `[google]` | Google Gemini | `google_genai:gemini-3-pro-preview` |
| `[openai]` | OpenAI | `openai:gpt-4o` |
| `[anthropic]` | Anthropic | `anthropic:claude-sonnet-4-20250514` |
| `[all]` | All providers | Any of the above |

4. **Verify API keys**: Check that `.env` exists in the workspace root with the required key.
   If `.env` doesn't exist, ask the user for the API key and create it:
   - `GOOGLE_API_KEY` for Google / Gemini
   - `OPENAI_API_KEY` for OpenAI
   - `ANTHROPIC_API_KEY` for Anthropic
   - `OPENROUTER_API_KEY` for OpenRouter

See `references/commands.md` for the full model and provider reference.

---

### Phase 2: Task Preparation

#### Step 3: Select or Create Task

**Action**: Check `{WORKSPACE}/tasks/` for existing tasks, then ask the user which task to work on.

1. **Discover existing tasks**:
   ```bash
   uv run mars list
   ```
   This shows all tasks with their status, description, input data, and best metrics.

2. **For existing tasks with previous results** (`workspace/` directory exists):
   - Tell the user: "This task has previous results (best metric: X)."
   - Ask: "Continue from here, or clean workspace and restart?"
   - If clean: `rm -rf tasks/<name>/workspace`

3. **Ask the user** whether to run an existing task or create a new one.

4. **For new tasks**, gather information BEFORE creating the task:
   - **Target variable** — what are we predicting?
   - **Evaluation metric** — how is success measured? (direction: higher/lower is better)
   - **Data format** — CSV, parquet, images, time-series, etc.
   - **Domain context** — what domain knowledge helps the model?

   Based on the answers, choose the appropriate template and name:
   ```bash
   uv run mars init <name> --template <type>
   # Templates: generic | classification | regression | nlp | vision
   ```

#### Step 4: Verify Task Readiness

**Action**: Check that the selected task has all required components. Query the user about any missing items.

**Required components checklist:**

| Component | Path | Status Check |
|-----------|------|--------------|
| Input data | `tasks/<name>/input/` | Directory exists and contains data files |
| Description | `tasks/<name>/description.md` | File exists and is well-written |

**If input data is missing**, ask the user:
- Where is the source data?
- What format is it in?
- Does it need preprocessing / feature extraction?

For **tabular data** (CSV/parquet): copy directly to `input/`.
For **large/raw data** (images, time-series, archives): help write a feature extraction script first.
For **GitHub projects**: clone repo, follow setup instructions, extract features.

Keep `input/` under 500 MB. MARS iterates many times, so smaller is faster.

**If description.md is missing or incomplete**, the agent should **proactively draft it**:
1. Read the input data files (e.g., `pd.read_parquet()`, `pd.read_csv()`)
2. Compute statistics: shape, column names, dtypes, value ranges, distributions, missing values
3. Combine with the user's answers from Step 3 (target, metric, domain knowledge)
4. Write a draft `description.md` following `references/description-guide.md`
5. Present the draft to the user for review and refinement

Essential sections:
1. **Problem Overview** — what the problem is and why it's challenging
2. **Metric** — exact metric name, direction (higher/lower is better)
3. **Data** — file names, shapes, column descriptions, statistics
4. **Domain Knowledge** — 5-10 insights about feature-target relationships
5. **Validation Strategy** — how to split data without leakage
6. **Goal** — must end with `Final Validation Metric: <value>` print format

> **Critical**: The `Final Validation Metric: <value>` stdout format is how MARS
> extracts performance scores. Without it, metric tracking fails.

---

### Phase 3: Task Execution & Management

#### Step 5: Run MARS (Background)

**Action**: Launch MARS as a background process and confirm it started successfully.

```bash
uv run mars run <task-name> --model <provider:model> --budget <seconds>
```

Use the Bash tool's `run_in_background: true` to keep the process running while you
continue to monitor.

**Recommended starting parameters:**

| Scenario | Budget | Model | Notes |
|----------|--------|-------|-------|
| Quick test | `1800` (30m) | `google_genai:gemini-3-flash-preview` | Fast iteration, 1-3 ideas |
| Standard run | `3600` (1h) | `google_genai:gemini-3-pro-preview` | Good balance, 2-5 ideas |
| Full exploration | `86400` (24h) | `google_genai:gemini-3-pro-preview` | Maximum quality |

Additional useful flags:
- `--max-debug 5` — reduce debug attempts for faster iteration (default: 10)
- `--temperature 0.7` — lower temperature for more consistent code
- `--script-timeout 600` — reduce per-script timeout for small datasets
- `-v` — verbose logging

See `references/commands.md` for the full command reference and model options.

#### Step 6: Monitor Progress

**Action**: Periodically check MARS progress while it runs in background.

**Primary method** — read the background task output:
```
Use TaskOutput tool with the background task ID to read live output.
Check periodically (every 2-5 minutes for short runs, less frequently for long runs).
```

**Secondary method** — check the search tree file directly:
```bash
cat tasks/<task-name>/workspace/tree.txt
```
This file updates after each node and gives a quick visual overview of progress.

**What to watch for in the output:**
- `Draft Phase (idea N)` — MARS is trying a new approach
- `Improve Phase` — MARS is improving an existing valid solution
- `Debug attempt K/10` — MARS is fixing bugs in generated code
- `New best node: <id> (metric=X)` — a new best solution was found
- `Tree has N nodes, M valid` — progress summary
- `MARS complete` — run finished

**Warning signs:**
- Debug attempts reaching 8-10/10 repeatedly → description.md needs improvement
- Only `bug` nodes after many iterations → data path or description issue
- Same error type repeating → structural problem, reduce `--max-debug`

#### Step 7: Check Results & Iterate

**Action**: When MARS completes, examine results and offer next steps to the user.

```bash
# Search tree overview
cat tasks/<task-name>/workspace/tree.txt

# Best solution code
cat tasks/<task-name>/workspace/best_solution/runfile.py

# Run best solution independently to verify (use absolute path to avoid cwd change)
uv run python tasks/<task-name>/workspace/best_solution/runfile.py
```

**Reading tree.txt:**
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
            ● 545.120 (best) (ID: node_00004)  ← Best solution
    ◍ bug (ID: node_00008)                ← Second idea, failed
        ◍ bug (ID: node_00009)            ← Debug attempts also failed
```

**If results are poor, offer re-run strategies:**

| Action | When | Expected Impact |
|--------|------|-----------------|
| **Improve description.md** | Tree has many `◍ bug` nodes | High |
| **Add more domain knowledge** | Valid solutions but bad metric | High |
| **Increase budget** | Few nodes explored (< 5) | Medium |
| **Change model** | Code quality issues | Medium |
| **Reduce data size** | Scripts timing out | Medium |
| **Lower `--max-debug`** | Too much time on bugs | Low |

Before re-running, clean workspace:
```bash
rm -rf tasks/<task-name>/workspace
```

---

## Data Path Mechanics

MARS executes generated scripts in `workspace/<idea>/<node>/` subdirectories.
It automatically creates a symlink (or Windows junction) from `./input` in each
node directory back to `tasks/<task-name>/input/`. Generated scripts reference
`./input/` as a relative path. Similarly, `library/` modules are importable via
PYTHONPATH injection.

## Result Directory Structure

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

## GitHub Project Workflow

When the user provides a GitHub URL:

1. `git clone <url> <local-path>`
2. Read the repo's README, challenge descriptions, data docs
3. Follow the repo's data setup instructions (download, extract, etc.)
4. If data is large/raw → write a feature extraction script
5. Continue from Phase 2, Step 3 (task selection)

## Troubleshooting

See `references/troubleshooting.md` for common issues and fixes.

Quick fixes:
- **No metric extracted** → Ensure `Final Validation Metric: <value>` is printed to stdout
- **Same error repeating** → Reduce `--max-debug`, improve description.md
- **Budget exceeded (2x+)** → Use `--max-debug 5`
- **Windows encoding** → All `open()` calls need `encoding="utf-8"`
- **All nodes are bugs** → Check `input/` files and column names in description.md
