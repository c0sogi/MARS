# Troubleshooting

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `models/<name> is not found` | Wrong model name | Check available models for provider |
| `UnicodeEncodeError: 'cp949'` | Windows Korean locale | All `open()` calls need `encoding="utf-8"` |
| `Python was not found` | Windows Store alias | Use `sys.executable` instead of `"python"` |
| `FileNotFoundError: input/` | Script can't find data | Ensure `input/` symlink exists in node dirs |
| `No metric extracted` | Missing print format | Ensure `Final Validation Metric: <value>` in stdout |
| Budget exceeded (2x+) | Debug loop runs past budget | Use `--max-debug 5` |
| Same error repeating 10x | LLM not applying lessons | Reduce `--max-debug`, improve description.md |
| No description file found | Missing task description | Add `description.md` (or `description.txt`, `README.md`, `task.md`) |
| Input directory not found | No data folder | Create `input/` inside task directory with data files |
| Script execution timeout | Script too slow | Increase `--script-timeout` or reduce data size |
| API rate limits | Too many LLM calls | Use faster model (`gemini-3-flash-preview`), switch provider, reduce `--max-debug` |
| `AssertionError` on LLM response | Model returns thinking blocks (list) | Fixed in llm.py — handles list content blocks |
| All nodes are `◍ bug` | Description or data path issue | Verify `input/` files, column names in description.md |

## Reading tree.txt Symbols

| Symbol | Meaning | Example |
|--------|---------|---------|
| `●` + metric value | Valid solution with metric | `● 545.120 (ID: node_00004)` |
| `●` + `-1.000000` | Root node (no solution) | `● -1.000000 (ID: node_00000)` |
| `◍ bug` | Failed solution (error or no valid metric) | `◍ bug (ID: node_00001)` |
| `(best)` | Current best solution | `● 545.120 (best) (ID: node_00004)` |
| Indentation | Parent-child (improve/debug) | Child indented under parent |

## Windows-Specific Notes

MARS was developed for Linux/Mac. On Windows, these fixes have been applied in the fork:

1. `runner.py` — Uses `sys.executable` instead of bare `python`
2. `runner.py` — Creates symlinks/junctions for `./input` in work directories
3. `runner.py` — Injects `library/` into PYTHONPATH for subprocess
4. Multiple files — All `open("w")` use `encoding="utf-8"`
5. `llm.py` — Handles list content blocks from models with thinking (e.g., gemini-3-pro-preview)
6. `search.py` — Pre-execution syntax validation, repeated error detection, budget-aware debug loop
7. `diff.py` — Post-diff syntax validation with automatic revert

## Monitoring a Running MARS

```bash
# Live progress: tail the output/log file
tail -f <output-file>

# Check tree state (updated after each node)
cat tasks/<task-name>/workspace/tree.txt

# Key events to grep for
grep -E "New best|MARS complete|Tree has|Draft Phase|bug" <log>
```

**Warning signs:**
- Debug attempts consistently reaching 8-10/10 → description.md too vague
- Only `◍ bug` nodes after 30+ minutes → data path issue or broken description
- Identical debug lessons repeating → LLM generating same bad patterns
- Budget significantly exceeded → reduce `--max-debug` to 5

## When Results Are Poor

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| All nodes are bugs | Data path or description error | Fix input/ files, verify column names |
| Valid but bad metric | Weak features or model | Add domain knowledge to description.md |
| Only 1-2 ideas explored | Budget too low or debug loops too long | Increase budget, reduce `--max-debug` |
| Same metric for all valid nodes | Improvement agent not helping | Add more specific modeling suggestions to description.md |

Before re-running, clean the workspace:
```bash
rm -rf tasks/<task-name>/workspace
```

## Debugging Tips

- **Read `tree.txt` first** — it shows the full search tree with metrics and which nodes succeeded/failed.
- **Check `saved_lessons/`** — MARS stores what it learned. If lessons are poor, the description.md likely needs improvement.
- **Look at `metadata_generation/`** — MARS auto-generates data analysis. Verify it understood your data correctly.
- **Read `idea_N/idea.txt`** — See what approaches MARS tried. If ideas are off-target, improve the description.
- **Verbose mode** (`-v`) — enables debug logging, useful for understanding why MARS made certain decisions.
