# Writing an Effective description.md

The quality of `description.md` is the single biggest factor in MARS's output quality.
A well-written description gives the LLM everything it needs to generate correct,
domain-aware code on the first attempt. A vague description leads to wasted budget
on bad solutions.

## Template

```markdown
# <Task Name>

## Problem Overview
<1-2 paragraphs: what the problem is, the domain, and what makes it challenging>

## Metric
<metric_name> (<lower/higher> is better)
<Additional details or formulas if the metric is non-standard>

## Data

### Data Location and Statistics
- **Total samples**: <N>
- **Feature columns**: <N> features + <N> metadata = <N> total
- **Target range**: [min, max], mean ≈ <value>
- **File size**: <size>

All data files are in `input/`:
- `<filename>` — <description of contents, shape, format>

### Feature Description
<Describe every column or feature group. Be specific about names and types.>

### Domain Knowledge (Critical)
<List 5-10 domain-specific insights:>
1. <Key relationship between features and target>
2. <Known constraints or invariants>
3. <Important feature interactions>
4. <Data quality issues or gotchas>

### Recommended Modeling Approaches
1. <Approach 1> — <why it's suitable>
2. <Approach 2> — <alternative strategy>

### Validation Strategy
<How to split data to avoid leakage. Group-based? Time-based? Stratified?>

## Goal
Build a model that predicts <target>. The model should:
1. Load training data from `input/<filename>`
2. Train a regression/classification model
3. Evaluate using <validation strategy>
4. Print the final validation score as: `Final Validation Metric: <value>`
```

## Tips

| Tip | Why |
|-----|-----|
| List ALL column names explicitly | LLM generates correct `df['column']` references |
| Include data statistics (shape, ranges, distributions) | LLM chooses appropriate preprocessing |
| Specify validation strategy precisely | Prevents data leakage in generated code |
| Add domain knowledge (5-10 insights) | LLM generates domain-aware features and constraints |
| Mention file format and loading method | `pd.read_parquet()` vs `pd.read_csv()` etc. |
| State metric direction (lower/higher is better) | MARS tracks improvement correctly |
| Include `Final Validation Metric: <value>` format | MARS extracts metrics from stdout |

## Common Mistakes

1. **Vague column descriptions** — "features include various measurements" is useless.
   Name every column and its type/range.

2. **Missing metric direction** — MARS needs to know if lower or higher is better
   to track improvement correctly.

3. **No validation strategy** — Without explicit instructions, generated code may
   create data leakage (e.g., fitting scalers on the full dataset before splitting).

4. **Forgetting the print format** — `Final Validation Metric: <value>` in stdout
   is how MARS extracts scores. Without it, the metric may not be captured.

5. **No domain knowledge** — Generic ML code ignores domain-specific relationships
   that could significantly improve performance.
