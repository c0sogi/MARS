"""Prompt template for model architecture search (Appendix F.6)."""

from __future__ import annotations


def format_prompt(*, task_description: str, num_model_candidates: int) -> str:
    return f"""\
==== Task ====
Your task is to propose {num_model_candidates} distinct model \
architectures to solve the problem. **Action:** Use Google Search to \
research state-of-the-art and efficient architectures relevant to this \
domain.

# Requirements
- **Broad Diversity:** The candidates must represent different \
algorithmic families. Do not propose multiple variations of the same \
underlying method (e.g., do not suggest two different ResNets). Aim \
for a mix of:
    * Instance-Based / Kernel Methods (e.g., k-NN, SVM)
    * Tree-Based Ensembles (e.g., LightGBM, XGBoost, CatBoost)
    * Deep Learning (e.g., CNN, MLP, Transformers, RNNs)
- **Problem Alignment:** Architectures must be specifically tailored to \
the data modality (e.g., tabular, image, time-series) and input \
structure.
- **Hybridization:** Incorporate hybrid or ensemble designs if they offer \
a clear advantage for heterogeneous data.
- **Efficiency First:** Prioritize "lightweight" designs. For each family\
, choose the architecture that offers the best trade-off between low \
computational cost (fast training/inference) and high performance.
- **Data Constraints:** If the training data is limited, explicitly \
address regularization or low-complexity designs to prevent \
overfitting.
- For each model, create a JSON object with the following two keys:
    - `reasoning`: Justification for why this architecture fits the \
constraints (efficiency, data size, and why it was chosen over others \
in its category).
    - `description`: A technical description of the architecture and \
design philosophy.

# Response Format
Your response should be in the following JSON format in a single markdown \
code block (wrapped in ```):
```json
[
    {{"reasoning": "k-NN is small and efficient...", "description": "We \
can use K-NN for this task..."}},
    {{"reasoning": "CNN is effective and efficient...", "description": "\
We can use CNN for this task..."}},
    {{"reasoning": "GBMs is an effective model...", "description": "We \
can use GBMs for this task..."}},
]
```

# Task Description
{task_description}
"""
