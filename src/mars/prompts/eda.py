"""Prompt template for exploratory data analysis (Appendix F.5)."""

from __future__ import annotations


def format_prompt(*, task_description: str, metadata_context: str) -> str:
    return f"""\
==== Task ====
Your task is to write a robust Python script to perform an Exploratory \
Data Analysis (EDA) on the training dataset. The script must adapt its \
analysis based on the data modality (Tabular, Image, Audio, or Text). \
The output should act as a report to inform feature engineering and \
preprocessing strategies.

# Requirements
1. Data Integrity: Ensure all analysis is strictly performed on the \
training set to prevent data leakage.
2. Target Variable Analysis
- Distribution: Calculate the distribution of the target variable.
- Imbalance/Skew:
    - If Classification: Calculate class balance ratios.
    - If Regression: Calculate Skewness and Kurtosis to assess normality.
3. Input Data Analysis (Modality-Specific)
- If Tabular Data:
    - Numerical: Report mean, std, min, max, and outlier counts (IQR \
method).
    - Categorical: Report cardinality; flag columns with > 50 categories \
or rare labels (< 1 percent frequency).
    - Missing Values: Report count/percentage of NaNs per column.
- If Image Data:
    - Dimensions: Analyze distributions of Image Widths, Heights, and \
Aspect Ratios.
    - Channels: Report the distribution of channel counts (e.g., \
Grayscale vs. RGB).
    - Pixel Stats: Calculate the global mean and standard deviation of \
pixel values (for normalization).
- If Audio Data:
    - Signal: Analyze distributions of Duration (seconds), Sampling Rates\
, and Bit Depths.
    - Channels: Check for mono vs. stereo inconsistency.
- If Text Data:
    - Lengths: Analyze distribution of sequence lengths (character and \
word counts).
    - Vocabulary: Report unique vocabulary size and OOV (Out of \
Vocabulary) potential.
4. Feature/Signal Relationships
- Structured (Tabular) Relationships:
    - Correlation: Pearson/Spearman for numerical; Mutual Information for \
categorical.
    - Importance: Train a lightweight Random Forest and report top 5 \
features.
    - Redundancy: Report collinear pairs (Correlation > 0.90).
- Unstructured (Meta-Feature) Relationships: Analyze the relationship \
between metadata and the target (e.g., "Do longer audio files \
correlate with specific classes?", "Are larger images associated with \
higher regression targets?").
5. Formatting & Output
- Organize the output into distinct, capitalized sections.
- Use f-strings to format floats to 4 decimal places for readability.

# Task Description
{task_description}

# Metadata Information
{metadata_context}
"""
