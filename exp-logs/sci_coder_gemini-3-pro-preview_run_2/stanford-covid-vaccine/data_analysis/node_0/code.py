import os
import ast
import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
METADATA_PATH = "./metadata/train.csv"
SEED = 42

# Set random seeds
np.random.seed(SEED)


def parse_list_column(x):
    """Parses a stringified list into a numpy array."""
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def main():
    print("STARTING EXPLORATORY DATA ANALYSIS\n")

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found.")
        return

    df = pd.read_csv(METADATA_PATH)

    # Parse target columns which are stored as string representations of lists
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # The scored targets for the competition metric are usually: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Convert string lists to numpy arrays
    for col in target_cols:
        df[col] = df[col].apply(parse_list_column)

    # 2. Target Variable Analysis
    print("==== TARGET VARIABLE ANALYSIS ====")

    # We will analyze the distribution of the values within the scored vectors.
    # Flattening all values from all samples for the scored columns.

    stats_dict = {}

    for col in scored_cols:
        # Concatenate all arrays in the column
        all_values = np.concatenate(df[col].values)

        mu = np.mean(all_values)
        sigma = np.std(all_values)
        skew = stats.skew(all_values)
        kurt = stats.kurtosis(all_values)
        min_val = np.min(all_values)
        max_val = np.max(all_values)

        stats_dict[col] = {
            "mean": mu,
            "std": sigma,
            "skew": skew,
            "kurt": kurt,
            "min": min_val,
            "max": max_val,
            "count": len(all_values),
        }

    print(
        f"{'Target':<15} {'Mean':<10} {'Std':<10} {'Skew':<10} {'Kurtosis':<10} {'Min':<10} {'Max':<10}"
    )
    print("-" * 80)
    for col, s in stats_dict.items():
        print(
            f"{col:<15} {s['mean']:.4f}     {s['std']:.4f}     {s['skew']:.4f}     {s['kurt']:.4f}     {s['min']:.4f}     {s['max']:.4f}"
        )

    print("\nObservation:")
    print(
        "High kurtosis indicates heavy tails (outliers). Positive skew indicates a tail on the right (higher reactivity/degradation)."
    )

    # 3. Input Data Analysis (Text/Sequence Modality)
    print("\n==== INPUT DATA ANALYSIS (SEQUENCE & STRUCTURE) ====")

    # Sequence Length Analysis
    seq_lengths = df["sequence"].apply(len)
    print(
        f"Sequence Lengths: Min={seq_lengths.min()}, Max={seq_lengths.max()}, Unique={seq_lengths.unique()}"
    )

    # Vocabulary Analysis (Sequence)
    # Concatenate a subset of sequences to get char distribution to save memory/time if needed,
    # but dataset is small enough to do all.
    all_sequences = "".join(df["sequence"].tolist())
    seq_counts = Counter(all_sequences)
    total_bases = sum(seq_counts.values())

    print("\nSequence Vocabulary Distribution (Bases):")
    for char, count in seq_counts.most_common():
        print(f"  {char}: {count} ({count/total_bases*100:.2f}%)")

    # Vocabulary Analysis (Structure)
    all_structures = "".join(df["structure"].tolist())
    struct_counts = Counter(all_structures)
    total_struct = sum(struct_counts.values())

    print("\nStructure Vocabulary Distribution (Dot-Bracket):")
    for char, count in struct_counts.most_common():
        print(f"  {char}: {count} ({count/total_struct*100:.2f}%)")

    # Vocabulary Analysis (Predicted Loop Type)
    all_loops = "".join(df["predicted_loop_type"].tolist())
    loop_counts = Counter(all_loops)
    total_loops = sum(loop_counts.values())

    print("\nPredicted Loop Type Distribution:")
    for char, count in loop_counts.most_common():
        print(f"  {char}: {count} ({count/total_loops*100:.2f}%)")

    # 4. Input Data Analysis (Tabular Metadata)
    print("\n==== INPUT DATA ANALYSIS (TABULAR METADATA) ====")

    # Signal to Noise
    sn = df["signal_to_noise"]
    q1 = np.percentile(sn, 25)
    q3 = np.percentile(sn, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = sn[(sn < lower_bound) | (sn > upper_bound)]

    print(f"Feature: signal_to_noise")
    print(f"  Mean: {sn.mean():.4f}")
    print(f"  Std : {sn.std():.4f}")
    print(f"  Min : {sn.min():.4f}")
    print(f"  Max : {sn.max():.4f}")
    print(
        f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(sn)*100:.2f}%)"
    )

    # 5. Feature/Signal Relationships
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # A. Target Correlations
    # We calculate the mean value per sample for each target to see if samples with high reactivity
    # also have high degradation.
    mean_targets = pd.DataFrame()
    for col in scored_cols:
        mean_targets[f"mean_{col}"] = df[col].apply(np.mean)

    corr_matrix = mean_targets.corr(method="pearson")
    print("Correlation between Mean Target Values per Sample:")
    print(corr_matrix.round(4))

    # B. Metadata vs Target Correlation
    # Correlation between signal_to_noise and mean reactivity
    sn_corr = np.corrcoef(df["signal_to_noise"], mean_targets["mean_reactivity"])[0, 1]
    print(f"\nCorrelation between signal_to_noise and mean_reactivity: {sn_corr:.4f}")

    # C. Feature Importance (Lightweight Random Forest)
    # We will derive some global features from the sequences to predict mean reactivity
    print(
        "\nFeature Importance (Predicting Mean Reactivity from Sequence Composition):"
    )

    # Feature Engineering
    # 1. Base counts
    df["count_A"] = df["sequence"].apply(lambda x: x.count("A"))
    df["count_G"] = df["sequence"].apply(lambda x: x.count("G"))
    df["count_C"] = df["sequence"].apply(lambda x: x.count("C"))
    df["count_U"] = df["sequence"].apply(lambda x: x.count("U"))

    # 2. Structure counts
    df["count_dot"] = df["structure"].apply(lambda x: x.count("."))
    df["count_open"] = df["structure"].apply(lambda x: x.count("("))
    # count_close should be roughly same as count_open

    # 3. Loop type counts (Top 3 most common: S, E, H based on typical RNA data)
    df["count_S"] = df["predicted_loop_type"].apply(lambda x: x.count("S"))  # Stem
    df["count_E"] = df["predicted_loop_type"].apply(lambda x: x.count("E"))  # External
    df["count_H"] = df["predicted_loop_type"].apply(lambda x: x.count("H"))  # Hairpin
    df["count_M"] = df["predicted_loop_type"].apply(lambda x: x.count("M"))  # Multiloop

    features = [
        "count_A",
        "count_G",
        "count_C",
        "count_U",
        "count_dot",
        "count_open",
        "count_S",
        "count_E",
        "count_H",
        "count_M",
        "signal_to_noise",
    ]

    X = df[features]
    y = mean_targets["mean_reactivity"]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Features influencing Mean Reactivity:")
    for i in range(5):
        idx = indices[i]
        print(f"  {i+1}. {features[idx]:<15} (Importance: {importances[idx]:.4f})")

    # Check for redundancy (Collinearity) among derived features
    print("\nChecking for Redundant Features (Correlation > 0.90):")
    feat_corr = X.corr().abs()
    # Select upper triangle of correlation matrix
    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if to_drop:
        for col in to_drop:
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            print(f"  {col} is highly correlated with: {correlated_with}")
    else:
        print("  No highly collinear pairs found among derived features.")


if __name__ == "__main__":
    main()
