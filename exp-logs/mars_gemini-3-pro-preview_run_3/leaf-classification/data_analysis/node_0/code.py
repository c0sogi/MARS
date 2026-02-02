import os
import pandas as pd
import numpy as np
import cv2
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy import stats


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def main():
    seed_everything()

    # ==========================================
    # 1. Data Loading
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Separate features and target
    target_col = "species"
    id_col = "id"
    path_col = "file_path"

    # Identify feature columns
    # Based on description: margin_1..64, shape_1..64, texture_1..64
    margin_cols = [c for c in df.columns if c.startswith("margin")]
    shape_cols = [c for c in df.columns if c.startswith("shape")]
    texture_cols = [c for c in df.columns if c.startswith("texture")]
    feature_cols = margin_cols + shape_cols + texture_cols

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")
    print("========================")

    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    min_count = class_counts.min()
    max_count = class_counts.max()
    mean_count = class_counts.mean()
    std_count = class_counts.std()

    print(f"Target Variable: '{target_col}' (Categorical)")
    print(f"Number of Classes: {num_classes}")
    print(f"Class Balance Statistics:")
    print(f"  Min Samples per Class: {min_count}")
    print(f"  Max Samples per Class: {max_count}")
    print(f"  Mean Samples per Class: {mean_count:.4f}")
    print(f"  Std Dev of Samples:     {std_count:.4f}")

    # Check for severe imbalance (e.g., ratio of max/min > 10)
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f"  Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")
    print("")

    # ==========================================
    # 3. Input Data Analysis (Tabular)
    # ==========================================
    print("INPUT DATA ANALYSIS (TABULAR)")
    print("=============================")

    # Missing Values
    total_cells = len(df) * len(feature_cols)
    total_missing = df[feature_cols].isnull().sum().sum()
    print(
        f"Missing Values: {total_missing} / {total_cells} ({total_missing/total_cells*100:.4f}%)"
    )

    # Numerical Stats Aggregated by Feature Group to avoid 192 lines of output
    def analyze_feature_group(group_name, cols):
        sub_df = df[cols]
        # Global stats for the group
        g_mean = sub_df.values.mean()
        g_std = sub_df.values.std()
        g_min = sub_df.values.min()
        g_max = sub_df.values.max()

        # Outliers (IQR method)
        Q1 = sub_df.quantile(0.25)
        Q3 = sub_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (
            ((sub_df < (Q1 - 1.5 * IQR)) | (sub_df > (Q3 + 1.5 * IQR))).sum().sum()
        )
        total_group_cells = len(sub_df) * len(cols)

        print(f"Group: {group_name} ({len(cols)} features)")
        print(f"  Global Mean: {g_mean:.4f}")
        print(f"  Global Std:  {g_std:.4f}")
        print(f"  Range:       [{g_min:.4f}, {g_max:.4f}]")
        print(f"  Outliers:    {outliers} ({outliers/total_group_cells*100:.2f}%)")

    analyze_feature_group("Margin", margin_cols)
    analyze_feature_group("Shape", shape_cols)
    analyze_feature_group("Texture", texture_cols)
    print("")

    # ==========================================
    # 4. Input Data Analysis (Image)
    # ==========================================
    print("INPUT DATA ANALYSIS (IMAGE)")
    print("===========================")

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # Pixel stats accumulators
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    # We will iterate over the dataset to gather image stats
    # Since dataset is small (~700 training images), this is feasible.

    valid_image_count = 0

    for idx, row in df.iterrows():
        rel_path = row[path_col]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # Read image
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        valid_image_count += 1

        # Dimensions
        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels.append(c)

        # Pixel stats (normalize to 0-1 for calculation)
        img_norm = img.astype(np.float32) / 255.0
        pixel_sum += np.sum(img_norm)
        pixel_sq_sum += np.sum(img_norm**2)
        total_pixels += w * h * c

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(f"Analyzed {valid_image_count} images.")
    print(
        f"Dimensions (Width):  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"Dimensions (Height): Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"Aspect Ratios:       Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}"
    )

    # Channel distribution
    unique_channels, counts_channels = np.unique(channels, return_counts=True)
    print(f"Channel Counts: {dict(zip(unique_channels, counts_channels))}")

    # Global Pixel Stats
    if total_pixels > 0:
        global_mean = pixel_sum / total_pixels
        global_var = (pixel_sq_sum / total_pixels) - (global_mean**2)
        global_std = np.sqrt(global_var)
        print(f"Pixel Values (0-1):  Mean={global_mean:.4f}, Std={global_std:.4f}")
    else:
        print("Pixel Values: N/A (No pixels processed)")
    print("")

    # ==========================================
    # 5. Feature/Signal Relationships
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("============================")

    # 5a. Structured (Tabular) Relationships
    # Correlation
    # Calculating full correlation matrix
    corr_matrix = df[feature_cols].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than 0.90
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    collinear_pairs_count = len(
        to_drop
    )  # This is a rough proxy, strictly it counts redundant features

    # Exact pair count
    high_corr_pairs = np.sum((upper.values > 0.90))

    print(f"Highly Correlated Feature Pairs (>0.90): {high_corr_pairs}")

    # Feature Importance (Random Forest)
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_col])
    X = df[feature_cols]

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Important Features (Random Forest):")
    for i in range(5):
        print(f"  {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")

    # 5b. Unstructured (Meta-Feature) Relationships
    # Relationship between Aspect Ratio and Species
    # We perform a quick ANOVA to see if Aspect Ratio differs by Species

    # Create a temporary dataframe for this analysis
    meta_df = pd.DataFrame({"species": df[target_col], "aspect_ratio": aspect_ratios})

    # Group aspect ratios by species
    groups = [
        group["aspect_ratio"].values for name, group in meta_df.groupby("species")
    ]

    # Perform One-Way ANOVA
    # Null hypothesis: The mean aspect ratio is the same for all species
    f_val, p_val = stats.f_oneway(*groups)

    print("\nMeta-Feature Analysis (Aspect Ratio vs Species):")
    print(f"  One-way ANOVA F-value: {f_val:.4f}")
    print(f"  One-way ANOVA p-value: {p_val:.4e}")
    if p_val < 0.05:
        print(
            "  Result: Significant relationship detected. Aspect ratio varies by species."
        )
    else:
        print("  Result: No significant relationship detected.")


if __name__ == "__main__":
    main()
