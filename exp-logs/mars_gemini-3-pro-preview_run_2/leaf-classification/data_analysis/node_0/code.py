import os
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

# Set constants and seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_eda():
    set_seed(RANDOM_SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Separate features
    feature_cols = [c for c in df.columns if c not in ["id", "species", "image_path"]]
    margin_cols = [c for c in feature_cols if "margin" in c]
    shape_cols = [c for c in feature_cols if "shape" in c]
    texture_cols = [c for c in feature_cols if "texture" in c]

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # ==============================================================================
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ==============================================================================
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_col = "species"
    class_counts = df[target_col].value_counts()
    n_classes = len(class_counts)

    print(f"Target Variable: '{target_col}'")
    print(f"Task Type: Multi-class Classification")
    print(f"Number of Classes: {n_classes}")

    # Imbalance / Distribution
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()

    print(f"Class Balance Summary:")
    print(f"  - Min samples per class: {min_samples}")
    print(f"  - Max samples per class: {max_samples}")
    print(f"  - Mean samples per class: {mean_samples:.4f}")
    print(f"  - Std samples per class: {class_counts.std():.4f}")

    if min_samples == max_samples:
        print("  - Status: Perfectly Balanced")
    else:
        ratio = max_samples / min_samples
        print(f"  - Imbalance Ratio (Max/Min): {ratio:.4f}")

    # ==============================================================================
    # SECTION 2: INPUT DATA ANALYSIS (TABULAR)
    # ==============================================================================
    print("\nINPUT DATA ANALYSIS (TABULAR)")
    print("-" * 30)

    print(f"Total Feature Columns: {len(feature_cols)}")

    # Missing Values
    total_nans = df[feature_cols].isna().sum().sum()
    print(
        f"Missing Values (NaNs): {total_nans} ({(total_nans / (len(df) * len(feature_cols)) * 100):.4f}%)"
    )

    # Numerical Stats Summary by Group
    # Since there are 192 columns, we summarize by group
    groups = {
        "Margin Features": margin_cols,
        "Shape Features": shape_cols,
        "Texture Features": texture_cols,
    }

    print("\nNumerical Statistics Summary by Feature Group:")
    for name, cols in groups.items():
        if not cols:
            continue
        subset = df[cols].values.flatten()
        q1 = np.percentile(subset, 25)
        q3 = np.percentile(subset, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((subset < lower_bound) | (subset > upper_bound)).sum()

        print(f"  {name} ({len(cols)} columns):")
        print(f"    - Global Mean: {np.mean(subset):.4f}")
        print(f"    - Global Std:  {np.std(subset):.4f}")
        print(f"    - Global Min:  {np.min(subset):.4f}")
        print(f"    - Global Max:  {np.max(subset):.4f}")
        print(
            f"    - Outlier Count (IQR method): {outliers} ({outliers/len(subset)*100:.2f}%)"
        )

    # ==============================================================================
    # SECTION 3: INPUT DATA ANALYSIS (IMAGE)
    # ==============================================================================
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []
    pixel_sums = 0.0
    pixel_sq_sums = 0.0
    total_pixels = 0

    # Process images
    # We construct full path: ./input/images/id.jpg
    # The metadata 'image_path' is relative to input dir, e.g., 'images/1.jpg'

    valid_images = 0

    for idx, row in df.iterrows():
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        valid_images += 1

        # Check dimensions
        if len(img.shape) == 2:
            h, w = img.shape
            c = 1
        else:
            h, w, c = img.shape

        widths.append(w)
        heights.append(h)
        channel_counts.append(c)
        aspect_ratios.append(w / h if h > 0 else 0)

        # Pixel stats (normalize to 0-1 for calculation)
        img_norm = img.astype(float) / 255.0
        pixel_sums += np.sum(img_norm)
        pixel_sq_sums += np.sum(img_norm**2)
        total_pixels += h * w * c

    if valid_images > 0:
        # Calculate global pixel stats
        global_mean = pixel_sums / total_pixels
        global_std = np.sqrt((pixel_sq_sums / total_pixels) - (global_mean**2))

        print(f"Analyzed {valid_images} images.")
        print("Dimensions:")
        print(
            f"  - Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"  - Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(
            f"  - Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
        )

        unique_channels = np.unique(channel_counts)
        print(f"Channels Distribution: {unique_channels}")

        print("Pixel Statistics (Normalized 0-1):")
        print(f"  - Global Mean: {global_mean:.4f}")
        print(f"  - Global Std:  {global_std:.4f}")
    else:
        print("No valid images found for analysis.")

    # ==============================================================================
    # SECTION 4: FEATURE/SIGNAL RELATIONSHIPS (STRUCTURED)
    # ==============================================================================
    print("\nFEATURE RELATIONSHIPS (STRUCTURED)")
    print("-" * 30)

    # 1. Redundancy (Correlation)
    # We compute correlation matrix for all features
    corr_matrix = df[feature_cols].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than 0.90
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    print(f"Redundancy Analysis:")
    print(f"  - Number of feature pairs with Correlation > 0.90: {len(to_drop)}")
    if len(to_drop) > 0:
        print(f"  - Example redundant features: {to_drop[:5]}")

    # 2. Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest):")

    le = LabelEncoder()
    y_enc = le.fit_transform(df[target_col])
    X = df[feature_cols].fillna(0)

    rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y_enc)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("  Top 5 Features:")
    for i in range(5):
        feat_name = feature_cols[indices[i]]
        score = importances[indices[i]]
        print(f"    {i+1}. {feat_name}: {score:.4f}")

    # ==============================================================================
    # SECTION 5: FEATURE/SIGNAL RELATIONSHIPS (UNSTRUCTURED)
    # ==============================================================================
    print("\nFEATURE RELATIONSHIPS (UNSTRUCTURED/META)")
    print("-" * 30)

    # Add meta features to dataframe for analysis
    # Note: widths/heights lists correspond to df rows assuming iteration order is preserved.
    # df.iterrows() preserves order.

    if len(widths) == len(df):
        df["img_width"] = widths
        df["img_height"] = heights
        df["img_aspect_ratio"] = aspect_ratios

        # Check if Aspect Ratio correlates with Species
        # We use One-Way ANOVA to see if the mean Aspect Ratio differs significantly between species

        # Group aspect ratios by species
        species_groups = [
            group["img_aspect_ratio"].values for name, group in df.groupby("species")
        ]

        # Perform ANOVA
        f_val, p_val = stats.f_oneway(*species_groups)

        print("Relationship: Image Aspect Ratio vs Species")
        print(f"  - ANOVA F-value: {f_val:.4f}")
        print(f"  - P-value: {p_val:.4e}")

        if p_val < 0.05:
            print(
                "  - Conclusion: Significant difference in Aspect Ratio across species."
            )
        else:
            print(
                "  - Conclusion: No significant difference in Aspect Ratio across species."
            )

        # Check correlation between Image Area and Texture Mean (example of cross-modal check)
        # Calculate mean texture for each row
        df["texture_mean"] = df[texture_cols].mean(axis=1)
        df["img_area"] = df["img_width"] * df["img_height"]

        corr_area_texture = df["img_area"].corr(df["texture_mean"])
        print("\nRelationship: Image Area vs Mean Texture Value")
        print(f"  - Pearson Correlation: {corr_area_texture:.4f}")

    else:
        print(
            "Skipping Meta-feature analysis due to mismatch in image processing counts."
        )


if __name__ == "__main__":
    perform_eda()
