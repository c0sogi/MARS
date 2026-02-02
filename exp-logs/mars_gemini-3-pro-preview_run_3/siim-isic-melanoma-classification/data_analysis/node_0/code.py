import pandas as pd
import numpy as np
import os
import cv2
import warnings
from scipy.stats import iqr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE_IMAGES = (
    1000  # Number of images to sample for pixel/dim analysis to save time
)


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target(df, target_col):
    print("2. TARGET VARIABLE ANALYSIS")

    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Distribution: {counts.to_dict()}")

    # Assuming Binary Classification based on dataset description (0/1)
    if len(counts) <= 2:
        ratio_0 = counts.get(0, 0) / total
        ratio_1 = counts.get(1, 0) / total
        print(f"Class Balance Ratios: Class 0: {ratio_0:.4f}, Class 1: {ratio_1:.4f}")
        if min(ratio_0, ratio_1) < 0.1:
            print("Imbalance: High class imbalance detected.")
    else:
        # Fallback for regression or multi-class (though task says binary)
        print("Target appears to be multi-class or continuous.")
    print("-" * 30)


def analyze_images(df, path_col):
    print("3. INPUT DATA ANALYSIS (IMAGE MODALITY)")

    # Sample data for image analysis to keep runtime low
    if len(df) > SAMPLE_SIZE_IMAGES:
        df_sample = df.sample(n=SAMPLE_SIZE_IMAGES, random_state=SEED).copy()
    else:
        df_sample = df.copy()

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    pixel_means = []
    pixel_stds = []

    # To store meta-features for relationship analysis later
    meta_features = {"img_width": [], "img_height": [], "img_mean_intensity": []}
    indices = []

    print(f"Analyzing a subset of {len(df_sample)} images...")

    for idx, row in df_sample.iterrows():
        full_path = os.path.join(INPUT_DIR, row[path_col])

        if not os.path.exists(full_path):
            continue

        # Read image
        try:
            img = cv2.imread(full_path)
            if img is None:
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h != 0 else 0)
            channels.append(c)

            # Pixel stats (normalize 0-1 for calc)
            img_norm = img / 255.0
            pixel_means.append(np.mean(img_norm))
            pixel_stds.append(np.std(img_norm))

            # Store for later
            meta_features["img_width"].append(w)
            meta_features["img_height"].append(h)
            meta_features["img_mean_intensity"].append(np.mean(img_norm))
            indices.append(idx)

        except Exception:
            continue

    # Dimensions
    print("Dimensions:")
    print(
        f"  Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Channels
    unique_channels, channel_counts = np.unique(channels, return_counts=True)
    print(f"Channels: {dict(zip(unique_channels, channel_counts))} (3=RGB)")

    # Pixel Stats
    print("Pixel Stats (Normalized 0-1):")
    print(f"  Global Mean: {np.mean(pixel_means):.4f}")
    print(f"  Global Std:  {np.mean(pixel_stds):.4f}")

    # Return dataframe with image meta-features for relationship analysis
    df_meta = pd.DataFrame(meta_features, index=indices)
    return df_meta


def analyze_tabular(df, num_cols, cat_cols):
    print("3. INPUT DATA ANALYSIS (TABULAR MODALITY)")

    # Numerical
    if num_cols:
        print("Numerical Columns:")
        for col in num_cols:
            series = df[col].dropna()
            if len(series) == 0:
                print(f"  {col}: All values missing.")
                continue

            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()

            # IQR Outliers
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr_val = q3 - q1
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val
            outliers = series[(series < lower_bound) | (series > upper_bound)]

            print(
                f"  {col}: Mean={mean_val:.4f}, Std={std_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}, Outliers={len(outliers)}"
            )

    # Categorical
    if cat_cols:
        print("Categorical Columns:")
        for col in cat_cols:
            series = df[col].astype(str)
            unique_counts = series.nunique()
            print(f"  {col}: Cardinality={unique_counts}")

            if unique_counts > 50:
                print(f"    Flag: High cardinality (>50 categories).")

            # Rare labels
            value_counts = series.value_counts(normalize=True)
            rare_labels = value_counts[value_counts < 0.01].index.tolist()
            if rare_labels:
                print(
                    f"    Rare labels (<1%): {len(rare_labels)} found (e.g., {rare_labels[:3]})"
                )

    # Missing Values
    print("Missing Values:")
    missing = df[num_cols + cat_cols].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in num_cols + cat_cols:
        print(f"  {col}: {missing[col]} ({missing_pct[col]:.4f}%)")
    print("-" * 30)


def analyze_relationships(df, df_img_stats, target_col, num_cols, cat_cols):
    print("4. FEATURE/SIGNAL RELATIONSHIPS")

    # Merge original df with image stats (inner join on index to keep only sampled rows for image stats)
    # We will use the full df for tabular-only analysis, and the merged one for image-target analysis.

    # 1. Structured Relationships (Full Data)
    print("Structured (Tabular) Relationships:")

    # Prepare data for correlation/importance
    # Drop rows where target is NaN (shouldn't happen in train, but safety first)
    df_clean = df.dropna(subset=[target_col]).copy()

    # Numerical Correlation (Pearson)
    if num_cols:
        correlations = {}
        for col in num_cols:
            # Simple fill for correlation calc
            series = df_clean[col].fillna(df_clean[col].mean())
            corr = np.corrcoef(series, df_clean[target_col])[0, 1]
            correlations[col] = corr

        print("  Numerical Correlations with Target:")
        for col, val in correlations.items():
            print(f"    {col}: {val:.4f}")

        # Redundancy
        if len(num_cols) > 1:
            print("  Redundancy (Collinear Pairs > 0.90):")
            found_redundancy = False
            # Check pairs
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    s1 = df_clean[c1].fillna(0)
                    s2 = df_clean[c2].fillna(0)
                    corr = np.corrcoef(s1, s2)[0, 1]
                    if abs(corr) > 0.9:
                        print(f"    {c1} - {c2}: {corr:.4f}")
                        found_redundancy = True
            if not found_redundancy:
                print("    None found.")

    # Categorical Mutual Information
    if cat_cols:
        print("  Categorical Mutual Information with Target:")
        # Encode
        le = LabelEncoder()
        for col in cat_cols:
            # Fill NaNs with 'missing'
            filled = df_clean[col].fillna("missing").astype(str)
            encoded = le.fit_transform(filled)
            mi = mutual_info_classif(
                encoded.reshape(-1, 1),
                df_clean[target_col],
                discrete_features=True,
                random_state=SEED,
            )
            print(f"    {col}: {mi[0]:.4f}")

    # Feature Importance (Random Forest)
    print("  Feature Importance (Top 5):")
    # Preprocessing for RF
    X = df_clean[num_cols + cat_cols].copy()
    y = df_clean[target_col]

    # Impute Num
    imputer_num = SimpleImputer(strategy="mean")
    if num_cols:
        X[num_cols] = imputer_num.fit_transform(X[num_cols])

    # Encode Cat
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("missing").astype(str))
        le_dict[col] = le

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = num_cols + cat_cols

    for i in range(min(5, len(features))):
        print(f"    {features[indices[i]]}: {importances[indices[i]]:.4f}")

    # 2. Unstructured (Meta-Feature) Relationships
    print("Unstructured (Image Meta-Feature) Relationships:")
    # Join target to image stats
    df_img_merged = df_img_stats.join(df[[target_col]], how="inner")

    if not df_img_merged.empty:
        img_cols = ["img_width", "img_height", "img_mean_intensity"]
        for col in img_cols:
            corr = df_img_merged[col].corr(df_img_merged[target_col])
            print(f"  Correlation {col} vs Target: {corr:.4f}")

        # Check if larger images are associated with malignancy
        # Group by target
        grp = df_img_merged.groupby(target_col)["img_mean_intensity"].mean()
        print(f"  Mean Pixel Intensity by Class: {grp.to_dict()}")
    else:
        print("  Insufficient image data sampled for relationship analysis.")

    print("-" * 30)


def main():
    set_seed(SEED)

    print("1. DATA INTEGRITY")
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded training set from {METADATA_PATH}. Shape: {df.shape}")
    print("-" * 30)

    # Define columns based on dataset description
    target_col = "target"
    # Tabular features
    num_cols = ["age_approx"]
    cat_cols = ["sex", "anatom_site_general_challenge"]
    # Image path
    path_col = "file_path"

    # 2. Target Analysis
    analyze_target(df, target_col)

    # 3. Image Analysis (generates meta features for step 4)
    df_img_stats = analyze_images(df, path_col)
    print("-" * 30)

    # 3. Tabular Analysis
    analyze_tabular(df, num_cols, cat_cols)

    # 4. Relationships
    analyze_relationships(df, df_img_stats, target_col, num_cols, cat_cols)


if __name__ == "__main__":
    main()
