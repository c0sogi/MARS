import os
import numpy as np
import pandas as pd
import cv2
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis


# 1. Setup and Configuration
def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)
    warnings.filterwarnings("ignore")

    INPUT_ROOT = "./input"
    METADATA_PATH = "./metadata/train.csv"

    print("Loading training metadata...")
    try:
        df = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {METADATA_PATH}")
        return

    # Define column types based on dataset description
    target_col = "target"
    id_cols = ["image_name", "patient_id", "file_path"]
    # 'diagnosis' and 'benign_malignant' are derived from/related to target and not available in test
    # We will analyze them but exclude them from predictive feature importance
    train_only_cols = ["diagnosis", "benign_malignant"]

    numerical_cols = ["age_approx"]
    categorical_cols = ["sex", "anatom_site_general_challenge"]

    print("=" * 40)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 40)

    # 2. Target Variable Analysis
    print("\n[TARGET VARIABLE ANALYSIS]")
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        ratios = df[target_col].value_counts(normalize=True)
        print(f"Target Variable: '{target_col}'")
        print(f"Distribution: {counts.to_dict()}")
        print(f"Ratios: {ratios.to_dict()}")

        # Binary Classification Imbalance
        if len(counts) == 2:
            minority_class = counts.idxmin()
            majority_class = counts.idxmax()
            imbalance_ratio = counts[majority_class] / counts[minority_class]
            print(f"Class Imbalance Ratio (Maj:Min): {imbalance_ratio:.4f}:1")
    else:
        print(f"Target column '{target_col}' not found.")

    # 3. Input Data Analysis (Tabular)
    print("\n[TABULAR DATA ANALYSIS]")

    # Missing Values
    print("Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        for col, val in missing.items():
            print(f"  {col}: {val} ({val/len(df)*100:.2f}%)")
    else:
        print("  None")

    # Numerical Analysis
    print("\nNumerical Features:")
    for col in numerical_cols:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                print(f"  {col}: All values missing")
                continue

            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()

            # Outliers via IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = series[
                ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))
            ]

            print(
                f"  {col}: Mean={mean_val:.4f}, Std={std_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}"
            )
            print(
                f"  {col} Outliers (IQR method): {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)"
            )

    # Categorical Analysis
    print("\nCategorical Features:")
    for col in categorical_cols:
        if col in df.columns:
            series = df[col].astype(str)  # Handle NaNs as string for counting
            n_unique = series.nunique()
            print(f"  {col}: Cardinality={n_unique}")

            if n_unique > 50:
                print(f"    Flag: High cardinality (>50 categories)")

            # Rare labels
            counts = series.value_counts(normalize=True)
            rare = counts[counts < 0.01]
            if not rare.empty:
                print(
                    f"    Rare labels (<1%): {len(rare)} categories (e.g., {list(rare.index[:3])})"
                )

    # 4. Input Data Analysis (Image)
    print("\n[IMAGE DATA ANALYSIS]")
    # Sampling for efficiency
    SAMPLE_SIZE = 2000
    if len(df) > SAMPLE_SIZE:
        # Stratified sample to ensure we see both classes
        # Handle cases where stratify group is too small
        try:
            df_sample = df.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), int(SAMPLE_SIZE * (len(x) / len(df)))), random_state=42
                )
            )
        except:
            df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df.copy()

    print(f"Analyzing a sample of {len(df_sample)} images...")

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    valid_images = 0

    for idx, row in df_sample.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_ROOT, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            # Read image
            img = cv2.imread(full_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channel_counts.append(c)

            # Pixel stats (normalize to 0-1 for calculation stability)
            img_norm = img / 255.0
            pixel_sum += np.sum(img_norm, axis=(0, 1))
            pixel_sq_sum += np.sum(img_norm**2, axis=(0, 1))
            pixel_count += h * w

            valid_images += 1

        except Exception as e:
            continue

    if valid_images > 0:
        # Dimensions
        print(f"Dimensions (Sample N={valid_images}):")
        print(
            f"  Width: Mean={np.mean(widths):.2f}, Std={np.std(widths):.2f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"  Height: Mean={np.mean(heights):.2f}, Std={np.std(heights):.2f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(
            f"  Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
        )

        # Channels
        unique_channels = np.unique(channel_counts)
        print(
            f"  Channels Distribution: {dict(zip(*np.unique(channel_counts, return_counts=True)))}"
        )

        # Pixel Stats
        global_mean = pixel_sum / pixel_count
        global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))
        print(f"Pixel Statistics (RGB, normalized 0-1):")
        print(f"  Mean: {global_mean}")
        print(f"  Std : {global_std}")

        # Add metadata to sample dataframe for later correlation analysis
        df_sample["img_width"] = widths
        df_sample["img_height"] = heights
        df_sample["img_aspect_ratio"] = aspect_ratios
    else:
        print("No valid images found in sample.")

    # 5. Feature/Signal Relationships
    print("\n[FEATURE RELATIONSHIPS]")

    # A. Structured Relationships
    print("Structured Data Relationships:")

    # Prepare data for correlation/importance
    # Select features available at test time
    features = numerical_cols + categorical_cols

    df_analysis = df[features + [target_col]].copy()

    # Encode Categoricals
    le_dict = {}
    for col in categorical_cols:
        df_analysis[col] = df_analysis[col].fillna("Missing")
        le = LabelEncoder()
        df_analysis[col] = le.fit_transform(df_analysis[col].astype(str))
        le_dict[col] = le

    # Impute Numericals
    imputer = SimpleImputer(strategy="mean")
    df_analysis[numerical_cols] = imputer.fit_transform(df_analysis[numerical_cols])

    # Correlation
    corr_matrix = df_analysis.corr()
    target_corr = (
        corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    )
    print("Correlation with Target (Top 5):")
    for idx, val in target_corr.head(5).items():
        print(f"  {idx}: {val:.4f}")

    # Redundancy (Collinearity)
    print("High Collinearity Pairs (>0.90):")
    high_corr_pairs = []
    feature_subset = df_analysis.drop(columns=[target_col])
    cols = feature_subset.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = feature_subset[cols[i]].corr(feature_subset[cols[j]])
            if abs(c) > 0.90:
                high_corr_pairs.append((cols[i], cols[j], c))

    if high_corr_pairs:
        for c1, c2, val in high_corr_pairs:
            print(f"  {c1} - {c2}: {val:.4f}")
    else:
        print("  None found")

    # Feature Importance (Random Forest)
    print("Feature Importance (Random Forest):")
    X = df_analysis.drop(columns=[target_col])
    y = df_analysis[target_col]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    for idx, val in importances.head(5).items():
        print(f"  {idx}: {val:.4f}")

    # B. Unstructured (Meta-Feature) Relationships
    print("\nMeta-Feature Relationships (Sampled Data):")
    if valid_images > 0:
        # Check correlation of image stats with target
        meta_features = ["img_width", "img_height", "img_aspect_ratio"]
        df_meta_corr = df_sample[meta_features + [target_col]].dropna()

        if not df_meta_corr.empty:
            meta_corr = df_meta_corr.corr()[target_col].drop(target_col)
            print("Correlation of Image Properties with Target:")
            for idx, val in meta_corr.items():
                print(f"  {idx}: {val:.4f}")

            # Check if larger images are associated with malignancy
            mal_mean_size = df_sample[df_sample[target_col] == 1]["img_width"].mean()
            ben_mean_size = df_sample[df_sample[target_col] == 0]["img_width"].mean()
            print(f"  Mean Width (Malignant): {mal_mean_size:.2f}")
            print(f"  Mean Width (Benign): {ben_mean_size:.2f}")
    else:
        print("  Skipped (No image data processed)")


if __name__ == "__main__":
    main()
