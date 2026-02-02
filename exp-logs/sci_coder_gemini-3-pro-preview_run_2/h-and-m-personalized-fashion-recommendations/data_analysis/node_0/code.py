import os
import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis, pearsonr
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
INPUT_DIR = Path("./input")
META_DIR = Path("./metadata")
IMAGE_SAMPLE_SIZE = 2000  # Number of images to analyze
RF_SAMPLE_SIZE = 50000  # Number of samples for Feature Importance


def print_header(title):
    print(f"\n{title.upper()}")
    print("=" * len(title))


def analyze_target(df):
    print_header("2. Target Variable Analysis")
    # Target: article_id (Multi-class classification context)
    # We analyze the frequency distribution of classes.

    target_counts = df["article_id"].value_counts()
    n_classes = len(target_counts)
    n_samples = len(df)

    print(f"Target Variable: article_id")
    print(f"Total Unique Classes (Articles): {n_classes}")
    print(f"Total Samples (Transactions): {n_samples}")

    # Imbalance / Skew
    # Calculate share of top 1% items
    top_1_percent_n = int(n_classes * 0.01)
    top_1_percent_vol = target_counts.iloc[:top_1_percent_n].sum()
    ratio_top_1 = top_1_percent_vol / n_samples

    print(f"Class Imbalance Analysis:")
    print(f"  - Top 1% of Articles account for {ratio_top_1:.4%} of all purchases.")
    print(
        f"  - Most frequent article ID: {target_counts.index[0]} (Count: {target_counts.iloc[0]})"
    )
    print(
        f"  - Least frequent article ID: {target_counts.index[-1]} (Count: {target_counts.iloc[-1]})"
    )
    print(f"  - Skewness of Class Frequencies: {skew(target_counts):.4f}")

    return target_counts


def analyze_tabular(df):
    print_header("3. Input Data Analysis (Tabular)")

    # Numerical Columns
    # price is in transactions, age is in customers
    num_cols = ["price", "age"]
    print("--- Numerical Data ---")
    for col in num_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        print(f"Feature: {col}")
        print(f"  - Mean: {series.mean():.4f}, Std: {series.std():.4f}")
        print(f"  - Min: {series.min():.4f}, Max: {series.max():.4f}")
        print(
            f"  - Outlier Count (IQR Method): {len(outliers)} ({len(outliers)/len(series):.2%})"
        )

    # Categorical Columns
    # Selected relevant categoricals
    cat_cols = [
        "sales_channel_id",
        "index_group_name",
        "section_name",
        "club_member_status",
        "colour_group_name",
    ]
    print("\n--- Categorical Data ---")
    for col in cat_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        n_unique = series.nunique()
        print(f"Feature: {col}")
        print(f"  - Cardinality: {n_unique}")

        if n_unique > 50:
            print(f"  - High Cardinality Flag: Yes (>50 categories)")

        # Rare labels check
        counts = series.value_counts(normalize=True)
        rare_labels = counts[counts < 0.01]
        if not rare_labels.empty:
            print(f"  - Rare Labels (<1% freq): {len(rare_labels)} categories")
        else:
            print(f"  - Rare Labels: None")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  - {col}: {count} NaNs ({count/len(df):.4%})")
    else:
        print("  - No missing values found in the analyzed columns.")


def analyze_images(df_train):
    print_header("3. Input Data Analysis (Image)")

    # Sample unique image paths
    unique_paths = df_train["image_path"].dropna().unique()

    if len(unique_paths) == 0:
        print("No image paths found.")
        return None

    sample_paths = np.random.choice(
        unique_paths, size=min(len(unique_paths), IMAGE_SAMPLE_SIZE), replace=False
    )

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []
    pixel_means = []
    pixel_stds = []

    print(f"Analyzing {len(sample_paths)} sampled images...")

    valid_count = 0
    for rel_path in sample_paths:
        full_path = INPUT_DIR / rel_path
        if not full_path.exists():
            continue

        try:
            # Read image
            img = cv2.imread(str(full_path))
            if img is None:
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            channel_counts.append(c)

            # Pixel stats (normalize 0-255 to 0-1 implicitly by just taking mean of values)
            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))

            valid_count += 1
        except Exception:
            continue

    if valid_count == 0:
        print("Could not load any images.")
        return None

    print(f"Successfully analyzed {valid_count} images.")

    # Dimensions
    print("--- Dimensions ---")
    print(f"  - Width: Mean={np.mean(widths):.2f}, Std={np.std(widths):.2f}")
    print(f"  - Height: Mean={np.mean(heights):.2f}, Std={np.std(heights):.2f}")
    print(
        f"  - Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Channels
    print("--- Channels ---")
    unique_channels, counts = np.unique(channel_counts, return_counts=True)
    print(f"  - Channel Distribution: {dict(zip(unique_channels, counts))}")

    # Pixel Stats
    print("--- Pixel Stats (0-255) ---")
    print(f"  - Global Pixel Mean: {np.mean(pixel_means):.4f}")
    print(f"  - Global Pixel Std: {np.mean(pixel_stds):.4f}")

    # Return stats for meta-feature analysis
    return pd.DataFrame(
        {
            "image_path": sample_paths[
                : len(aspect_ratios)
            ],  # Alignment assumption: valid_count matches logic
            "aspect_ratio": aspect_ratios,
            "pixel_mean": pixel_means,
        }
    )


def analyze_relationships(df, image_stats, article_counts):
    print_header("4. Feature/Signal Relationships")

    # --- Structured Relationships ---
    print("--- Structured (Tabular) Relationships ---")

    # Correlation (Numerical)
    # We use a sample to save time if dataset is huge
    sample_df = df.sample(n=min(len(df), 100000), random_state=SEED)

    if "age" in sample_df.columns and "price" in sample_df.columns:
        corr, _ = pearsonr(
            sample_df["age"].fillna(sample_df["age"].mean()), sample_df["price"]
        )
        print(f"  - Correlation (Age vs Price): {corr:.4f}")

    # Redundancy (Collinear pairs)
    # Check article numerical codes
    article_num_cols = [
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "department_no",
        "index_group_no",
        "section_no",
        "garment_group_no",
    ]
    # Filter to cols that exist
    article_num_cols = [c for c in article_num_cols if c in df.columns]

    # Calculate correlation matrix on unique articles to avoid weighting by popularity
    unique_articles = df.drop_duplicates("article_id")[article_num_cols]
    if not unique_articles.empty:
        corr_matrix = unique_articles.corr().abs()
        # Select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

        if high_corr:
            print(f"  - Redundant Features (Corr > 0.90):")
            for col in high_corr:
                correlated_with = upper.index[upper[col] > 0.90].tolist()
                print(f"    * {col} correlates with {correlated_with}")
        else:
            print(f"  - No highly collinear pairs found (> 0.90) among article codes.")

    # Importance (Random Forest)
    # Task: Predict Log(Popularity) of an article based on its metadata
    # This tells us which article features drive sales.
    print("\n--- Feature Importance (Proxy Task: Predict Article Popularity) ---")

    # Prepare data
    # 1. Get popularity per article
    pop_df = article_counts.reset_index()
    pop_df.columns = ["article_id", "count"]
    pop_df["log_count"] = np.log1p(pop_df["count"])

    # 2. Get article features
    # We need to access the original articles dataframe or extract from merged df
    # Extracting from merged df (drop duplicates)
    article_features = [
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "department_no",
        "index_group_no",
        "section_no",
        "garment_group_no",
    ]

    # Ensure columns exist
    article_features = [c for c in article_features if c in df.columns]

    features_df = df[["article_id"] + article_features].drop_duplicates("article_id")

    # Merge
    rf_data = pd.merge(pop_df, features_df, on="article_id", how="inner")

    # Train RF
    if not rf_data.empty:
        X = rf_data[article_features].fillna(-1)  # Simple imputation
        y = rf_data["log_count"]

        rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
        )
        rf.fit(X, y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("  Top 5 Features driving Article Popularity:")
        for i in range(min(5, len(indices))):
            print(
                f"    {i+1}. {article_features[indices[i]]} (Importance: {importances[indices[i]]:.4f})"
            )

    # --- Unstructured Relationships ---
    print("\n--- Unstructured (Meta-Feature) Relationships ---")
    if image_stats is not None and not image_stats.empty:
        # Check correlation between Image Aspect Ratio and Popularity (count)
        # We need to map image_path back to article_id or just merge on image_path if unique

        # In merged df, we have article_id and image_path.
        # Let's get mapping
        path_map = df[["article_id", "image_path"]].drop_duplicates()

        # Merge stats with article_id
        meta_rel = pd.merge(image_stats, path_map, on="image_path", how="inner")
        # Merge with popularity
        meta_rel = pd.merge(meta_rel, pop_df, on="article_id", how="inner")

        if len(meta_rel) > 10:
            corr_ar, _ = pearsonr(meta_rel["aspect_ratio"], meta_rel["log_count"])
            corr_bright, _ = pearsonr(meta_rel["pixel_mean"], meta_rel["log_count"])

            print(
                f"  - Correlation (Image Aspect Ratio vs Log-Popularity): {corr_ar:.4f}"
            )
            print(
                f"  - Correlation (Image Brightness vs Log-Popularity): {corr_bright:.4f}"
            )

            if abs(corr_ar) < 0.1:
                print(
                    "    -> Little to no linear relationship between image shape and popularity."
                )
        else:
            print("  - Not enough matched image samples for correlation analysis.")
    else:
        print(
            "  - Image analysis skipped or failed, cannot compute meta-relationships."
        )


def main():
    # 1. Data Loading
    print("Loading data...")
    try:
        train_df = pd.read_parquet(META_DIR / "train.parquet")
        articles_df = pd.read_csv(INPUT_DIR / "articles.csv")
        customers_df = pd.read_csv(INPUT_DIR / "customers.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 2. Preprocessing / Merging
    # To save memory, we might drop some text columns from articles/customers before merging
    # but for EDA we want access to them. We rely on the machine's 220GB RAM.

    # Convert t_dat to datetime
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])

    # Merge
    # Left join transactions with articles and customers
    # Note: article_id in train.parquet is string (padded).
    # article_id in articles.csv is int64. We must convert articles.csv id to string padded.
    articles_df["article_id"] = articles_df["article_id"].astype(str).str.zfill(10)

    # customer_id is string in both.

    merged_df = train_df.merge(articles_df, on="article_id", how="left")
    merged_df = merged_df.merge(customers_df, on="customer_id", how="left")

    # 3. Target Analysis
    article_counts = analyze_target(merged_df)

    # 4. Tabular Analysis
    analyze_tabular(merged_df)

    # 5. Image Analysis
    # We pass the unique image paths from the training set
    image_stats = analyze_images(merged_df)

    # 6. Relationships
    analyze_relationships(merged_df, image_stats, article_counts)

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
