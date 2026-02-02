import os
import random
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Configuration
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
TRAIN_PATH = METADATA_DIR / "train_metadata.parquet"
ARTICLES_PATH = INPUT_DIR / "articles.csv"
CUSTOMERS_PATH = INPUT_DIR / "customers.csv"
SEED = 42

# Set Random Seeds
random.seed(SEED)
np.random.seed(SEED)


def print_section(title):
    print(f"\n{'='*40}")
    print(f"{title.upper()}")
    print(f"{'='*40}")


def analyze_numerical(series, name):
    desc = series.describe()
    q1 = desc["25%"]
    q3 = desc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]

    print(f"Column: {name}")
    print(f"  Mean: {desc['mean']:.4f}, Std: {desc['std']:.4f}")
    print(f"  Min: {desc['min']:.4f}, Max: {desc['max']:.4f}")
    print(f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(series):.2%})")


def analyze_categorical(series, name):
    counts = series.value_counts(dropna=False)
    n_unique = len(counts)
    print(f"Column: {name}")
    print(f"  Cardinality: {n_unique}")

    if n_unique > 50:
        print(f"  > 50 Categories: Yes")

    # Rare labels (< 1%)
    total = len(series)
    rare_mask = counts / total < 0.01
    n_rare = rare_mask.sum()
    print(f"  Rare Labels (<1% freq): {n_rare} categories")


def get_image_stats(image_paths, sample_size=500):
    valid_paths = [INPUT_DIR / p for p in image_paths if isinstance(p, str)]
    # Filter for existence
    existing_paths = [p for p in valid_paths if p.exists()]

    if not existing_paths:
        return None

    sampled_paths = random.sample(existing_paths, min(len(existing_paths), sample_size))

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    pixel_means = []
    pixel_stds = []

    for p in sampled_paths:
        try:
            img = cv2.imread(str(p))
            if img is not None:
                h, w, c = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                channels.append(c)

                # Pixel stats (normalize to 0-1 range for calculation)
                img_norm = img / 255.0
                pixel_means.append(np.mean(img_norm))
                pixel_stds.append(np.std(img_norm))
        except Exception:
            continue

    return {
        "widths": np.array(widths),
        "heights": np.array(heights),
        "aspect_ratios": np.array(aspect_ratios),
        "channels": np.array(channels),
        "pixel_means": np.array(pixel_means),
        "pixel_stds": np.array(pixel_stds),
    }


def main():
    # 1. Load Data
    print_section("1. Data Loading")
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
        articles_df = pd.read_csv(ARTICLES_PATH)
        customers_df = pd.read_csv(CUSTOMERS_PATH)
        print(f"Train Transactions: {len(train_df)}")
        print(f"Articles: {len(articles_df)}")
        print(f"Customers: {len(customers_df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Target Variable Analysis (Implicit)
    print_section("2. Target Variable Analysis")
    # In recommender systems, the 'target' is the interaction.
    # We analyze the distribution of interactions per user and per item.

    # Purchases per User
    user_counts = train_df["customer_id"].value_counts()
    print("Distribution of Purchases per Customer:")
    print(f"  Mean: {user_counts.mean():.4f}, Median: {user_counts.median():.4f}")
    print(
        f"  Skewness: {user_counts.skew():.4f} (High skew indicates long-tail user activity)"
    )

    # Purchases per Article
    item_counts = train_df["article_id"].value_counts()
    print("\nDistribution of Purchases per Article:")
    print(f"  Mean: {item_counts.mean():.4f}, Median: {item_counts.median():.4f}")
    print(f"  Skewness: {item_counts.skew():.4f} (High skew indicates popularity bias)")

    # 3. Input Data Analysis
    print_section("3. Input Data Analysis")

    # --- Tabular: Transactions ---
    print("--- Transaction Data (Numerical) ---")
    analyze_numerical(train_df["price"], "price")

    print("\n--- Transaction Data (Categorical) ---")
    analyze_categorical(train_df["sales_channel_id"], "sales_channel_id")

    # --- Tabular: Customers ---
    print("\n--- Customer Data ---")
    analyze_numerical(customers_df["age"], "age")

    cat_cols_cust = ["club_member_status", "fashion_news_frequency"]
    for col in cat_cols_cust:
        analyze_categorical(customers_df[col], col)

    # Missing Values Customers
    print("\nMissing Values (Customers):")
    missing_cust = customers_df.isnull().sum()
    for col, val in missing_cust[missing_cust > 0].items():
        print(f"  {col}: {val} ({val/len(customers_df):.2%})")

    # --- Tabular: Articles ---
    print("\n--- Article Data ---")
    cat_cols_art = [
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "section_name",
    ]
    for col in cat_cols_art:
        analyze_categorical(articles_df[col], col)

    # --- Image Data ---
    print("\n--- Image Data Analysis ---")
    # Use image paths from train_df to ensure we check relevant images
    # train_df has 'image_path' column
    unique_img_paths = train_df["image_path"].dropna().unique()
    img_stats = get_image_stats(unique_img_paths, sample_size=500)

    if img_stats:
        print(f"Analyzed {len(img_stats['widths'])} sampled images.")
        print(
            f"Widths: Mean={np.mean(img_stats['widths']):.4f}, Std={np.std(img_stats['widths']):.4f}"
        )
        print(
            f"Heights: Mean={np.mean(img_stats['heights']):.4f}, Std={np.std(img_stats['heights']):.4f}"
        )
        print(f"Aspect Ratios: Mean={np.mean(img_stats['aspect_ratios']):.4f}")

        # Channels
        unique_channels, channel_counts = np.unique(
            img_stats["channels"], return_counts=True
        )
        print(f"Channel Distribution: {dict(zip(unique_channels, channel_counts))}")

        # Pixel Stats
        print(f"Global Pixel Mean (0-1): {np.mean(img_stats['pixel_means']):.4f}")
        print(f"Global Pixel Std (0-1): {np.mean(img_stats['pixel_stds']):.4f}")
    else:
        print("No images found or accessible for analysis.")

    # --- Text Data ---
    print("\n--- Text Data Analysis (Article Descriptions) ---")
    # Analyze detail_desc
    texts = articles_df["detail_desc"].dropna().astype(str).tolist()
    if texts:
        char_lens = [len(t) for t in texts]
        word_lens = [len(t.split()) for t in texts]

        # Vocabulary
        all_words = [w.lower() for t in texts for w in t.split()]
        vocab_size = len(set(all_words))

        print(f"Sample Size: {len(texts)}")
        print(f"Char Length: Mean={np.mean(char_lens):.4f}, Max={np.max(char_lens)}")
        print(f"Word Length: Mean={np.mean(word_lens):.4f}, Max={np.max(word_lens)}")
        print(f"Vocabulary Size: {vocab_size}")
    else:
        print("No text data found.")

    # 4. Feature Relationships
    print_section("4. Feature Relationships")

    # Prepare a merged dataset for correlation and RF
    # Sample transactions to save time
    sample_n = 50000
    if len(train_df) > sample_n:
        df_sample = train_df.sample(n=sample_n, random_state=SEED).copy()
    else:
        df_sample = train_df.copy()

    # Merge with customers and articles
    df_merged = df_sample.merge(customers_df, on="customer_id", how="left")
    df_merged = df_merged.merge(articles_df, on="article_id", how="left")

    # --- Structured Relationships ---
    print("--- Structured Data Relationships ---")

    # Correlation (Numerical)
    num_cols = ["price", "age"]
    corr_mat = df_merged[num_cols].corr()
    print("Correlation Matrix (Price vs Age):")
    print(corr_mat)

    # Feature Importance (Random Forest)
    # Proxy Task: Predict 'sales_channel_id' (1 vs 2)
    print("\n--- Feature Importance (Proxy Task: Predict Sales Channel) ---")

    # Select features
    features = [
        "age",
        "price",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "department_no",
        "index_group_no",
        "section_no",
        "garment_group_no",
    ]

    target = "sales_channel_id"

    # Preprocessing
    X = df_merged[features].copy()
    y = df_merged[target].copy()

    # Handle NaNs
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Drop rows where target is NaN (unlikely but safe)
    valid_indices = ~y.isna()
    X_final = X_imputed[valid_indices]
    y_final = y[valid_indices]

    if len(y_final) > 0:
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
        )
        rf.fit(X_final, y_final)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("Top 5 Features for predicting Sales Channel:")
        for f in range(min(5, len(features))):
            print(f"  {features[indices[f]]}: {importances[indices[f]]:.4f}")
    else:
        print("Insufficient data for Random Forest training.")

    # --- Unstructured Relationships ---
    print("\n--- Meta-Feature Relationships ---")

    # 1. Description Length vs Popularity (Purchase Count)
    # Aggregate purchase counts per article
    article_pop = train_df["article_id"].value_counts().reset_index()
    article_pop.columns = ["article_id", "purchase_count"]

    # Merge with description length
    articles_df["desc_len"] = articles_df["detail_desc"].astype(str).apply(len)
    pop_desc = article_pop.merge(
        articles_df[["article_id", "desc_len"]], on="article_id"
    )

    corr_desc_pop = pop_desc["purchase_count"].corr(pop_desc["desc_len"])
    print(
        f"Correlation between Description Length and Purchase Count: {corr_desc_pop:.4f}"
    )

    # 2. Image Aspect Ratio vs Popularity (using sampled stats if available)
    # We can't easily correlate individual image stats with global popularity without processing all images.
    # Instead, let's check if Price correlates with Product Group (Categorical vs Numerical)

    print("\nPrice Mean by Product Group (Top 5 Groups):")
    group_price = (
        df_merged.groupby("product_group_name")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    for group, price in group_price.items():
        print(f"  {group}: {price:.4f}")


if __name__ == "__main__":
    main()
