import pandas as pd
import numpy as np
import os
import cv2
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    # Target is article_id
    target_counts = df["article_id"].value_counts()
    n_classes = len(target_counts)
    total_samples = len(df)

    print(f"Target Variable: article_id (Multi-class Classification)")
    print(f"Total Transactions: {total_samples}")
    print(f"Total Unique Classes (Articles Bought): {n_classes}")

    # Distribution stats
    most_freq = target_counts.iloc[0]
    least_freq = target_counts.iloc[-1]
    median_freq = target_counts.median()

    print(
        f"Most Frequent Class Count: {most_freq} ({(most_freq/total_samples)*100:.4f}%)"
    )
    print(f"Least Frequent Class Count: {least_freq}")
    print(f"Median Class Count: {median_freq:.0f}")

    # Imbalance
    imbalance_ratio = most_freq / median_freq if median_freq > 0 else most_freq
    print(f"Imbalance Ratio (Most/Median): {imbalance_ratio:.4f}")

    # Skewness of the class distribution
    skewness = target_counts.skew()
    kurtosis = target_counts.kurt()
    print(f"Class Distribution Skewness: {skewness:.4f}")
    print(f"Class Distribution Kurtosis: {kurtosis:.4f}")
    print("\n")


def analyze_tabular(name, df):
    print(f"INPUT DATA ANALYSIS: TABULAR ({name})")
    print("-" * 30)

    # Numerical
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(f"Numerical Columns: {list(num_cols)}")
        stats = df[num_cols].describe().T
        stats["IQR"] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)

        # Outlier counts (1.5 IQR rule)
        outliers = {}
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = ((df[col] < lower) | (df[col] > upper)).sum()
            outliers[col] = count

        print(
            f"{'Column':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers':<10}"
        )
        for col in num_cols:
            row = stats.loc[col]
            print(
                f"{col:<25} {row['mean']:.4f}     {row['std']:.4f}     {row['min']:.4f}     {row['max']:.4f}     {outliers[col]:<10}"
            )
    else:
        print("No Numerical Columns.")

    print("-" * 10)

    # Categorical
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        print(f"Categorical Columns: {list(cat_cols)}")
        print(f"{'Column':<25} {'Cardinality':<12} {'Rare Labels (<1%)':<10}")
        for col in cat_cols:
            cardinality = df[col].nunique()
            # Check for rare labels (sample if too large)
            if len(df) > 100000:
                vc = (
                    df[col].value_counts(normalize=True).head(100)
                )  # Check top 100 for efficiency
            else:
                vc = df[col].value_counts(normalize=True)

            # This is an approximation for very high cardinality columns to save time
            rare_flag = "Yes" if (vc < 0.01).any() else "No"
            print(f"{col:<25} {cardinality:<12} {rare_flag:<10}")
    else:
        print("No Categorical Columns.")

    print("-" * 10)

    # Missing Values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("Missing Values:")
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df)*100:.4f}%)")
    else:
        print("No Missing Values.")
    print("\n")


def analyze_images(df_train, input_dir):
    print("INPUT DATA ANALYSIS: IMAGE")
    print("=" * 30)

    # Sample images
    sample_size = 1000
    if "image_path" not in df_train.columns:
        print("No image_path column found.")
        return

    # Filter for unique images to avoid processing same image multiple times
    unique_paths = df_train["image_path"].dropna().unique()

    if len(unique_paths) > sample_size:
        sampled_paths = np.random.choice(unique_paths, sample_size, replace=False)
    else:
        sampled_paths = unique_paths

    widths = []
    heights = []
    r_means, g_means, b_means = [], [], []
    r_stds, g_stds, b_stds = [], [], []
    channels_count = []

    missing_files = 0

    for rel_path in sampled_paths:
        full_path = os.path.join(input_dir, rel_path)
        if not os.path.exists(full_path):
            missing_files += 1
            continue

        try:
            # Read as is (to check channels)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                missing_files += 1
                continue

            h, w = img.shape[:2]
            c = 1 if len(img.shape) == 2 else img.shape[2]

            widths.append(w)
            heights.append(h)
            channels_count.append(c)

            # Compute stats (convert to RGB for consistency if color)
            if c == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mean, std = cv2.meanStdDev(img_rgb)
                r_means.append(mean[0][0])
                g_means.append(mean[1][0])
                b_means.append(mean[2][0])
                r_stds.append(std[0][0])
                g_stds.append(std[1][0])
                b_stds.append(std[2][0])
            elif c == 1:
                mean, std = cv2.meanStdDev(img)
                r_means.append(mean[0][0])  # Treat as R for storage
                r_stds.append(std[0][0])

        except Exception:
            missing_files += 1

    print(
        f"Analyzed {len(widths)} images (Sampled from {len(unique_paths)} unique paths). Missing/Error: {missing_files}"
    )

    if len(widths) > 0:
        widths = np.array(widths)
        heights = np.array(heights)
        aspect_ratios = widths / heights

        print(
            f"Dimensions (Width): Mean={np.mean(widths):.2f}, Std={np.std(widths):.2f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"Dimensions (Height): Mean={np.mean(heights):.2f}, Std={np.std(heights):.2f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(
            f"Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
        )

        # Channel distribution
        c_counts = Counter(channels_count)
        print(f"Channel Counts: {dict(c_counts)}")

        # Pixel Stats
        print(
            f"Pixel Values (Global Mean): R={np.mean(r_means):.4f}, G={np.mean(g_means):.4f}, B={np.mean(b_means):.4f}"
        )
        print(
            f"Pixel Values (Global Std): R={np.mean(r_stds):.4f}, G={np.mean(g_stds):.4f}, B={np.mean(b_stds):.4f}"
        )
    print("\n")


def analyze_text(df_articles):
    print("INPUT DATA ANALYSIS: TEXT")
    print("=" * 30)

    col = "detail_desc"
    if col not in df_articles.columns:
        print(f"Column {col} not found.")
        return

    texts = df_articles[col].astype(str).fillna("")

    # Lengths
    char_lens = texts.apply(len)
    word_lens = texts.apply(lambda x: len(x.split()))

    print(f"Text Column: {col}")
    print(
        f"Sequence Length (Chars): Mean={char_lens.mean():.4f}, Std={char_lens.std():.4f}, Max={char_lens.max()}"
    )
    print(
        f"Sequence Length (Words): Mean={word_lens.mean():.4f}, Std={word_lens.std():.4f}, Max={word_lens.max()}"
    )

    # Vocabulary
    all_words = " ".join(
        texts.sample(min(10000, len(texts)), random_state=42).tolist()
    ).split()
    vocab_size = len(set(all_words))
    print(f"Estimated Vocabulary Size (from sample): {vocab_size}")
    print("\n")


def analyze_relationships(df_train, df_articles, df_customers):
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # 1. Structured Correlations
    # Merge a sample of train with customers and articles to check correlations
    # Sampling is necessary for memory/speed
    sample_train = df_train.sample(n=min(100000, len(df_train)), random_state=42)

    # Prepare merge
    # Select numeric cols from customers
    cust_cols = ["customer_id", "age"]
    # Select numeric cols from articles
    art_cols = [
        "article_id",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "section_no",
    ]

    merged = sample_train.merge(df_customers[cust_cols], on="customer_id", how="left")
    merged = merged.merge(df_articles[art_cols], on="article_id", how="left")

    # Numeric correlations
    num_cols = ["price", "age", "product_type_no", "section_no"]
    corr_matrix = merged[num_cols].corr()

    print("Structured Data Correlations (Pearson):")
    print(corr_matrix)

    # Redundancy Check
    high_corr = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

    if high_corr:
        print(f"Redundant Features (>0.9): {high_corr}")
    else:
        print("No highly correlated numerical pairs (>0.9) found.")

    print("-" * 20)

    # 2. Feature Importance (Proxy Task: Predict Item Popularity)
    print("Feature Importance (Target: Item Popularity/Log Sales):")

    # Calculate popularity from full train set
    item_popularity = df_train["article_id"].value_counts().reset_index()
    item_popularity.columns = ["article_id", "purchase_count"]
    item_popularity["log_popularity"] = np.log1p(item_popularity["purchase_count"])

    # Merge with article features
    # Use features that might describe the item visually/categorically
    features = [
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "department_no",
        "index_group_no",
        "section_no",
        "garment_group_no",
    ]

    data_rf = item_popularity.merge(
        df_articles[["article_id"] + features], on="article_id", how="left"
    )
    data_rf = data_rf.dropna()

    X = data_rf[features]
    y = data_rf["log_popularity"]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.DataFrame(
        {"feature": features, "importance": rf.feature_importances_}
    )
    importances = importances.sort_values("importance", ascending=False)

    print("Top 5 Features driving Item Popularity:")
    print(importances.head(5))

    print("-" * 20)

    # 3. Unstructured Meta-Feature Relationships
    print("Unstructured Meta-Feature Relationships:")

    # Text Length vs Popularity
    if "detail_desc" in df_articles.columns:
        df_articles["desc_len"] = df_articles["detail_desc"].astype(str).apply(len)
        meta_corr_df = data_rf.merge(
            df_articles[["article_id", "desc_len"]], on="article_id", how="left"
        )
        corr_text = meta_corr_df["log_popularity"].corr(meta_corr_df["desc_len"])
        print(f"Correlation (Description Length vs Log Popularity): {corr_text:.4f}")

    # Note: Image size vs Popularity would require checking file sizes for all items,
    # which is slow. We skip that for speed, as we already did image sampling.


def main():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Load Data
    print("Loading Data...")
    # Use optimized types to save memory
    train_df = pd.read_csv(
        os.path.join(METADATA_DIR, "train.csv"),
        dtype={"article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
    )

    articles_df = pd.read_csv(
        os.path.join(INPUT_DIR, "articles.csv"), dtype={"article_id": "int32"}
    )
    customers_df = pd.read_csv(os.path.join(INPUT_DIR, "customers.csv"))

    print(f"Train Shape: {train_df.shape}")
    print(f"Articles Shape: {articles_df.shape}")
    print(f"Customers Shape: {customers_df.shape}")
    print("\n")

    # 1. Target Analysis
    analyze_target(train_df)

    # 2. Tabular Analysis
    analyze_tabular("Transactions (Train)", train_df[["price", "sales_channel_id"]])
    analyze_tabular("Customers", customers_df)
    # Select a subset of interesting article columns to keep output concise
    art_cols_to_check = [
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "section_name",
    ]
    analyze_tabular("Articles Metadata", articles_df[art_cols_to_check])

    # 3. Image Analysis
    analyze_images(train_df, INPUT_DIR)

    # 4. Text Analysis
    analyze_text(articles_df)

    # 5. Relationships
    analyze_relationships(train_df, articles_df, customers_df)


if __name__ == "__main__":
    main()
