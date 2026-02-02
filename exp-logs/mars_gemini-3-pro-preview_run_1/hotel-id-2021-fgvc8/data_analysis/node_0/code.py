import os
import pandas as pd
import numpy as np
import cv2
import random
from datetime import datetime

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dim stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"{METADATA_PATH} not found.")
    df = pd.read_csv(METADATA_PATH)
    return df


def analyze_target(df):
    """
    Analyzes the target variable 'hotel_id'.
    """
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_col = "hotel_id"
    class_counts = df[target_col].value_counts()

    n_classes = len(class_counts)
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()

    # Imbalance ratio (Max / Min)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else np.inf

    print(f"Target Variable: {target_col}")
    print(f"Task Type: Classification")
    print(f"Total Samples: {len(df)}")
    print(f"Number of Classes (Unique Hotels): {n_classes}")
    print(f"Class Distribution Stats:")
    print(f"  Min Samples per Class: {min_samples}")
    print(f"  Max Samples per Class: {max_samples}")
    print(f"  Mean Samples per Class: {mean_samples:.4f}")
    print(f"  Median Samples per Class: {median_samples:.4f}")
    print(f"  Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Top 5 classes
    print("Top 5 Most Frequent Classes:")
    for cls, count in class_counts.head(5).items():
        print(f"  Hotel ID {cls}: {count} images")

    # Singleton check (already handled in metadata generation, but good to verify)
    singletons = (class_counts == 1).sum()
    print(f"Classes with only 1 sample: {singletons}")
    print("")


def analyze_images(df):
    """
    Analyzes image dimensions, channels, and pixel stats on a subset.
    """
    print("INPUT DATA ANALYSIS (IMAGE MODALITY)")
    print("-" * 30)

    # Sample data
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # For pixel stats (Welford's algorithm or simple accumulation for mean/std)
    # We will use simple accumulation for estimation
    # Accumulators for Mean/Std
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    pixel_count = 0

    print(f"Analyzing a random sample of {len(sample_df)} images...")

    valid_images = 0

    for _, row in sample_df.iterrows():
        # Construct full path
        # file_path in metadata is relative to INPUT_DIR
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channels_list.append(c)

            # Pixel stats
            # Normalize to 0-1 for calculation
            img_norm = img / 255.0
            channel_sum += np.sum(img_norm, axis=(0, 1))
            channel_sq_sum += np.sum(img_norm**2, axis=(0, 1))
            pixel_count += h * w

            valid_images += 1

        except Exception:
            continue

    if valid_images == 0:
        print("Error: No valid images found in sample.")
        return

    # Dimension Stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print("Image Dimensions:")
    print(
        f"  Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )

    print("Aspect Ratios (Width / Height):")
    print(f"  Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}")

    # Channel Stats
    unique_channels = np.unique(channels_list)
    print(f"Channel Distribution: {unique_channels}")

    # Pixel Stats
    # Global Mean = Sum / N
    # Global Std = Sqrt( (SumSq / N) - (Mean^2) )
    global_mean = channel_sum / pixel_count
    global_std = np.sqrt((channel_sq_sum / pixel_count) - (global_mean**2))

    print("Pixel Statistics (RGB, normalized 0-1):")
    print(
        f"  Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
    )
    print(
        f"  Std : R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
    )
    print("")

    # Store stats in df for relationship analysis
    # We can't easily add these back to the main df without processing all,
    # but we can return the sample stats for meta-feature analysis
    sample_df["width"] = widths
    sample_df["height"] = heights
    sample_df["aspect_ratio"] = aspect_ratios

    return sample_df


def analyze_metadata_features(df, sample_df):
    """
    Analyzes tabular features (chain, timestamp) and relationships.
    """
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # --- Chain Analysis ---
    if "chain" in df.columns:
        chain_counts = df["chain"].value_counts()
        print("Feature: Chain ID")
        print(f"  Unique Chains: {df['chain'].nunique()}")
        print(
            f"  Most Common Chain: {chain_counts.idxmax()} ({chain_counts.max()} images)"
        )
        print(
            f"  Least Common Chain: {chain_counts.idxmin()} ({chain_counts.min()} images)"
        )

        # Relationship: Chain vs Hotel
        # Does a hotel belong to multiple chains? (Should be No)
        hotel_chain_counts = df.groupby("hotel_id")["chain"].nunique()
        multi_chain_hotels = (hotel_chain_counts > 1).sum()
        print(f"  Hotels belonging to >1 chain: {multi_chain_hotels}")

    # --- Timestamp Analysis ---
    if "timestamp" in df.columns:
        # Convert to datetime, handle errors
        try:
            # Timestamps look like ISO format or similar. Let's infer.
            # train.csv description says "timestamp".
            # Let's try to parse a few to check format or just use pd.to_datetime
            times = pd.to_datetime(df["timestamp"], errors="coerce")
            valid_times = times.dropna()

            print("\nFeature: Timestamp")
            print(f"  Valid Timestamps: {len(valid_times)} / {len(df)}")
            if len(valid_times) > 0:
                print(f"  Range: {valid_times.min()} to {valid_times.max()}")

                # Seasonality / Distribution
                years = valid_times.dt.year
                months = valid_times.dt.month
                hours = valid_times.dt.hour

                print(f"  Year Distribution: {years.value_counts().to_dict()}")
                print(f"  Top 3 Months: {months.value_counts().head(3).index.tolist()}")
                print(f"  Top 3 Hours: {hours.value_counts().head(3).index.tolist()}")
        except Exception as e:
            print(f"\nFeature: Timestamp - Analysis failed: {e}")

    # --- Unstructured Relationships (using the image sample) ---
    if sample_df is not None:
        print("\nMeta-Feature Relationships (Sampled Data):")

        # Correlation: Image Size vs Chain
        # Do some chains have higher res images?
        sample_df["image_area"] = sample_df["width"] * sample_df["height"]

        # Correlation between image area and chain (treat chain as categorical, so use ANOVA or just mean comparison)
        # Here we just check if there's a visible difference in means for top chains
        top_chains = sample_df["chain"].value_counts().head(3).index
        for c in top_chains:
            mean_area = sample_df[sample_df["chain"] == c]["image_area"].mean()
            print(f"  Mean Image Area (pixels) for Chain {c}: {mean_area:.0f}")

        # Correlation: Aspect Ratio vs Hotel ID (is it specific?)
        # Hard to do with high cardinality, but we can check variance
        print(f"  Global Aspect Ratio Std: {sample_df['aspect_ratio'].std():.4f}")


def main():
    set_seed(SEED)

    try:
        df = load_data()

        # 1. Target Analysis
        analyze_target(df)

        # 2. Image Analysis (on sample)
        sample_df_with_stats = analyze_images(df)

        # 3. Metadata & Relationships
        analyze_metadata_features(df, sample_df_with_stats)

        print("\nEDA Complete.")

    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
