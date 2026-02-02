import os
import cv2
import numpy as np
import pandas as pd
import multiprocessing
import random
from datetime import datetime


# Configuration
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SAMPLE_SIZE = 10000  # Number of images to sample for pixel/dimension analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_image_stats(args):
    """
    Worker function to retrieve stats for a single image.
    args: (relative_path, input_root)
    Returns: (width, height, channels, mean_pixel, std_pixel) or None
    """
    rel_path, input_root = args
    full_path = os.path.join(input_root, rel_path)

    try:
        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        # Dimensions
        h, w, c = img.shape

        # Pixel stats (Global for the image)
        # Convert to float for accurate mean/std calc
        img_float = img.astype(np.float32) / 255.0
        mean_val = np.mean(img_float)
        std_val = np.std(img_float)

        return (w, h, c, mean_val, std_val)
    except Exception:
        return None


def main():
    set_seed(SEED)

    # Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    # ---------------------------------------------------------
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("SECTION 1: TARGET VARIABLE ANALYSIS")

    target_col = "hotel_id"
    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    print(f"Target Variable: {target_col}")
    print(f"Total Samples: {total_samples}")
    print(f"Number of Unique Classes (Hotels): {num_classes}")

    # Imbalance / Skew
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()

    print(f"Class Balance Statistics:")
    print(f"  Min samples per class: {min_samples}")
    print(f"  Max samples per class: {max_samples}")
    print(f"  Mean samples per class: {mean_samples:.4f}")
    print(f"  Median samples per class: {median_samples:.4f}")

    # Top/Bottom classes
    print(f"  Most frequent class ratio: {max_samples/total_samples:.4f}")
    print(f"  Least frequent class ratio: {min_samples/total_samples:.4f}")
    print(f"  Imbalance Ratio (Max/Min): {max_samples/min_samples:.4f}")

    # ---------------------------------------------------------
    # SECTION 2: INPUT DATA ANALYSIS (IMAGE MODALITY)
    # ---------------------------------------------------------
    print("\nSECTION 2: INPUT DATA ANALYSIS (IMAGE MODALITY)")

    # Sample data for image analysis to save time
    if len(df) > SAMPLE_SIZE:
        # Stratified sampling if possible, otherwise random
        # Given high cardinality, simple random sampling is safer/faster than strict stratification
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    print(
        f"Analyzing a subset of {len(sample_df)} images for dimensions and pixel stats..."
    )

    # Prepare arguments for parallel processing
    # file_path column is relative to input dir
    tasks = [(row["file_path"], INPUT_DIR) for _, row in sample_df.iterrows()]

    # Run sequential processing
    results = [get_image_stats(task) for task in tasks]

    widths = []
    heights = []
    channels = []
    means = []
    stds = []
    aspect_ratios = []

    # Filter None results
    valid_results = [r for r in results if r is not None]

    for w, h, c, m, s in valid_results:
        widths.append(w)
        heights.append(h)
        channels.append(c)
        means.append(m)
        stds.append(s)
        aspect_ratios.append(w / h if h > 0 else 0)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    means = np.array(means)
    stds = np.array(stds)

    # Dimensions
    print("Image Dimensions:")
    print(
        f"  Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )

    # Aspect Ratios
    print("Aspect Ratios (Width/Height):")
    print(f"  Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}")
    print(f"  Min: {np.min(aspect_ratios):.4f}, Max: {np.max(aspect_ratios):.4f}")

    # Channels
    unique_channels, channel_counts = np.unique(channels, return_counts=True)
    print("Channel Distribution:")
    for c, count in zip(unique_channels, channel_counts):
        print(f"  {c} Channels: {count} images ({count/len(valid_results):.4f})")

    # Pixel Stats
    print("Pixel Intensity Statistics (Normalized 0-1):")
    print(f"  Global Mean: {np.mean(means):.4f}")
    print(f"  Global Std:  {np.mean(stds):.4f}")

    # ---------------------------------------------------------
    # SECTION 3: TABULAR/METADATA ANALYSIS
    # ---------------------------------------------------------
    print("\nSECTION 3: TABULAR/METADATA ANALYSIS")

    # Chain Analysis
    if "chain" in df.columns:
        chain_counts = df["chain"].value_counts()
        print("Chain Feature Analysis:")
        print(f"  Number of unique chains: {df['chain'].nunique()}")
        print(
            f"  Most common chain ID: {chain_counts.idxmax()} (Count: {chain_counts.max()})"
        )
        print(
            f"  Least common chain ID: {chain_counts.idxmin()} (Count: {chain_counts.min()})"
        )

        # Check for missing/zero chains
        zero_chain_count = (df["chain"] == 0).sum()
        print(
            f"  Unknown Chain (ID=0) count: {zero_chain_count} ({zero_chain_count/len(df):.4f})"
        )

    # Timestamp Analysis
    if "timestamp" in df.columns:
        print("Timestamp Feature Analysis:")
        # Convert to datetime, handling errors
        times = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_times = times.dropna()

        if len(valid_times) > 0:
            print(f"  Time Range: {valid_times.min()} to {valid_times.max()}")
            print(f"  Missing Timestamps: {len(df) - len(valid_times)}")

            # Extract Year and Month
            years = valid_times.dt.year
            months = valid_times.dt.month

            print(f"  Most frequent year: {years.mode()[0]}")
            print(f"  Most frequent month: {months.mode()[0]}")
        else:
            print("  No valid timestamps found.")

    # ---------------------------------------------------------
    # SECTION 4: FEATURE/SIGNAL RELATIONSHIPS
    # ---------------------------------------------------------
    print("\nSECTION 4: FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Chain vs Hotel ID Nesting
    # Does a hotel_id belong to only one chain?
    if "chain" in df.columns:
        hotel_chain_counts = df.groupby("hotel_id")["chain"].nunique()
        multi_chain_hotels = (hotel_chain_counts > 1).sum()
        print("Meta-Feature Relationship: Chain vs Hotel ID")
        if multi_chain_hotels == 0:
            print("  Perfect Nesting: Each Hotel ID belongs to exactly one Chain ID.")
        else:
            print(
                f"  Imperfect Nesting: {multi_chain_hotels} hotels belong to multiple chains."
            )

        # Correlation (Cramer's V or similar is complex, let's do simple Mutual Information proxy via groupby)
        # Just reporting average hotels per chain
        hotels_per_chain = df.groupby("chain")["hotel_id"].nunique()
        print(f"  Avg Hotels per Chain: {hotels_per_chain.mean():.4f}")
        print(f"  Max Hotels in a Chain: {hotels_per_chain.max()}")

    # 2. Image Metadata vs Target
    # Do certain hotels have consistently larger images?
    # We use the sampled data for this.
    # Map back stats to the dataframe
    # Since we processed a list of tasks, we need to align them.
    # The 'tasks' list was created from 'sample_df'. 'valid_results' might be shorter if errors occurred.
    # We will just do a quick correlation on the valid subset.

    if len(valid_results) > 0:
        # Create a temporary dataframe for the sampled stats
        # We need to be careful about alignment.
        # Let's assume few errors. If errors, alignment breaks.
        # Safer approach:
        valid_indices = [i for i, r in enumerate(results) if r is not None]
        valid_stats = [results[i] for i in valid_indices]

        # Get corresponding hotel_ids
        sampled_hotel_ids = sample_df.iloc[valid_indices]["hotel_id"].values
        sampled_widths = np.array([s[0] for s in valid_stats])
        sampled_heights = np.array([s[1] for s in valid_stats])

        # Correlation between image size (area) and hotel_id frequency?
        # (Are popular hotels represented by larger images?)
        # First, get frequency of each hotel in the full dataset
        hotel_freq_map = df["hotel_id"].value_counts().to_dict()
        sampled_freqs = np.array([hotel_freq_map[hid] for hid in sampled_hotel_ids])
        sampled_areas = sampled_widths * sampled_heights

        corr_area_freq = np.corrcoef(sampled_areas, sampled_freqs)[0, 1]

        print("Image Signal vs Target Frequency:")
        print(
            f"  Correlation between Image Area and Hotel Class Frequency: {corr_area_freq:.4f}"
        )
        print(
            "  (Values near 0 indicate image resolution is independent of how common the hotel is in the dataset)"
        )


if __name__ == "__main__":
    main()
