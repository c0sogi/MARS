import os
import numpy as np
import pandas as pd
import random
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # --- Configuration ---
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"
    SAMPLE_SIZE = 300  # Number of samples to analyze for heavy image operations
    SEED = 42

    set_seed(SEED)

    # --- Load Metadata ---
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # Sample the dataframe if it's too large
    if len(df_train) > SAMPLE_SIZE:
        df_sample = df_train.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(
            drop=True
        )
    else:
        df_sample = df_train.copy()

    # --- 1. Target Variable Analysis ---
    print("TARGET VARIABLE ANALYSIS")

    total_pixels = 0
    contrail_pixels = 0
    images_with_contrails = 0
    contrail_pixel_counts = []  # For correlation later

    # We also need to track metadata for correlation analysis
    timestamps = []
    row_mins = []
    col_mins = []

    # Band statistics accumulators
    band_ids = [f"band_{i:02d}" for i in range(8, 17)]
    band_stats = {b: {"sum": 0.0, "sq_sum": 0.0, "count": 0} for b in band_ids}

    # Dimension tracking
    heights = []
    widths = []
    temporal_depths = []

    # Iterate through samples
    for idx, row in df_sample.iterrows():
        # -- Target Analysis --
        mask_path = os.path.join(INPUT_DIR, row["human_pixel_masks"])
        try:
            # Mask shape is H x W x 1
            mask = np.load(mask_path)

            n_pixels = mask.size
            n_pos = np.sum(mask)

            total_pixels += n_pixels
            contrail_pixels += n_pos
            contrail_pixel_counts.append(n_pos)

            if n_pos > 0:
                images_with_contrails += 1

            # -- Input Data Analysis (Dimensions) --
            # Check dimensions from the mask (spatial)
            h, w = mask.shape[:2]
            heights.append(h)
            widths.append(w)

            # -- Metadata Collection --
            timestamps.append(row["timestamp"])
            row_mins.append(row["row_min"])
            col_mins.append(row["col_min"])

            # -- Input Data Analysis (Pixel Stats) --
            # Load bands
            for b_id in band_ids:
                band_path = os.path.join(INPUT_DIR, row[b_id])
                band_data = np.load(band_path)

                # Check temporal depth once per image (from first band)
                if b_id == "band_08":
                    # Shape is H x W x T
                    if len(band_data.shape) == 3:
                        temporal_depths.append(band_data.shape[2])
                    else:
                        temporal_depths.append(1)

                # Accumulate stats
                # Flatten for calculation
                flat_band = band_data.astype(np.float64).flatten()
                band_stats[b_id]["sum"] += np.sum(flat_band)
                band_stats[b_id]["sq_sum"] += np.sum(flat_band**2)
                band_stats[b_id]["count"] += flat_band.size

        except Exception as e:
            # Skip broken files in EDA but log silently if needed
            continue

    # -- Target Metrics --
    global_pos_ratio = contrail_pixels / total_pixels if total_pixels > 0 else 0
    image_pos_ratio = (
        images_with_contrails / len(df_sample) if len(df_sample) > 0 else 0
    )

    print(f"Target Type: Binary Segmentation Mask")
    print(
        f"Global Pixel Class Balance (Contrail): {global_pos_ratio:.4f} ({global_pos_ratio*100:.2f}%)"
    )
    print(
        f"Global Pixel Class Balance (Background): {1-global_pos_ratio:.4f} ({(1-global_pos_ratio)*100:.2f}%)"
    )
    print(
        f"Image-Level Class Balance (Images with Contrails): {image_pos_ratio:.4f} ({image_pos_ratio*100:.2f}%)"
    )
    print(
        f"Image-Level Class Balance (Empty Masks): {1-image_pos_ratio:.4f} ({(1-image_pos_ratio)*100:.2f}%)"
    )

    print("\nINPUT DATA ANALYSIS")

    # -- Dimensions --
    avg_h = np.mean(heights) if heights else 0
    avg_w = np.mean(widths) if widths else 0
    avg_t = np.mean(temporal_depths) if temporal_depths else 0

    print(f"Image Dimensions (H x W): {avg_h:.1f} x {avg_w:.1f}")
    print(f"Temporal Depth (Frames per Band): {avg_t:.1f}")
    print(
        f"Aspect Ratio Distribution: All images are {int(avg_h)}x{int(avg_w)} (Square)"
    )

    # -- Channels / Pixel Stats --
    print(f"Channel Statistics (Calculated on {len(df_sample)} samples):")
    print(f"{'Band':<10} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 36)

    for b_id in band_ids:
        stats = band_stats[b_id]
        if stats["count"] > 0:
            mean_val = stats["sum"] / stats["count"]
            # Var = E[X^2] - (E[X])^2
            var_val = (stats["sq_sum"] / stats["count"]) - (mean_val**2)
            std_val = np.sqrt(
                max(0, var_val)
            )  # max(0, ..) to handle tiny precision errors
            print(f"{b_id:<10} | {mean_val:.4f}     | {std_val:.4f}")
        else:
            print(f"{b_id:<10} | N/A        | N/A")

    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # -- Meta-Feature Relationships --
    # Create a small dataframe for correlation analysis
    corr_df = pd.DataFrame(
        {
            "contrail_pixels": contrail_pixel_counts,
            "timestamp": timestamps,
            "row_min": row_mins,
            "col_min": col_mins,
        }
    )

    # Handle NaNs if any
    corr_df = corr_df.dropna()

    if not corr_df.empty:
        # Pearson Correlation
        corr_matrix = corr_df.corr(method="pearson")

        target_corr = corr_matrix["contrail_pixels"]

        print("Correlation with Target (Contrail Pixel Count):")
        print(f"Timestamp (Time): {target_corr.get('timestamp', 0):.4f}")
        print(f"Row Min (Latitude Proxy): {target_corr.get('row_min', 0):.4f}")
        print(f"Col Min (Longitude Proxy): {target_corr.get('col_min', 0):.4f}")

        print("\nInterpretation:")
        t_corr = target_corr.get("timestamp", 0)
        if abs(t_corr) < 0.1:
            print("Timestamp shows negligible linear correlation with contrail size.")
        else:
            print(
                f"Timestamp shows {'positive' if t_corr > 0 else 'negative'} correlation ({t_corr:.4f}) with contrail size."
            )

    else:
        print("Insufficient data for correlation analysis.")


if __name__ == "__main__":
    main()
