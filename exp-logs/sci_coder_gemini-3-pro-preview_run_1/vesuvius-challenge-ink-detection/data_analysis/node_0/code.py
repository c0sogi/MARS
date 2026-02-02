import os
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = (
    100  # Number of patches to sample for image stats to stay within time limits
)
SLICE_INDEX = 32  # Middle slice of the 65-slice stack


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_image_slice(surface_vol_path, slice_idx, x, y, w, h):
    """
    Loads a specific crop from a specific slice of the 3D volume.
    """
    # Construct path to the specific tif file
    # surface_vol_path is relative, e.g., "train/1/surface_volume"
    # The files are named 00.tif, 01.tif, ...
    slice_filename = f"{slice_idx:02d}.tif"
    full_path = os.path.join(INPUT_DIR, surface_vol_path, slice_filename)

    if not os.path.exists(full_path):
        return None

    # Load the entire slice (these are large images)
    # Optimization: In a real training loop, we might use memory mapping or tiling libraries.
    # For EDA on a few samples, loading with OpenCV is acceptable but we must crop immediately.
    # However, standard cv2.imread loads the whole image.
    # Given the constraints and library list, we use cv2.
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # Crop
    crop = img[y : y + h, x : x + w]
    return crop


def load_ink_label(inklabels_path, x, y, w, h):
    """
    Loads the binary ink label crop.
    """
    full_path = os.path.join(INPUT_DIR, inklabels_path)
    if not os.path.exists(full_path):
        return None

    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    crop = img[y : y + h, x : x + w]
    # Binarize (just in case)
    return (crop > 0).astype(np.uint8)


def main():
    set_seed(SEED)

    # --- 1. Data Integrity & Loading ---
    if not os.path.exists(METADATA_FILE):
        print("Error: Metadata file not found.")
        return

    df = pd.read_csv(METADATA_FILE)

    # Ensure we only analyze training data
    # The metadata file provided is already strictly training data, but we verify.
    # (No explicit 'split' column in train.csv, it is implicitly train)

    # --- 2. Target Variable Analysis ---
    print("TARGET VARIABLE ANALYSIS")

    # Patch-level distribution
    if "has_ink" in df.columns:
        counts = df["has_ink"].value_counts()
        total = len(df)
        print(f"Target Variable: has_ink (Patch Level)")
        print(f"Distribution: {counts.to_dict()}")

        # Class Balance
        pos_ratio = counts.get(1, 0) / total
        neg_ratio = counts.get(0, 0) / total
        print(f"Class Balance: Positive: {pos_ratio:.4f}, Negative: {neg_ratio:.4f}")
    else:
        print("Target variable 'has_ink' not found in metadata.")

    # Pixel-level distribution (Sampling)
    # We sample patches to estimate the global pixel-level ink ratio
    pixel_ink_counts = []
    pixel_total_counts = []

    sample_df = df.sample(n=min(len(df), SAMPLE_SIZE), random_state=SEED).copy()

    for _, row in sample_df.iterrows():
        if "inklabels_path" in row and pd.notna(row["inklabels_path"]):
            label_crop = load_ink_label(
                row["inklabels_path"], row["x"], row["y"], row["w"], row["h"]
            )
            if label_crop is not None:
                pixel_ink_counts.append(np.sum(label_crop))
                pixel_total_counts.append(label_crop.size)

    if pixel_total_counts:
        total_pixels = sum(pixel_total_counts)
        total_ink = sum(pixel_ink_counts)
        pixel_pos_ratio = total_ink / total_pixels
        print(f"Pixel Level Balance (Estimated from {len(sample_df)} samples):")
        print(f"Ink Pixels: {total_ink}")
        print(f"Total Pixels: {total_pixels}")
        print(f"Pixel Positive Ratio: {pixel_pos_ratio:.4f}")
    else:
        print("Could not load ink labels for pixel analysis.")

    print("-" * 30)

    # --- 3. Input Data Analysis (Image) ---
    print("INPUT DATA ANALYSIS")

    # We analyze the X-ray slices (Input)
    widths = []
    heights = []
    means = []
    stds = []
    mins = []
    maxs = []

    # We will also collect features for section 4 here
    meta_features = []

    for _, row in sample_df.iterrows():
        # Load middle slice
        img_crop = load_image_slice(
            row["surface_volume_path"],
            SLICE_INDEX,
            row["x"],
            row["y"],
            row["w"],
            row["h"],
        )

        if img_crop is not None:
            h, w = img_crop.shape
            widths.append(w)
            heights.append(h)

            # Pixel stats
            means.append(np.mean(img_crop))
            stds.append(np.std(img_crop))
            mins.append(np.min(img_crop))
            maxs.append(np.max(img_crop))

            # Store for relationship analysis
            meta_features.append(
                {
                    "mean_intensity": np.mean(img_crop),
                    "std_intensity": np.std(img_crop),
                    "x": row["x"],
                    "y": row["y"],
                    "has_ink": row["has_ink"],
                }
            )

    if widths:
        print(f"Dimensions (Sampled N={len(widths)}):")
        print(
            f"Width: Mean={np.mean(widths):.2f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"Height: Mean={np.mean(heights):.2f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )

        # Aspect Ratio
        ars = np.array(widths) / np.array(heights)
        print(f"Aspect Ratio: Mean={np.mean(ars):.4f}, Std={np.std(ars):.4f}")

        print("Channels:")
        print(
            "Format: 3D Volume (65 slices). Analysis performed on Slice 32 (Grayscale)."
        )

        print("Pixel Stats (Global Estimate):")
        print(f"Mean Intensity: {np.mean(means):.4f}")
        print(
            f"Std Deviation: {np.mean(stds):.4f}"
        )  # Average of local stds gives an idea of texture variance
        print(f"Global Min: {np.min(mins)}")
        print(f"Global Max: {np.max(maxs)}")
    else:
        print("No image data could be loaded.")

    print("-" * 30)

    # --- 4. Feature/Signal Relationships ---
    print("FEATURE/SIGNAL RELATIONSHIPS")

    if meta_features:
        feat_df = pd.DataFrame(meta_features)

        # Correlation
        print("Structured Relationships (Meta-features vs Target):")
        correlations = feat_df.corr()["has_ink"].drop("has_ink")
        print("Pearson Correlation with 'has_ink':")
        print(correlations.sort_values(ascending=False).to_string(float_format="%.4f"))

        # Feature Importance (Random Forest)
        X = feat_df.drop(columns=["has_ink"])
        y = feat_df["has_ink"]

        # Handle potential NaNs if any (unlikely here)
        X = X.fillna(0)

        rf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=X.columns)
        print("\nFeature Importance (Random Forest):")
        print(
            importances.sort_values(ascending=False)
            .head(5)
            .to_string(float_format="%.4f")
        )

        # Redundancy check
        print("\nRedundancy (Collinear Features > 0.90):")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        if to_drop:
            print(f"Collinear features found: {to_drop}")
        else:
            print("No highly collinear features found among meta-features.")

        # Unstructured/Meta analysis
        print("\nMeta-Feature Analysis:")
        # Check if ink is more prevalent in certain spatial regions
        # Bin 'y' coordinates to see if ink is top/bottom heavy in fragments
        feat_df["y_bin"] = pd.cut(
            feat_df["y"], bins=3, labels=["Top", "Middle", "Bottom"]
        )
        ink_by_y = feat_df.groupby("y_bin", observed=True)["has_ink"].mean()
        print("Ink Probability by Vertical Position (y):")
        print(ink_by_y.to_string(float_format="%.4f"))

    else:
        print("Insufficient data for feature relationship analysis.")


if __name__ == "__main__":
    main()
