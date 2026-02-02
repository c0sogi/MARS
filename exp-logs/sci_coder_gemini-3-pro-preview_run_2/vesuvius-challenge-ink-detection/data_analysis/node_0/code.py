import os
import cv2
import numpy as np
import pandas as pd
import random
from glob import glob
from scipy import stats

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = 200  # Number of patches to sample for volume statistics to save time

# --- Seeding ---
random.seed(SEED)
np.random.seed(SEED)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def load_image(path, flags=cv2.IMREAD_UNCHANGED):
    full_path = os.path.join(INPUT_DIR, path)
    if not os.path.exists(full_path):
        return None
    return cv2.imread(full_path, flags)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")

    # Identify unique fragments in the training set
    fragment_ids = df["fragment_id"].unique()

    total_valid_pixels = 0
    total_ink_pixels = 0

    print(f"Analyzing {len(fragment_ids)} unique fragments from training metadata...")

    for fid in fragment_ids:
        # Get paths from the first entry of this fragment
        frag_row = df[df["fragment_id"] == fid].iloc[0]
        label_path = frag_row["label_path"]
        mask_path = frag_row["mask_path"]

        # Load images
        # Labels and Masks are 2D PNGs
        label_img = load_image(label_path, cv2.IMREAD_GRAYSCALE)
        mask_img = load_image(mask_path, cv2.IMREAD_GRAYSCALE)

        if label_img is None or mask_img is None:
            continue

        # Ensure binary
        # Mask: >0 is valid
        # Label: >0 is ink
        valid_mask = mask_img > 0
        ink_mask = label_img > 0

        # We only care about pixels inside the valid mask
        valid_pixels_count = np.count_nonzero(valid_mask)
        # Ink pixels must be inside valid mask (usually they are, but strictly enforcing)
        ink_pixels_count = np.count_nonzero(np.logical_and(ink_mask, valid_mask))

        total_valid_pixels += valid_pixels_count
        total_ink_pixels += ink_pixels_count

        ink_ratio = (
            ink_pixels_count / valid_pixels_count if valid_pixels_count > 0 else 0
        )
        print(
            f"Fragment {fid}: Ink Ratio = {ink_ratio:.4f} ({ink_pixels_count}/{valid_pixels_count} pixels)"
        )

    # Global Stats
    global_ink_ratio = (
        total_ink_pixels / total_valid_pixels if total_valid_pixels > 0 else 0
    )
    print("-" * 30)
    print(f"Global Ink Pixel Ratio: {global_ink_ratio:.4f}")
    print(
        f"Class Balance (No-Ink : Ink): {1-global_ink_ratio:.4f} : {global_ink_ratio:.4f}"
    )

    # Skewness check (Bernoulli distribution of pixels)
    # For binary classification, imbalance is the key metric.
    if global_ink_ratio < 0.05:
        print("Observation: High Class Imbalance detected (Ink is rare).")
    else:
        print("Observation: Moderate Class Imbalance.")
    print("\n")


def analyze_input_images(df):
    print("INPUT DATA ANALYSIS (IMAGE/VOLUME)")

    # 1. Dimensions of full fragments
    fragment_ids = df["fragment_id"].unique()
    print("Fragment Dimensions:")
    for fid in fragment_ids:
        frag_row = df[df["fragment_id"] == fid].iloc[0]
        mask_img = load_image(frag_row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            h, w = mask_img.shape
            print(f"Fragment {fid}: {w}x{h} pixels")

    # 2. Channel/Volume Analysis
    # We will sample patches to check the Z-stack properties
    print(f"\nSampling {SAMPLE_SIZE} patches for Volume Statistics...")

    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    pixel_means = []
    pixel_stds = []
    pixel_mins = []
    pixel_maxs = []
    dtypes = set()

    # We will look at the middle slice (approx index 32) for statistics
    # to avoid loading 65 slices * 200 patches which might be slow.
    Z_SLICE_INDEX = 32

    for _, row in sample_df.iterrows():
        vol_dir = row["volume_path"]
        # Construct path to slice 32
        slice_filename = f"{Z_SLICE_INDEX:02d}.tif"
        slice_path = os.path.join(vol_dir, slice_filename)

        # Load the specific crop from the large TIFF
        # Since we can't easily crop without loading the whole file or using memory mapping,
        # and standard cv2.imread loads the whole image, we load the whole slice image
        # (which is cached by OS hopefully) and crop.
        # Note: Ideally we use memory mapping for large TIFFs, but cv2 is standard here.
        # To optimize, we will cache the loaded full slice for the current fragment loop if possible,
        # but here we just load.

        img_slice = load_image(slice_path, cv2.IMREAD_UNCHANGED)

        if img_slice is None:
            continue

        # Crop patch
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]
        # Handle boundary checks
        img_h, img_w = img_slice.shape
        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)

        patch = img_slice[y:y_end, x:x_end]

        if patch.size == 0:
            continue

        dtypes.add(patch.dtype)
        pixel_means.append(np.mean(patch))
        pixel_stds.append(np.std(patch))
        pixel_mins.append(np.min(patch))
        pixel_maxs.append(np.max(patch))

    # Report Stats
    print(f"Data Types Found: {[str(d) for d in dtypes]}")
    if pixel_means:
        print(f"Pixel Value Mean (Slice {Z_SLICE_INDEX}): {np.mean(pixel_means):.4f}")
        print(f"Pixel Value Std  (Slice {Z_SLICE_INDEX}): {np.mean(pixel_stds):.4f}")
        print(f"Pixel Value Min  (Global): {np.min(pixel_mins):.4f}")
        print(f"Pixel Value Max  (Global): {np.max(pixel_maxs):.4f}")

    # Check for normalization needs
    if "uint16" in [str(d) for d in dtypes]:
        print(
            "Observation: Data is 16-bit. Normalization to [0,1] or standardization is recommended."
        )

    print("\n")
    return sample_df, pixel_means  # Return for relationship analysis


def analyze_relationships(sample_df, pixel_means):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    if sample_df is None or not pixel_means:
        print("Insufficient data for relationship analysis.")
        return

    # Add calculated means to the dataframe copy
    # Note: pixel_means aligns with sample_df iteration order
    df_rel = sample_df.copy()
    df_rel["mean_intensity"] = pixel_means

    # 1. Unstructured Relationship: Mean Intensity vs Target (Has Ink)
    # Do ink patches have higher/lower intensity in X-ray?

    ink_patches = df_rel[df_rel["has_ink"] == 1]["mean_intensity"]
    no_ink_patches = df_rel[df_rel["has_ink"] == 0]["mean_intensity"]

    print("Relationship: Mean Intensity (Slice 32) vs Ink Presence")

    if len(ink_patches) > 0:
        print(
            f"Ink Patch Intensity:    Mean={ink_patches.mean():.4f}, Std={ink_patches.std():.4f}, Count={len(ink_patches)}"
        )
    else:
        print("Ink Patch Intensity:    N/A (No ink patches in sample)")

    if len(no_ink_patches) > 0:
        print(
            f"No-Ink Patch Intensity: Mean={no_ink_patches.mean():.4f}, Std={no_ink_patches.std():.4f}, Count={len(no_ink_patches)}"
        )
    else:
        print("No-Ink Patch Intensity: N/A")

    # Correlation
    # Point-Biserial Correlation (Continuous vs Binary)
    if len(ink_patches) > 0 and len(no_ink_patches) > 0:
        corr, p_val = stats.pointbiserialr(df_rel["has_ink"], df_rel["mean_intensity"])
        print(f"Point-Biserial Correlation: {corr:.4f} (p-value: {p_val:.4f})")

        if abs(corr) < 0.1:
            print(
                "Observation: Very weak linear correlation between single-slice intensity and ink label."
            )
            print(
                "Suggestion: Ink detection likely requires texture features, 3D shape analysis, or multi-slice context."
            )
        else:
            print("Observation: Detectable correlation between intensity and ink.")

    print("\n")


def main():
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 2. Modality Check
    # We infer modality from column names and file extensions
    # Columns: mask_path, volume_path, etc.
    print("DATA MODALITY DETECTION")
    if "volume_path" in df.columns and "mask_path" in df.columns:
        print("Modality: Image / Volumetric Data (Vesuvius Ink Detection)")
    else:
        print("Modality: Unknown / Tabular")
    print("\n")

    # 3. Target Analysis
    analyze_target(df)

    # 4. Input Analysis
    sample_df, pixel_means = analyze_input_images(df)

    # 5. Relationships
    analyze_relationships(sample_df, pixel_means)


if __name__ == "__main__":
    main()
