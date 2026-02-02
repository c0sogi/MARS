import os
import pandas as pd
import numpy as np
import random
import ast
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE = 300  # Number of images to sample for pixel analysis

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")


def analyze_targets(df):
    print_header("TARGET VARIABLE ANALYSIS")

    # 1. Study Level Labels
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    print("--- Class Distribution (Study Level) ---")
    total_studies = df["study_id"].nunique()
    study_df = df.drop_duplicates("study_id")

    for col in label_cols:
        count = study_df[col].sum()
        ratio = count / total_studies
        print(f"{col:<30}: {count} ({ratio:.4%})")

    # Check for Multi-label studies
    study_df["label_sum"] = study_df[label_cols].sum(axis=1)
    multi_label_count = (study_df["label_sum"] > 1).sum()
    no_label_count = (study_df["label_sum"] == 0).sum()
    print(
        f"\nMulti-label Studies           : {multi_label_count} ({multi_label_count/total_studies:.4%})"
    )
    print(
        f"Studies with No Label         : {no_label_count} ({no_label_count/total_studies:.4%})"
    )

    # 2. Image Level Labels (Bounding Boxes)
    print("\n--- Bounding Box Analysis (Image Level) ---")

    def parse_boxes(x):
        if pd.isna(x):
            return []
        try:
            # The format in the CSV is usually a string representation of a list of dicts
            return ast.literal_eval(x)
        except:
            return []

    # Parse boxes
    # Note: 'boxes' column contains the bounding boxes
    box_counts = df["boxes"].apply(lambda x: len(parse_boxes(x)))

    print(
        f"Images with 0 boxes           : {(box_counts == 0).sum()} ({(box_counts == 0).mean():.4%})"
    )
    print(
        f"Images with >0 boxes          : {(box_counts > 0).sum()} ({(box_counts > 0).mean():.4%})"
    )
    print(f"Max boxes in single image     : {box_counts.max()}")
    print(f"Avg boxes per image           : {box_counts.mean():.4f}")

    # Analyze Box Areas (if any exist)
    all_areas = []
    # We iterate a sample to avoid massive overhead if dataset is huge,
    # but here we can do all since it's just parsing strings
    valid_boxes_df = df[df["boxes"].notna()]

    for _, row in valid_boxes_df.iterrows():
        boxes = parse_boxes(row["boxes"])
        for box in boxes:
            # Box format usually: {'x': ..., 'y': ..., 'width': ..., 'height': ...}
            if "width" in box and "height" in box:
                all_areas.append(box["width"] * box["height"])

    if all_areas:
        print(f"Avg Bounding Box Area (px^2)  : {np.mean(all_areas):.4f}")
        print(f"Min Bounding Box Area (px^2)  : {np.min(all_areas):.4f}")
        print(f"Max Bounding Box Area (px^2)  : {np.max(all_areas):.4f}")

    return study_df, box_counts


def analyze_images(df):
    print_header("INPUT DATA ANALYSIS (IMAGE MODALITY)")

    # Check for pydicom
    try:
        import pydicom
    except ImportError:
        print("WARNING: 'pydicom' library not found. Skipping pixel-level analysis.")
        return None

    print(f"Sampling {SAMPLE_SIZE} images for statistical analysis...")

    sample_df = df.sample(n=min(len(df), SAMPLE_SIZE), random_state=SEED)

    widths = []
    heights = []
    aspect_ratios = []
    pixel_means = []
    pixel_stds = []
    channels = []

    start_time = time.time()

    for idx, row in sample_df.iterrows():
        # Construct full path
        # file_path in metadata is relative to input dir
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            dcm = pydicom.dcmread(full_path, stop_before_pixels=False)

            # Dimensions
            h = dcm.Rows
            w = dcm.Columns
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

            # Channels
            # DICOMs are typically grayscale (1 channel), but SamplesPerPixel tag tells us
            c = getattr(dcm, "SamplesPerPixel", 1)
            channels.append(c)

            # Pixel Stats
            # Use float to avoid overflow during mean calc
            pixels = dcm.pixel_array.astype(float)
            pixel_means.append(np.mean(pixels))
            pixel_stds.append(np.std(pixels))

        except Exception as e:
            # Fail silently for individual corrupt files in EDA
            continue

    elapsed = time.time() - start_time
    print(f"Analysis complete in {elapsed:.2f} seconds.")

    if not widths:
        print("No images could be read successfully.")
        return None

    # Report Dimensions
    print("\n--- Image Dimensions ---")
    print(
        f"Width  (Mean ± Std)           : {np.mean(widths):.4f} ± {np.std(widths):.4f}"
    )
    print(f"       (Min / Max)            : {np.min(widths)} / {np.max(widths)}")
    print(
        f"Height (Mean ± Std)           : {np.mean(heights):.4f} ± {np.std(heights):.4f}"
    )
    print(f"       (Min / Max)            : {np.min(heights)} / {np.max(heights)}")
    print(
        f"Aspect Ratio (Mean ± Std)     : {np.mean(aspect_ratios):.4f} ± {np.std(aspect_ratios):.4f}"
    )

    # Report Channels
    unique_channels, counts = np.unique(channels, return_counts=True)
    print("\n--- Channels ---")
    for c, count in zip(unique_channels, counts):
        print(f"Channel Count {c}: {count} images")

    # Report Pixel Stats
    print("\n--- Pixel Intensity Statistics ---")
    print(f"Global Pixel Mean             : {np.mean(pixel_means):.4f}")
    print(f"Global Pixel Std Dev          : {np.mean(pixel_stds):.4f}")

    # Return collected stats for relationship analysis
    stats_df = pd.DataFrame(
        {
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "pixel_mean": pixel_means,
            "study_id": sample_df["study_id"].values,
        }
    )
    return stats_df


def analyze_relationships(df, study_df, box_counts, image_stats_df):
    print_header("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Meta-Feature Relationships (Box Count vs Class)
    print("--- Relationship: Bounding Box Count vs Study Label ---")

    # Merge box counts back to study level (taking max box count per study if multiple images,
    # though usually 1 image per study in this dataset context, but let's be safe)
    # Actually, df is image level. Let's merge box counts with labels.

    df_rel = df.copy()
    df_rel["num_boxes"] = box_counts

    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    for label in label_cols:
        # Get average box count for positive cases of this label
        subset = df_rel[df_rel[label] == 1]
        avg_boxes = subset["num_boxes"].mean()
        print(f"Avg Boxes for '{label}': {avg_boxes:.4f}")

    # 2. Label Correlations
    print("\n--- Study Label Correlations ---")
    corr_matrix = study_df[label_cols].corr()
    print(corr_matrix.round(4))

    # 3. Image Size vs Label (using the sampled stats)
    if image_stats_df is not None:
        print("\n--- Relationship: Image Area vs Study Label (Sampled) ---")

        # Merge stats with labels
        # image_stats_df has study_id, we need to join with study_df labels
        merged_stats = pd.merge(
            image_stats_df,
            study_df[["study_id"] + label_cols],
            on="study_id",
            how="left",
        )
        merged_stats["area"] = merged_stats["width"] * merged_stats["height"]

        for label in label_cols:
            # Point Biserial correlation roughly
            # Compare mean area of positive vs negative
            pos_area = merged_stats[merged_stats[label] == 1]["area"].mean()
            neg_area = merged_stats[merged_stats[label] == 0]["area"].mean()

            print(
                f"Mean Area (px^2) for {label:<25} Positive: {pos_area:12.4f} | Negative: {neg_area:12.4f}"
            )


def main():
    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded training metadata with {len(df)} rows.")

    # Run Analysis
    study_df, box_counts = analyze_targets(df)
    image_stats_df = analyze_images(df)
    analyze_relationships(df, study_df, box_counts, image_stats_df)

    print_header("EDA COMPLETE")


if __name__ == "__main__":
    main()
