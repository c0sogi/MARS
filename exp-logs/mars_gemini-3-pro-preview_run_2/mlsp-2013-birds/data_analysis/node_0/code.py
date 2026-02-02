import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import cv2
import random
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    SUPPLEMENTAL_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    print("EDA REPORT\n" + "=" * 50)

    # --- 1. Load Data ---
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found.")
        return

    df_train = pd.read_csv(TRAIN_CSV)

    # Identify Label Columns (species_0 to species_18)
    label_cols = [c for c in df_train.columns if c.startswith("species_")]
    num_species = len(label_cols)

    print("DATASET SUMMARY")
    print(f"Training Samples: {len(df_train)}")
    print(f"Total Species (Targets): {num_species}")
    print("-" * 30)

    # --- 2. Target Variable Analysis ---
    print("\nTARGET VARIABLE ANALYSIS (MULTI-LABEL CLASSIFICATION)")

    # Class Balance
    label_counts = df_train[label_cols].sum().sort_values(ascending=False)
    total_samples = len(df_train)

    print("\n[Class Balance Ratios]")
    print(f"{'Species':<15} | {'Count':<10} | {'Ratio':<10}")
    print("-" * 45)
    for species, count in label_counts.items():
        ratio = count / total_samples
        print(f"{species:<15} | {count:<10} | {ratio:.4f}")

    # Label Cardinality (Avg labels per sample)
    labels_per_sample = df_train[label_cols].sum(axis=1)
    avg_cardinality = labels_per_sample.mean()
    print(f"\nAverage Label Cardinality: {avg_cardinality:.4f}")

    # Label Density (Avg labels per sample / Total labels)
    label_density = avg_cardinality / num_species
    print(f"Label Density: {label_density:.4f}")

    # Rows with no labels
    no_label_count = (labels_per_sample == 0).sum()
    print(
        f"Samples with No Labels: {no_label_count} ({no_label_count/total_samples:.2%})"
    )

    # --- 3. Audio Data Analysis ---
    print("\nINPUT DATA ANALYSIS: AUDIO")

    durations = []
    sample_rates = []
    channels = []
    bit_depths = []

    # Iterate through a sample of files to save time if dataset is huge,
    # but here dataset is small (206), so we do all.
    for idx, row in df_train.iterrows():
        # Construct full path. Metadata contains relative path like 'essential_data/...'
        rel_path = row["file_path_wav"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        try:
            with sf.SoundFile(full_path) as f:
                durations.append(len(f) / f.samplerate)
                sample_rates.append(f.samplerate)
                channels.append(f.channels)
                # subtype gives format like 'PCM_16'
                bit_depths.append(f.subtype)
        except Exception as e:
            pass

    if durations:
        print(f"\n[Audio Signal Properties]")
        print(
            f"Duration Mean: {np.mean(durations):.4f}s, Std: {np.std(durations):.4f}s"
        )
        print(f"Duration Min: {np.min(durations):.4f}s, Max: {np.max(durations):.4f}s")

        unique_sr = np.unique(sample_rates)
        print(f"Sampling Rates: {unique_sr}")

        unique_ch = np.unique(channels)
        print(f"Channels: {unique_ch} (1=Mono, 2=Stereo)")

        unique_bd = np.unique(bit_depths)
        print(f"Bit Depths/Subtypes: {unique_bd}")
    else:
        print("No audio files could be processed.")

    # --- 4. Image Data Analysis ---
    print("\nINPUT DATA ANALYSIS: IMAGE (SPECTROGRAMS)")

    widths = []
    heights = []
    img_channels = []
    pixel_means = []
    pixel_stds = []

    for idx, row in df_train.iterrows():
        rel_path = row["file_path_spec"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                c = 1 if len(img.shape) == 2 else img.shape[2]

                widths.append(w)
                heights.append(h)
                img_channels.append(c)

                # Compute simple stats
                pixel_means.append(np.mean(img))
                pixel_stds.append(np.std(img))

    if widths:
        print(f"\n[Image Dimensions]")
        print(f"Width Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}")
        print(f"Height Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}")
        print(f"Aspect Ratio Mean: {np.mean(np.array(widths)/np.array(heights)):.4f}")

        print(f"\n[Channels]")
        print(f"Unique Channel Counts: {np.unique(img_channels)}")

        print(f"\n[Pixel Statistics (0-255)]")
        print(f"Global Pixel Mean: {np.mean(pixel_means):.4f}")
        print(f"Global Pixel Std: {np.mean(pixel_stds):.4f}")
    else:
        print("No image files could be processed.")

    # --- 5. Feature/Signal Relationships (Tabular) ---
    print("\nFEATURE RELATIONSHIPS (TABULAR: HISTOGRAM OF SEGMENTS)")

    # Load supplemental tabular data
    hist_path = os.path.join(SUPPLEMENTAL_DIR, "histogram_of_segments.txt")

    if os.path.exists(hist_path):
        # The file format is: rec_id,val1,val2...
        # We need to parse this carefully.
        try:
            # Read all lines
            with open(hist_path, "r") as f:
                lines = f.readlines()

            data_rows = []
            # Skip header if it exists (check first line)
            start_idx = 0
            if "rec_id" in lines[0]:
                start_idx = 1

            for line in lines[start_idx:]:
                parts = line.strip().split(",")
                if len(parts) > 1:
                    rec_id = int(parts[0])
                    features = [float(x) for x in parts[1:]]
                    data_rows.append([rec_id] + features)

            # Create DataFrame
            num_features = len(data_rows[0]) - 1
            feat_cols = [f"feat_{i}" for i in range(num_features)]
            df_feats = pd.DataFrame(data_rows, columns=["rec_id"] + feat_cols)

            # Merge with training data to ensure we only analyze training set
            df_merged = df_train[["rec_id"] + label_cols].merge(
                df_feats, on="rec_id", how="inner"
            )

            X = df_merged[feat_cols]
            y = df_merged[label_cols]

            print(f"Matched {len(df_merged)} training samples with tabular features.")

            # Numerical Analysis
            print(f"\n[Numerical Feature Stats]")
            feat_means = X.mean()
            feat_stds = X.std()
            print(
                f"Feature Means - Min: {feat_means.min():.4f}, Max: {feat_means.max():.4f}, Avg: {feat_means.mean():.4f}"
            )
            print(
                f"Feature Stds  - Min: {feat_stds.min():.4f}, Max: {feat_stds.max():.4f}, Avg: {feat_stds.mean():.4f}"
            )

            # Sparsity check (since it's a histogram/bag-of-words)
            sparsity = (X == 0).sum().sum() / (X.shape[0] * X.shape[1])
            print(f"Feature Sparsity (Zero Ratio): {sparsity:.4f}")

            # Feature Importance (Random Forest)
            print(f"\n[Feature Importance]")
            rf = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            )
            rf.fit(X, y)

            # Aggregate importance across all estimators
            importances = np.mean(
                [est.feature_importances_ for est in rf.estimators_], axis=0
            )
            indices = np.argsort(importances)[::-1]

            print("Top 5 Features by Importance:")
            for i in range(5):
                print(f"  {feat_cols[indices[i]]}: {importances[indices[i]]:.4f}")

            # Redundancy (Collinearity)
            print(f"\n[Redundancy Check]")
            # Compute correlation matrix
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

            print(f"Number of features with Correlation > 0.90: {len(to_drop)}")
            if len(to_drop) > 0:
                print(f"Examples: {to_drop[:5]}")

        except Exception as e:
            print(f"Failed to process tabular data: {e}")
    else:
        print("Tabular feature file not found.")

    # --- 6. Meta-Feature Relationships ---
    print("\nMETA-FEATURE RELATIONSHIPS")
    # Check if number of detected species correlates with file properties
    # (Though duration is likely constant, let's check pixel intensity)

    if len(pixel_means) == len(df_train):
        meta_df = pd.DataFrame(
            {
                "pixel_mean": pixel_means,
                "pixel_std": pixel_stds,
                "num_species": df_train[label_cols].sum(axis=1),
            }
        )

        corr_mean, _ = pearsonr(meta_df["pixel_mean"], meta_df["num_species"])
        corr_std, _ = pearsonr(meta_df["pixel_std"], meta_df["num_species"])

        print(f"Correlation (Pixel Mean vs Num Species): {corr_mean:.4f}")
        print(f"Correlation (Pixel Std  vs Num Species): {corr_std:.4f}")
        print(
            "Interpretation: Higher pixel intensity/variance might indicate more bird activity."
        )

    print("\n" + "=" * 50)
    print("EDA COMPLETE")


if __name__ == "__main__":
    main()
