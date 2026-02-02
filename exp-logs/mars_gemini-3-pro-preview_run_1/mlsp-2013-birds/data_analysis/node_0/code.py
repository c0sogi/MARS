import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")

    # Supplemental paths
    HISTOGRAM_PATH = os.path.join(
        INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    print("============================================")
    print("       EXPLORATORY DATA ANALYSIS REPORT     ")
    print("============================================")

    # 1. Load Metadata
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found.")
        return

    df_train = pd.read_csv(TRAIN_CSV)
    print(f"Training Set Size: {len(df_train)} samples")

    # Identify Label Columns
    label_cols = [c for c in df_train.columns if c.startswith("species_")]
    num_classes = len(label_cols)
    print(f"Number of Species (Classes): {num_classes}")
    print("")

    # ==========================================================
    # 2. TARGET VARIABLE ANALYSIS
    # ==========================================================
    print("SECTION 1: TARGET VARIABLE ANALYSIS")

    y_train = df_train[label_cols].values

    # Class Frequencies
    class_counts = y_train.sum(axis=0)
    total_labels = y_train.sum()

    # Label Cardinality (Average number of labels per sample)
    cardinality = total_labels / len(df_train)

    # Label Density (Cardinality / Num Classes)
    density = cardinality / num_classes

    # Check for empty samples
    empty_samples = np.sum(y_train.sum(axis=1) == 0)

    print(f"Label Cardinality (Avg labels/sample): {cardinality:.4f}")
    print(f"Label Density: {density:.4f}")
    print(
        f"Samples with No Labels: {empty_samples} ({empty_samples/len(df_train)*100:.2f}%)"
    )

    # Class Imbalance
    min_count = class_counts.min()
    max_count = class_counts.max()
    mean_count = class_counts.mean()

    print(f"Class Counts - Min: {min_count}, Max: {max_count}, Mean: {mean_count:.2f}")
    print(
        f"Imbalance Ratio (Max/Min): {max_count/min_count:.4f}"
        if min_count > 0
        else "Imbalance Ratio: Inf (Some classes have 0 samples)"
    )

    # Top 3 and Bottom 3 Classes
    class_indices = np.argsort(class_counts)
    top_3 = class_indices[-3:][::-1]
    bot_3 = class_indices[:3]

    print(f"Most Frequent Species Indices: {top_3} (Counts: {class_counts[top_3]})")
    print(f"Least Frequent Species Indices: {bot_3} (Counts: {class_counts[bot_3]})")
    print("")

    # ==========================================================
    # 3. AUDIO DATA ANALYSIS
    # ==========================================================
    print("SECTION 2: AUDIO DATA ANALYSIS")

    durations = []
    sample_rates = []
    channels = []
    bit_depths = []

    # We will check a subset if dataset is huge, but here N=208 is small enough to check all
    # Construct full paths
    # file_path in csv is relative to input dir, e.g., essential_data/src_wavs/...

    valid_audio_count = 0

    for idx, row in df_train.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            info = sf.info(full_path)
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels.append(info.channels)
            # subtype usually contains bit depth info (e.g., 'PCM_16')
            bit_depths.append(info.subtype)
            valid_audio_count += 1
        except Exception as e:
            pass

    if valid_audio_count > 0:
        durations = np.array(durations)
        print(f"Analyzed {valid_audio_count} Audio Files")
        print(
            f"Duration (sec) - Mean: {durations.mean():.4f}, Std: {durations.std():.4f}, Min: {durations.min():.4f}, Max: {durations.max():.4f}"
        )

        unique_sr = np.unique(sample_rates)
        print(f"Sampling Rates found: {unique_sr}")

        unique_ch = np.unique(channels)
        print(f"Channel counts found: {unique_ch} (1=Mono, 2=Stereo)")

        unique_bd = np.unique(bit_depths)
        print(f"Bit Depths/Subtypes found: {unique_bd}")

        if len(unique_sr) > 1 or len(unique_ch) > 1:
            print(
                "WARNING: Inconsistent audio formats detected. Resampling/Remixing required."
            )
        else:
            print("Audio format is consistent across the training set.")
    else:
        print("No valid audio files found to analyze.")
    print("")

    # ==========================================================
    # 4. IMAGE DATA ANALYSIS (SPECTROGRAMS)
    # ==========================================================
    print("SECTION 3: IMAGE (SPECTROGRAM) ANALYSIS")

    # Map rec_id to spectrogram filename.
    # The wav filename is in 'file_path'. The spectrograms are in supplemental_data/spectrograms
    # and have the same basename but .bmp extension.

    widths = []
    heights = []
    means = []
    stds = []

    valid_img_count = 0

    for idx, row in df_train.iterrows():
        wav_rel_path = row["file_path"]
        wav_basename = os.path.basename(wav_rel_path)
        bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"

        img_path = os.path.join(SPECTROGRAM_DIR, bmp_basename)

        if not os.path.exists(img_path):
            continue

        try:
            # Load as grayscale since spectrograms are usually single channel or mapped to colormap
            # The description says "pixel value for an image", implying grayscale intensity or mapped.
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            if len(img.shape) == 2:
                h, w = img.shape
                c = 1
            else:
                h, w, c = img.shape

            widths.append(w)
            heights.append(h)

            # Calculate pixel stats (normalize 0-255 to 0-1 for stats)
            img_norm = img.astype(float) / 255.0
            means.append(np.mean(img_norm))
            stds.append(np.std(img_norm))

            valid_img_count += 1
        except Exception:
            pass

    if valid_img_count > 0:
        print(f"Analyzed {valid_img_count} Spectrogram Images")

        widths = np.array(widths)
        heights = np.array(heights)

        print(
            f"Image Widths  - Mean: {widths.mean():.2f}, Min: {widths.min()}, Max: {widths.max()}"
        )
        print(
            f"Image Heights - Mean: {heights.mean():.2f}, Min: {heights.min()}, Max: {heights.max()}"
        )

        # Check aspect ratio
        aspect_ratios = widths / heights
        print(
            f"Aspect Ratios - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
        )

        # Pixel stats
        print(
            f"Pixel Values (0-1) - Global Mean: {np.mean(means):.4f}, Global Std: {np.mean(stds):.4f}"
        )
    else:
        print("No spectrogram images found in supplemental data.")
    print("")

    # ==========================================================
    # 5. TABULAR DATA ANALYSIS (HISTOGRAM FEATURES)
    # ==========================================================
    print("SECTION 4: TABULAR FEATURE ANALYSIS")

    # Load histogram_of_segments.txt
    # Format: rec_id, [histogram features...]
    # It might not have a header, or header might be 'rec_id,[histogram...]'
    # Let's inspect the file format by trying to read it.

    if os.path.exists(HISTOGRAM_PATH):
        try:
            # The file content example shows: rec_id,[histogram of segment features]
            # 0,0.00,0.00...
            # This implies a CSV structure.
            # We need to handle the header line carefully.

            # Read first line to check header
            with open(HISTOGRAM_PATH, "r") as f:
                first_line = f.readline()

            has_header = "rec_id" in first_line

            if has_header:
                df_hist = pd.read_csv(HISTOGRAM_PATH)
            else:
                df_hist = pd.read_csv(HISTOGRAM_PATH, header=None)
                # Rename first column to rec_id
                cols = ["rec_id"] + [f"feat_{i}" for i in range(df_hist.shape[1] - 1)]
                df_hist.columns = cols

            # Clean up column names if needed. The example header is "rec_id,[histogram of segment features]"
            # which might be parsed as a single column name or two.
            # If the parser read it correctly, the first column is rec_id.
            if df_hist.columns[0] != "rec_id":
                df_hist.rename(columns={df_hist.columns[0]: "rec_id"}, inplace=True)

            # Merge with training data to analyze features only for training samples
            df_merged = pd.merge(
                df_train[["rec_id"] + label_cols], df_hist, on="rec_id", how="inner"
            )

            feature_cols = [
                c for c in df_merged.columns if c not in ["rec_id"] + label_cols
            ]
            X_tab = df_merged[feature_cols].values
            y_tab = df_merged[label_cols].values

            print(f"Tabular Features Shape (Train): {X_tab.shape}")

            # Numerical Stats
            print(
                f"Feature Values - Min: {X_tab.min():.4f}, Max: {X_tab.max():.4f}, Mean: {X_tab.mean():.4f}"
            )

            # Sparsity
            sparsity = 1.0 - (np.count_nonzero(X_tab) / float(X_tab.size))
            print(f"Feature Sparsity (Zero Ratio): {sparsity:.4f}")

            # Check for NaNs
            nan_counts = np.isnan(X_tab).sum()
            print(f"Missing Values (NaNs) in Features: {nan_counts}")

            # Feature Importance using Random Forest
            print("Training lightweight Random Forest for feature importance...")
            rf = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            )
            rf.fit(X_tab, y_tab)

            # Aggregate importance across all estimators
            # Each estimator in MultiOutputClassifier is a classifier for one label
            importances = np.mean(
                [est.feature_importances_ for est in rf.estimators_], axis=0
            )

            top_5_idx = np.argsort(importances)[-5:][::-1]
            print("Top 5 Important Features (Indices):")
            for idx in top_5_idx:
                print(f"  Feature {feature_cols[idx]}: {importances[idx]:.4f}")

            # Correlation Analysis
            # Check for highly correlated features (Redundancy)
            # Since dimension is ~100, we can compute correlation matrix
            corr_matrix = np.corrcoef(X_tab, rowvar=False)
            # Upper triangle only
            upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            high_corr_pairs = np.where(np.abs(corr_matrix) > 0.90)
            high_corr_pairs = [(i, j) for i, j in zip(*high_corr_pairs) if i < j]

            print(
                f"Number of Highly Correlated Feature Pairs (>0.90): {len(high_corr_pairs)}"
            )

        except Exception as e:
            print(f"Error analyzing tabular data: {e}")
    else:
        print("Histogram features file not found.")
    print("")

    # ==========================================================
    # 6. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================================
    print("SECTION 5: META-FEATURE RELATIONSHIPS")

    # Relationship between Signal Energy (from spectrogram) and Label Count
    # Hypothesis: More birds -> Higher energy or more complex signal?

    # We need to align spectrogram stats with labels
    # We'll reuse the loop logic but store data for correlation

    energies = []
    label_counts = []

    for idx, row in df_train.iterrows():
        wav_rel_path = row["file_path"]
        wav_basename = os.path.basename(wav_rel_path)
        bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
        img_path = os.path.join(SPECTROGRAM_DIR, bmp_basename)

        if not os.path.exists(img_path):
            continue

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Simple proxy for energy: mean pixel intensity
            energy = np.mean(img)
            energies.append(energy)

            # Label count
            l_count = row[label_cols].sum()
            label_counts.append(l_count)

        except:
            pass

    if len(energies) > 10:
        corr = np.corrcoef(energies, label_counts)[0, 1]
        print(
            f"Correlation between Spectrogram Mean Intensity and Number of Species: {corr:.4f}"
        )
        if abs(corr) < 0.1:
            print(
                "  -> Interpretation: No linear relationship between signal loudness/brightness and number of species."
            )
        elif corr > 0:
            print(
                "  -> Interpretation: Louder/Brighter signals tend to have more species."
            )
        else:
            print(
                "  -> Interpretation: Louder/Brighter signals tend to have fewer species."
            )

    print("============================================")
    print("              EDA COMPLETE                  ")
    print("============================================")


if __name__ == "__main__":
    main()
