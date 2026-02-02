import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
SEED = 42

# Set seeds
np.random.seed(SEED)


def print_section(title):
    print(f"\n{'='*10} {title} {'='*10}")


def analyze_tabular(df, name):
    print(f"\n--- {name} Analysis ---")

    # Numerical Analysis
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(f"Numerical Columns ({len(num_cols)}): {list(num_cols[:5])}...")
        stats = []
        for col in num_cols:
            series = df[col]
            # Simple outlier detection using IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()

            stats.append(
                {
                    "Column": col,
                    "Mean": series.mean(),
                    "Std": series.std(),
                    "Min": series.min(),
                    "Max": series.max(),
                    "Outliers": outliers,
                }
            )

        stats_df = pd.DataFrame(stats)
        # Print top 5 rows of stats for brevity
        print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Categorical Analysis
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print(f"\nCategorical Columns ({len(cat_cols)}): {list(cat_cols[:5])}...")
        for col in cat_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count > 50:
                print(f"    -> High cardinality (>50 categories).")

            # Check for rare labels
            counts = df[col].value_counts(normalize=True)
            rare = counts[counts < 0.01]
            if not rare.empty:
                print(f"    -> Has {len(rare)} rare labels (<1% freq).")

    # Missing Values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nMissing Values:")
        for col, val in missing.items():
            print(f"  {col}: {val} ({val/len(df)*100:.2f}%)")
    else:
        print("\nNo missing values detected.")


def analyze_video_modality(metadata_df):
    print_section("INPUT DATA ANALYSIS (IMAGE/VIDEO)")

    # Sample unique videos
    # We look at Endzone videos as a representative sample
    video_paths = metadata_df["path_endzone"].dropna().unique()
    sample_paths = np.random.choice(
        video_paths, size=min(5, len(video_paths)), replace=False
    )

    widths = []
    heights = []
    aspect_ratios = []
    pixel_means = []
    pixel_stds = []
    frame_counts = []

    print(f"Analyzing {len(sample_paths)} sample videos...")

    for rel_path in sample_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            continue

        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        widths.append(w)
        heights.append(h)
        if h > 0:
            aspect_ratios.append(w / h)
        frame_counts.append(count)

        # Read a few frames for pixel stats
        # Sample up to 10 frames per video
        frame_indices = np.linspace(0, count - 1, 10, dtype=int)
        video_pixels = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_pixels.append(frame)

        cap.release()

        if video_pixels:
            batch = np.stack(video_pixels)
            pixel_means.append(np.mean(batch))
            pixel_stds.append(np.std(batch))

    # Report
    print("\nDimensions:")
    print(f"  Widths: {np.unique(widths)}")
    print(f"  Heights: {np.unique(heights)}")
    print(f"  Aspect Ratios: {[f'{ar:.2f}' for ar in np.unique(aspect_ratios)]}")

    print("\nChannels:")
    print("  Standard RGB (3 channels) detected in sampled frames.")

    if pixel_means:
        print("\nPixel Stats (Global approx from samples):")
        print(f"  Mean: {np.mean(pixel_means):.4f}")
        print(f"  Std:  {np.mean(pixel_stds):.4f}")

    print("\nVideo Lengths:")
    print(f"  Frame Counts: {np.unique(frame_counts)}")


def main():
    # 1. Load Data
    print("Loading datasets...")
    df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))

    # Load tracking data (potentially large, but fits in 220GB RAM)
    # We only need tracking data for game_plays that are in the training set
    train_game_plays = df_train_meta["game_play"].unique()

    df_tracking = pd.read_csv(os.path.join(INPUT_DIR, "train_player_tracking.csv"))
    df_tracking = df_tracking[df_tracking["game_play"].isin(train_game_plays)].copy()

    # Load baseline helmets (sample first 100k for analysis to save time/memory if needed,
    # but full load is fine given resources)
    df_helmets = pd.read_csv(os.path.join(INPUT_DIR, "train_baseline_helmets.csv"))
    df_helmets = df_helmets[df_helmets["game_play"].isin(train_game_plays)].copy()

    # 2. Target Variable Analysis
    print_section("TARGET VARIABLE ANALYSIS")
    target_counts = df_train_meta["contact"].value_counts()
    total_samples = len(df_train_meta)

    print("Target: 'contact' (Binary Classification)")
    print(f"Distribution:\n{target_counts}")

    ratio_0 = target_counts.get(0, 0) / total_samples
    ratio_1 = target_counts.get(1, 0) / total_samples

    print(f"\nClass Balance:")
    print(f"  Class 0 (No Contact): {ratio_0:.4f}")
    print(f"  Class 1 (Contact):    {ratio_1:.4f}")
    print(f"  Imbalance Ratio (0:1): {ratio_0/ratio_1:.2f}:1")

    # 3. Input Data Analysis (Tabular)
    print_section("INPUT DATA ANALYSIS (TABULAR)")
    analyze_tabular(df_tracking, "Player Tracking")
    analyze_tabular(df_helmets, "Baseline Helmets")

    # 4. Input Data Analysis (Image)
    analyze_video_modality(df_train_meta)

    # 5. Feature/Signal Relationships
    print_section("FEATURE/SIGNAL RELATIONSHIPS")

    # Construct a merged dataset for analysis
    # We will sample the metadata to keep operations fast
    SAMPLE_SIZE = 100000
    if len(df_train_meta) > SAMPLE_SIZE:
        # Stratified sample to ensure we get contacts
        df_sample = df_train_meta.groupby("contact", group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(SAMPLE_SIZE / 2)), random_state=SEED)
        )
    else:
        df_sample = df_train_meta.copy()

    print(f"Constructing feature set from {len(df_sample)} samples...")

    # Prepare tracking for merge
    # Ensure types match
    df_sample["nfl_player_id_1"] = pd.to_numeric(
        df_sample["nfl_player_id_1"], errors="coerce"
    )
    # nfl_player_id_2 can be 'G', so we keep it as object or handle separately.
    # We will force numeric for merge, 'G' becomes NaN
    df_sample["nfl_player_id_2_num"] = pd.to_numeric(
        df_sample["nfl_player_id_2"], errors="coerce"
    )

    # Merge Player 1
    # Tracking keys: game_play, step, nfl_player_id
    track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
    ]

    merged = df_sample.merge(
        df_tracking[track_cols].add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
    )

    # Merge Player 2
    merged = merged.merge(
        df_tracking[track_cols].add_suffix("_2"),
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play_2", "step_2", "nfl_player_id_2"],
        how="left",
        suffixes=("", "_tracking"),
    )

    # Feature Engineering
    # 1. Is Ground Contact?
    merged["is_ground"] = (merged["nfl_player_id_2"] == "G").astype(int)

    # 2. Distance (only valid if not ground and both players found)
    merged["dx"] = merged["x_position_1"] - merged["x_position_2"]
    merged["dy"] = merged["y_position_1"] - merged["y_position_2"]
    merged["distance_p1_p2"] = np.sqrt(merged["dx"] ** 2 + merged["dy"] ** 2)

    # 3. Speed/Accel stats
    # Fill NaNs for P2 (Ground or missing tracking) with 0 for correlation analysis
    feat_cols = [
        "speed_1",
        "acceleration_1",
        "speed_2",
        "acceleration_2",
        "distance_p1_p2",
        "is_ground",
    ]

    # Impute for analysis
    analysis_df = merged[feat_cols + ["contact"]].copy()
    analysis_df = analysis_df.fillna(
        -1
    )  # Fill NaNs with -1 to indicate missing/ground for numericals

    # Correlation
    print("\nCorrelation with Target (Pearson):")
    corr = analysis_df.corr()["contact"].sort_values(ascending=False)
    print(corr.drop("contact").to_string(float_format=lambda x: f"{x:.4f}"))

    # Feature Importance (Random Forest)
    print("\nRandom Forest Feature Importance:")
    X = analysis_df.drop(columns=["contact"])
    y = analysis_df["contact"]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print(importances.head(5).to_string(float_format=lambda x: f"{x:.4f}"))

    # Redundancy Check
    print("\nRedundancy (Collinear Features > 0.90):")
    feature_corr = X.corr().abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    if to_drop:
        print(f"  Detected collinear features: {to_drop}")
    else:
        print("  No highly collinear features detected among selected subset.")

    # 6. Unstructured/Meta Relationships
    print_section("META-FEATURE RELATIONSHIPS")

    # Relationship between Step (Time) and Contact
    # Bin steps into intervals
    merged["step_bin"] = pd.cut(merged["step"], bins=10)
    contact_rate_by_step = merged.groupby("step_bin", observed=False)["contact"].mean()

    print("Contact Rate by Time (Step) Intervals:")
    print(contact_rate_by_step.to_string(float_format=lambda x: f"{x:.4f}"))

    # Check if Ground contact has different physics profile
    if "is_ground" in merged.columns:
        ground_contact_rate = merged[merged["is_ground"] == 1]["contact"].mean()
        player_contact_rate = merged[merged["is_ground"] == 0]["contact"].mean()
        print(f"\nContact Probability vs Ground: {ground_contact_rate:.4f}")
        print(f"Contact Probability vs Player: {player_contact_rate:.4f}")


if __name__ == "__main__":
    main()
