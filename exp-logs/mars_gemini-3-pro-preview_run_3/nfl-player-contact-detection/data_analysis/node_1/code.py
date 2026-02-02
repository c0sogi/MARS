import pandas as pd
import numpy as np
import os
import cv2
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
SEED = 42


def set_seed(seed):
    np.random.seed(seed)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    target_col = "contact"
    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Target Variable: {target_col}")
    print(f"Distribution:\n{counts.to_string()}")

    pos_ratio = counts.get(1, 0) / total
    neg_ratio = counts.get(0, 0) / total

    print(f"Class Balance (Positive): {pos_ratio:.4f}")
    print(f"Class Balance (Negative): {neg_ratio:.4f}")

    imbalance_ratio = counts.get(0, 0) / max(1, counts.get(1, 0))
    print(f"Imbalance Ratio (Neg/Pos): {imbalance_ratio:.4f}")
    print("-" * 30)


def analyze_tabular_tracking(df_train_ids):
    print("INPUT DATA ANALYSIS: TABULAR (TRACKING)")

    # Load tracking data
    # To save memory/time, we could filter, but dataset is manageable (~1.2M rows)
    df_tracking = pd.read_csv(TRACKING_PATH)

    # Filter to training set games only
    train_game_plays = set(df_train_ids["game_play"].unique())
    df_tracking = df_tracking[df_tracking["game_play"].isin(train_game_plays)].copy()

    print(f"Tracking Data Shape (Filtered): {df_tracking.shape}")

    # Numerical Analysis
    num_cols = [
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]
    print("\nNumerical Features Analysis:")
    for col in num_cols:
        if col in df_tracking.columns:
            stats = df_tracking[col].describe()
            q1 = stats["25%"]
            q3 = stats["75%"]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (
                (df_tracking[col] < lower_bound) | (df_tracking[col] > upper_bound)
            ).sum()

            print(
                f"{col}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}, Outliers={outliers}"
            )

    # Categorical Analysis
    cat_cols = ["position", "team"]
    print("\nCategorical Features Analysis:")
    for col in cat_cols:
        if col in df_tracking.columns:
            unique_vals = df_tracking[col].nunique()
            print(f"{col}: Cardinality={unique_vals}")
            if unique_vals > 50:
                print(f"  Flag: High cardinality ({unique_vals})")

            # Check for rare labels
            counts = df_tracking[col].value_counts(normalize=True)
            rare = counts[counts < 0.01]
            if not rare.empty:
                print(f"  Flag: {len(rare)} rare categories (<1%) found.")

    # Missing Values
    print("\nMissing Values:")
    missing = df_tracking.isnull().sum()
    missing_pct = (missing / len(df_tracking)) * 100
    for col in df_tracking.columns:
        if missing[col] > 0:
            print(f"{col}: {missing[col]} ({missing_pct[col]:.4f}%)")

    print("-" * 30)
    return df_tracking


def analyze_tabular_helmets(df_train_ids):
    print("INPUT DATA ANALYSIS: TABULAR (HELMETS)")

    # Load helmets data (only first 1M rows to save time if file is huge, but here we load and filter)
    # The file is large (~3.4M), let's read distinct game_plays first or just read all if memory allows.
    # 3.4M rows is fine for 220GB RAM.
    df_helmets = pd.read_csv(HELMETS_PATH)

    train_game_plays = set(df_train_ids["game_play"].unique())
    df_helmets = df_helmets[df_helmets["game_play"].isin(train_game_plays)].copy()

    print(f"Helmets Data Shape (Filtered): {df_helmets.shape}")

    # Bounding Box Analysis
    bbox_cols = ["left", "width", "top", "height"]
    print("\nBounding Box Dimensions:")
    for col in bbox_cols:
        if col in df_helmets.columns:
            mean_val = df_helmets[col].mean()
            std_val = df_helmets[col].std()
            print(f"{col}: Mean={mean_val:.4f}, Std={std_val:.4f}")

    # Missing Values
    missing = df_helmets.isnull().sum().sum()
    print(f"\nTotal Missing Values in Helmets Data: {missing}")
    print("-" * 30)


def analyze_video(df_train):
    print("INPUT DATA ANALYSIS: VIDEO")

    # Sample a few videos
    # We look at Sideline views as a representative sample
    sample_videos = df_train["video_path_sideline"].dropna().unique()
    if len(sample_videos) > 5:
        sample_videos = np.random.choice(sample_videos, 5, replace=False)

    widths = []
    heights = []
    fps_list = []
    pixel_means = []
    pixel_stds = []

    print(f"Analyzing {len(sample_videos)} sample videos...")

    for rel_path in sample_videos:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            continue

        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        widths.append(w)
        heights.append(h)
        fps_list.append(fps)

        # Read one frame for pixel stats
        ret, frame = cap.read()
        if ret:
            pixel_means.append(np.mean(frame))
            pixel_stds.append(np.std(frame))

        cap.release()

    if widths:
        print(
            f"Widths: Mean={np.mean(widths):.1f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"Heights: Mean={np.mean(heights):.1f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(f"FPS: Mean={np.mean(fps_list):.4f}")
        print(f"Pixel Mean (Global est.): {np.mean(pixel_means):.4f}")
        print(f"Pixel Std (Global est.): {np.mean(pixel_stds):.4f}")

        # Aspect Ratio
        ar = np.array(widths) / np.array(heights)
        print(f"Aspect Ratio Mean: {np.mean(ar):.4f}")
    else:
        print("No videos could be analyzed.")

    print("-" * 30)


def analyze_relationships(df_train, df_tracking):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Structured Relationships
    # We need to link contact labels to tracking data.
    # df_train has: game_play, step, nfl_player_id_1, nfl_player_id_2, contact
    # df_tracking has: game_play, step, nfl_player_id, x_position, y_position, speed, etc.

    # Sample for efficiency
    sample_size = min(50000, len(df_train))
    df_sample = df_train.sample(n=sample_size, random_state=SEED).copy()

    # Filter out Ground contacts for feature importance (since Ground has no tracking data)
    df_sample_pp = df_sample[df_sample["nfl_player_id_2"] != "G"].copy()

    # Convert IDs to numeric for merging
    df_sample_pp["nfl_player_id_1"] = pd.to_numeric(df_sample_pp["nfl_player_id_1"])
    df_sample_pp["nfl_player_id_2"] = pd.to_numeric(df_sample_pp["nfl_player_id_2"])

    # Merge Player 1 info
    df_merged = pd.merge(
        df_sample_pp,
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="inner",
        suffixes=("", "_p1"),
    )

    # Merge Player 2 info
    df_merged = pd.merge(
        df_merged,
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="inner",
        suffixes=("", "_p2"),
    )

    if len(df_merged) == 0:
        print(
            "Insufficient intersection between labels and tracking data for relationship analysis."
        )
        return

    # Feature Engineering
    # Euclidean Distance
    df_merged["distance_p1_p2"] = np.sqrt(
        (df_merged["x_position_p1"] - df_merged["x_position_p2"]) ** 2
        + (df_merged["y_position_p1"] - df_merged["y_position_p2"]) ** 2
    )

    # Correlation Analysis
    features = [
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
        "distance_p1_p2",
    ]
    corr_matrix = df_merged[features + ["contact"]].corr()

    print("Top Correlations with Target (contact):")
    target_corr = (
        corr_matrix["contact"].drop("contact").sort_values(ascending=False, key=abs)
    )
    print(target_corr.to_string())

    # Redundancy Check
    print("\nRedundant Features (Correlation > 0.90):")
    feature_corr = df_merged[features].corr()
    redundant_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if abs(feature_corr.iloc[i, j]) > 0.90:
                redundant_pairs.append(
                    (features[i], features[j], feature_corr.iloc[i, j])
                )

    if redundant_pairs:
        for f1, f2, val in redundant_pairs:
            print(f"{f1} - {f2}: {val:.4f}")
    else:
        print("None found among selected features.")

    # Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest):")
    X = df_merged[features].fillna(0)
    y = df_merged["contact"]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print(importances.to_string())

    # 2. Unstructured/Meta Relationships
    # Relationship between 'step' (time in play) and contact
    # We use the original sample (including Ground contacts)
    print("\nMeta-Feature Relationship (Step vs Contact):")
    # Bin step into intervals
    df_sample["step_bin"] = pd.cut(df_sample["step"], bins=5)
    step_contact_rate = df_sample.groupby("step_bin", observed=True)["contact"].mean()
    print("Contact Rate by Step Interval:")
    print(step_contact_rate.to_string())

    print("-" * 30)


def main():
    set_seed(SEED)

    # Load Metadata
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Error: Metadata file not found at {TRAIN_META_PATH}")
        return

    df_train = pd.read_csv(TRAIN_META_PATH)

    # 1. Target Analysis
    analyze_target(df_train)

    # 2. Tabular Analysis (Tracking)
    df_tracking = analyze_tabular_tracking(df_train)

    # 3. Tabular Analysis (Helmets)
    analyze_tabular_helmets(df_train)

    # 4. Video Analysis
    analyze_video(df_train)

    # 5. Relationships
    analyze_relationships(df_train, df_tracking)


if __name__ == "__main__":
    main()
