import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Constants
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
SEED = 42

# Set seeds for reproducibility
np.random.seed(SEED)


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_target(df):
    print_section("Target Variable Analysis")
    target_col = "contact"

    counts = df[target_col].value_counts()
    total = len(df)

    print(f"Target Variable: '{target_col}'")
    print(f"Total Samples: {total}")
    for label, count in counts.items():
        ratio = count / total
        print(f"Class {label}: {count} ({ratio:.4f})")

    # Imbalance check
    if len(counts) > 1:
        imbalance_ratio = counts.max() / counts.min()
        print(f"Imbalance Ratio (Maj/Min): {imbalance_ratio:.4f}")
    else:
        print("Only one class present.")


def analyze_video_data(df, sample_size=20):
    print_section("Video Data Analysis (Image Modality)")

    # We have three views. Let's sample from the Sideline view as a representative.
    # Filter for unique videos to avoid sampling the same video multiple times
    unique_videos = df[["video_path_sideline"]].drop_duplicates()

    if len(unique_videos) > sample_size:
        sampled_videos = unique_videos.sample(n=sample_size, random_state=SEED)[
            "video_path_sideline"
        ].values
    else:
        sampled_videos = unique_videos["video_path_sideline"].values

    print(f"Analyzing a sample of {len(sampled_videos)} videos...")

    widths = []
    heights = []
    aspect_ratios = []
    frame_counts = []
    fps_list = []
    pixel_means = []
    pixel_stds = []

    for v_path in sampled_videos:
        if not os.path.exists(v_path):
            continue

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        widths.append(w)
        heights.append(h)
        if h > 0:
            aspect_ratios.append(w / h)
        frame_counts.append(count)
        fps_list.append(fps)

        # Read a middle frame for pixel stats
        cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
        ret, frame = cap.read()
        if ret:
            # Normalize to 0-1 for stats
            img_norm = frame.astype(float) / 255.0
            pixel_means.append(np.mean(img_norm))
            pixel_stds.append(np.std(img_norm))

        cap.release()

    if widths:
        print(
            f"Widths: Mean={np.mean(widths):.2f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"Heights: Mean={np.mean(heights):.2f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(f"Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}")
        print(
            f"Frame Counts: Mean={np.mean(frame_counts):.2f}, Range=[{np.min(frame_counts)}, {np.max(frame_counts)}]"
        )
        print(f"FPS: Mean={np.mean(fps_list):.2f}")
        print(
            f"Pixel Intensity (Norm 0-1): Mean={np.mean(pixel_means):.4f}, Std={np.mean(pixel_stds):.4f}"
        )
        print("Channel Count: 3 (RGB/BGR assumed from cv2)")
    else:
        print("No videos could be processed.")


def analyze_tabular_features(df_meta, df_tracking):
    print_section("Tabular Feature Analysis & Engineering")

    # 1. Merge Strategy
    # We need to merge tracking data for player 1 and player 2.
    # df_meta has: game_play, step, nfl_player_id_1, nfl_player_id_2
    # df_tracking has: game_play, step, nfl_player_id, x_position, y_position, ...

    # To save memory/time, sample the metadata first
    SAMPLE_SIZE = 50000
    if len(df_meta) > SAMPLE_SIZE:
        df_sample = df_meta.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        df_sample = df_meta.copy()

    print(f"Using a sample of {len(df_sample)} interactions for tabular analysis.")

    # Prepare tracking data
    # Ensure keys match types
    df_sample["game_play"] = df_sample["game_play"].astype(str)
    df_sample["step"] = df_sample["step"].astype(int)
    df_sample["nfl_player_id_1"] = df_sample["nfl_player_id_1"].astype(int)

    # Handle 'G' in player 2
    # We separate Player-Player interactions from Player-Ground
    df_pp = df_sample[df_sample["nfl_player_id_2"] != "G"].copy()
    df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

    print(f"Player-Player Interactions in sample: {len(df_pp)}")

    # Merge Player 1
    df_merged = pd.merge(
        df_pp,
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_p1"),
    )
    # Rename cols for p1
    p1_cols = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
    ]
    rename_dict_p1 = {c: f"{c}_p1" for c in p1_cols}
    df_merged.rename(columns=rename_dict_p1, inplace=True)

    # Merge Player 2
    df_merged = pd.merge(
        df_merged,
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_p2"),
    )
    # Rename cols for p2 (the merge might have handled suffixes, but let's ensure specific cols)
    # If columns collided, pandas adds suffixes.
    # We specifically want the tracking columns for p2.
    # Since we dropped 'nfl_player_id' from right side in logic implicitly, let's check columns.

    # Calculate derived features
    # Distance
    if "x_position_p1" in df_merged.columns and "x_position_y" in df_merged.columns:
        # Note: merge default suffix is _x, _y if collision.
        # Let's be safer with column selection.
        pass

    # Let's re-do merge cleanly
    track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
    ]
    df_track_sub = df_tracking[track_cols]

    df_pp = df_sample[df_sample["nfl_player_id_2"] != "G"].copy()
    df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

    # Merge P1
    df_m = df_pp.merge(
        df_track_sub,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    df_m = df_m.rename(
        columns={
            c: f"{c}_p1"
            for c in ["x_position", "y_position", "speed", "acceleration", "direction"]
        }
    )
    df_m = df_m.drop(columns=["nfl_player_id"])

    # Merge P2
    df_m = df_m.merge(
        df_track_sub,
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    df_m = df_m.rename(
        columns={
            c: f"{c}_p2"
            for c in ["x_position", "y_position", "speed", "acceleration", "direction"]
        }
    )
    df_m = df_m.drop(columns=["nfl_player_id"])

    # Drop rows where tracking data is missing
    before_drop = len(df_m)
    df_m = df_m.dropna(subset=["x_position_p1", "x_position_p2"])
    print(
        f"Rows with valid tracking data for both players: {len(df_m)} (dropped {before_drop - len(df_m)})"
    )

    # Feature Engineering
    df_m["distance"] = np.sqrt(
        (df_m["x_position_p1"] - df_m["x_position_p2"]) ** 2
        + (df_m["y_position_p1"] - df_m["y_position_p2"]) ** 2
    )
    df_m["speed_diff"] = np.abs(df_m["speed_p1"] - df_m["speed_p2"])
    df_m["acc_diff"] = np.abs(df_m["acceleration_p1"] - df_m["acceleration_p2"])

    # Numerical Analysis
    num_features = [
        "distance",
        "speed_p1",
        "speed_p2",
        "speed_diff",
        "acceleration_p1",
        "acceleration_p2",
    ]

    print("\nNumerical Feature Stats (Player-Player):")
    for feat in num_features:
        col_data = df_m[feat]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        outliers = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()

        print(
            f"{feat}: Mean={col_data.mean():.4f}, Std={col_data.std():.4f}, Min={col_data.min():.4f}, Max={col_data.max():.4f}, Outliers={outliers}"
        )

    # Correlation
    print("\nCorrelations with Target (Player-Player):")
    corr_matrix = df_m[num_features + ["contact"]].corr()
    print(corr_matrix["contact"].sort_values(ascending=False))

    # Collinearity
    print("\nChecking for Redundancy (Correlation > 0.90):")
    high_corr_pairs = []
    features_check = num_features
    for i in range(len(features_check)):
        for j in range(i + 1, len(features_check)):
            c = df_m[features_check[i]].corr(df_m[features_check[j]])
            if abs(c) > 0.9:
                high_corr_pairs.append((features_check[i], features_check[j], c))

    if high_corr_pairs:
        for f1, f2, c in high_corr_pairs:
            print(f"High Correlation: {f1} vs {f2} ({c:.4f})")
    else:
        print("No highly collinear pairs found.")

    return df_m


def analyze_feature_importance(df):
    print_section("Feature Importance (Random Forest)")

    features = [
        "distance",
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
        "direction_p1",
        "direction_p2",
    ]
    target = "contact"

    # Prepare data
    X = df[features].fillna(
        0
    )  # Simple imputation for safety, though we dropped NaNs earlier
    y = df[target]

    # Train simple RF
    clf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
    )
    clf.fit(X, y)

    # Importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top Feature Importances:")
    for i in range(len(features)):
        print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

    print(
        "\nNote: 'distance' is expected to be the dominant feature for contact detection."
    )


def main():
    # 1. Load Data
    print("Loading Metadata...")
    if not os.path.exists(TRAIN_METADATA_PATH):
        print(f"Error: {TRAIN_METADATA_PATH} not found.")
        return

    df_train = pd.read_csv(TRAIN_METADATA_PATH)

    # 2. Target Analysis
    analyze_target(df_train)

    # 3. Video Analysis
    # Only analyze if video files exist (local environment check)
    # We assume standard path structure from description
    analyze_video_data(df_train, sample_size=10)

    # 4. Tabular/Tracking Analysis
    print("\nLoading Tracking Data...")
    if os.path.exists(TRACKING_PATH):
        df_tracking = pd.read_csv(TRACKING_PATH)

        # Perform merge and feature engineering
        df_features = analyze_tabular_features(df_train, df_tracking)

        # 5. Feature Importance
        if not df_features.empty:
            analyze_feature_importance(df_features)
    else:
        print(f"Tracking data not found at {TRACKING_PATH}. Skipping tabular analysis.")


if __name__ == "__main__":
    main()
