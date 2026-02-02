import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import scipy.io
import soundfile as sf
from collections import Counter, defaultdict

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def load_mat_file(path):
    """Safely load .mat file."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        return None


def main():
    # ==========================================
    # 1. Data Loading & Integrity
    # ==========================================
    if not os.path.exists(METADATA_FILE):
        print("Error: Metadata file not found.")
        return

    df = pd.read_csv(METADATA_FILE)

    # Parse labels
    df["parsed_labels"] = df["labels"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("DATA INTEGRITY")
    print(f"Analysis performed on {len(df)} training samples.")
    print("-" * 30)

    print("TARGET VARIABLE ANALYSIS")

    # Flatten all labels to get a list of all gesture instances
    all_gestures = []
    gesture_durations = defaultdict(list)

    for _, row in df.iterrows():
        for label in row["parsed_labels"]:
            gid = label["id"]
            all_gestures.append(gid)
            # Calculate duration in frames
            duration = label["end"] - label["begin"] + 1
            gesture_durations[gid].append(duration)

    gesture_counts = Counter(all_gestures)
    total_instances = len(all_gestures)
    unique_classes = sorted(gesture_counts.keys())

    print(f"Total Gesture Instances: {total_instances}")
    print(f"Number of Classes: {len(unique_classes)}")

    # Distribution stats
    counts = list(gesture_counts.values())
    print(f"Min Class Frequency: {min(counts)}")
    print(f"Max Class Frequency: {max(counts)}")
    print(f"Mean Class Frequency: {np.mean(counts):.4f}")

    # Class Balance
    # Calculate imbalance ratio (Max count / Min count)
    imbalance_ratio = max(counts) / min(counts) if counts else 0
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    print("Top 5 Frequent Classes (ID: Count):")
    for gid, count in gesture_counts.most_common(5):
        print(f"  Class {gid}: {count} ({count/total_instances*100:.2f}%)")

    print("-" * 30)

    # ==========================================
    # 3. Input Data Analysis (Modality-Specific)
    # ==========================================
    print("INPUT DATA ANALYSIS")

    # Containers for meta-analysis
    meta_stats = {
        "video_width": [],
        "video_height": [],
        "video_fps": [],
        "video_frame_count": [],
        "video_duration_sec": [],
        "audio_duration_sec": [],
        "audio_samplerate": [],
        "audio_channels": [],
        "skeleton_num_frames": [],
        "skeleton_valid": [],
    }

    # Pixel stats accumulators (Welford's algorithm or simple sum for approx)
    # We will use simple sum on a subset for speed
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    # Skeleton coordinate stats
    skel_coords = {"x": [], "y": [], "z": []}

    # Iterate through samples
    # To save time, we fully process headers for all, but sample pixels/skeleton data from a subset
    subset_indices = np.random.choice(df.index, size=min(30, len(df)), replace=False)

    for idx, row in df.iterrows():
        # Paths
        rgb_path = os.path.join(INPUT_DIR, row["rgb_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])
        data_path = os.path.join(INPUT_DIR, row["data_path"])

        # --- Video Analysis ---
        if os.path.exists(rgb_path):
            cap = cv2.VideoCapture(rgb_path)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                meta_stats["video_width"].append(w)
                meta_stats["video_height"].append(h)
                meta_stats["video_fps"].append(fps)
                meta_stats["video_frame_count"].append(fc)
                meta_stats["video_duration_sec"].append(fc / fps if fps > 0 else 0)

                # Pixel Stats (Only on subset)
                if idx in subset_indices:
                    # Read a middle frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fc // 2)
                    ret, frame = cap.read()
                    if ret:
                        # BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pixels = frame.reshape(-1, 3) / 255.0

                        # Sample pixels to reduce computation (10% of pixels)
                        sample_mask = np.random.random(pixels.shape[0]) < 0.1
                        sampled_pixels = pixels[sample_mask]

                        if len(sampled_pixels) > 0:
                            pixel_sum += np.sum(sampled_pixels, axis=0)
                            pixel_sq_sum += np.sum(sampled_pixels**2, axis=0)
                            pixel_count += len(sampled_pixels)

                cap.release()
            else:
                # Fill NaNs if failed
                meta_stats["video_width"].append(np.nan)

        # --- Audio Analysis ---
        if os.path.exists(audio_path):
            try:
                info = sf.info(audio_path)
                meta_stats["audio_duration_sec"].append(info.duration)
                meta_stats["audio_samplerate"].append(info.samplerate)
                meta_stats["audio_channels"].append(info.channels)
            except:
                meta_stats["audio_duration_sec"].append(np.nan)

        # --- Skeleton Analysis (Tabular/Structured) ---
        # Only process subset to save time
        if idx in subset_indices and os.path.exists(data_path):
            mat = load_mat_file(data_path)
            if mat and hasattr(mat, "Video"):
                video_struct = mat.Video
                if hasattr(video_struct, "NumFrames"):
                    meta_stats["skeleton_num_frames"].append(video_struct.NumFrames)

                # Try to extract coordinate ranges
                if hasattr(video_struct, "Frames"):
                    frames = video_struct.Frames
                    # Check a few frames
                    if isinstance(frames, (list, np.ndarray)) and len(frames) > 0:
                        # Sample 5 frames
                        f_indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
                        for f_idx in f_indices:
                            frame_data = frames[f_idx]
                            if hasattr(frame_data, "Skeleton") and hasattr(
                                frame_data.Skeleton, "WorldPosition"
                            ):
                                wp = frame_data.Skeleton.WorldPosition
                                # WorldPosition might be an object or array
                                try:
                                    if hasattr(wp, "X"):
                                        skel_coords["x"].append(wp.X)
                                        skel_coords["y"].append(wp.Y)
                                        skel_coords["z"].append(wp.Z)
                                except:
                                    pass

    # --- Video Report ---
    print("Image Data (RGB):")
    widths = np.array(meta_stats["video_width"])
    heights = np.array(meta_stats["video_height"])

    print(
        f"  Dimensions (Width): Mean={np.nanmean(widths):.1f}, Std={np.nanstd(widths):.1f}"
    )
    print(
        f"  Dimensions (Height): Mean={np.nanmean(heights):.1f}, Std={np.nanstd(heights):.1f}"
    )

    # Aspect Ratios
    ar = widths / heights
    print(f"  Aspect Ratio Mean: {np.nanmean(ar):.4f}")

    # Pixel Stats
    if pixel_count > 0:
        rgb_mean = pixel_sum / pixel_count
        rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))
        print(
            f"  Pixel Mean (Normalized 0-1): R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}"
        )
        print(
            f"  Pixel Std  (Normalized 0-1): R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}"
        )

    # --- Audio Report ---
    print("\nAudio Data:")
    durations = np.array(meta_stats["audio_duration_sec"])
    srs = np.array(meta_stats["audio_samplerate"])
    chans = np.array(meta_stats["audio_channels"])

    print(
        f"  Duration (sec): Mean={np.nanmean(durations):.4f}, Min={np.nanmin(durations):.4f}, Max={np.nanmax(durations):.4f}"
    )
    print(f"  Sample Rates: {np.unique(srs[~np.isnan(srs)])}")
    print(f"  Channels: {np.unique(chans[~np.isnan(chans)])}")

    # --- Skeleton Report ---
    print("\nSkeleton Data (Structured):")
    if skel_coords["x"]:
        x_vals = np.array(skel_coords["x"])
        y_vals = np.array(skel_coords["y"])
        z_vals = np.array(skel_coords["z"])
        print(
            f"  WorldPosition X (mm): Mean={np.mean(x_vals):.2f}, Min={np.min(x_vals):.2f}, Max={np.max(x_vals):.2f}"
        )
        print(
            f"  WorldPosition Y (mm): Mean={np.mean(y_vals):.2f}, Min={np.min(y_vals):.2f}, Max={np.max(y_vals):.2f}"
        )
        print(
            f"  WorldPosition Z (mm): Mean={np.mean(z_vals):.2f}, Min={np.min(z_vals):.2f}, Max={np.max(z_vals):.2f}"
        )
    else:
        print("  No valid skeleton coordinates extracted from subset.")

    print("-" * 30)

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Construct Meta-DataFrame for correlation
    # We need to align lists. Since we iterated rows, lists are aligned (with NaNs).
    meta_df = pd.DataFrame(
        {
            "video_duration": meta_stats["video_duration_sec"],
            "audio_duration": meta_stats["audio_duration_sec"],
            "frame_count": meta_stats["video_frame_count"],
            "num_gestures": df["num_gestures"],
        }
    ).dropna()

    print("\nStructured (Meta-Feature) Relationships:")

    # Correlation
    corr_matrix = meta_df.corr()
    print("  Correlation Matrix (Pearson):")
    print(corr_matrix.round(4).to_string())

    # Check Video-Audio Sync
    time_diff = np.abs(meta_df["video_duration"] - meta_df["audio_duration"])
    print(f"  Mean Absolute Diff (Video vs Audio Duration): {time_diff.mean():.4f} sec")
    if time_diff.mean() > 0.5:
        print("  WARNING: Significant discrepancy between video and audio durations.")

    print("\nUnstructured Relationships (Gesture Duration vs Class):")
    # Analyze if certain gestures are consistently longer/shorter
    # Calculate mean duration per class
    class_durations = {}
    for gid, durs in gesture_durations.items():
        class_durations[gid] = np.mean(durs)

    # Sort by duration
    sorted_durations = sorted(class_durations.items(), key=lambda x: x[1])

    print("  Shortest Average Gestures (Frames):")
    for gid, d in sorted_durations[:3]:
        print(f"    Class {gid}: {d:.2f} frames")

    print("  Longest Average Gestures (Frames):")
    for gid, d in sorted_durations[-3:]:
        print(f"    Class {gid}: {d:.2f} frames")

    # Feature Importance (Proxy)
    # Does the length of the video predict the number of gestures?
    # Simple correlation check
    corr_len_count = meta_df["video_duration"].corr(meta_df["num_gestures"])
    print(f"  Correlation (Video Duration vs Num Gestures): {corr_len_count:.4f}")


if __name__ == "__main__":
    main()
