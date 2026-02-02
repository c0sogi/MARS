import os
import sys
import random
import numpy as np
import pandas as pd
import scipy.io
import cv2
import soundfile as sf
import warnings
from collections import defaultdict, Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42

# Label Map for reference (from prompt)
LABEL_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_mat_data(rel_path):
    """Robustly loads mat file and extracts Labels and Skeleton info."""
    full_path = os.path.join(INPUT_DIR, rel_path)
    try:
        mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return None, None

        video = mat["Video"]
        labels_data = []

        # Extract Labels
        if hasattr(video, "Labels"):
            raw_labels = video.Labels
            # Handle single object vs array
            if not isinstance(raw_labels, np.ndarray):
                raw_labels = [raw_labels]
            elif raw_labels.size == 1:
                raw_labels = [raw_labels.item()]

            for l in raw_labels:
                if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                    name = l.Name
                    if name in LABEL_MAP:
                        labels_data.append(
                            {
                                "id": LABEL_MAP[name],
                                "name": name,
                                "begin": l.Begin,
                                "end": l.End,
                                "duration": l.End - l.Begin + 1,
                            }
                        )

        # Extract Skeleton Stats (Sample first frame if available)
        skeleton_stats = {}
        if hasattr(video, "Frames"):
            frames = video.Frames
            # Check if frames is populated
            if isinstance(frames, np.ndarray) and frames.size > 0:
                # Take a sample frame (middle)
                sample_idx = len(frames) // 2
                skel_frame = frames[sample_idx]

                # The structure of Skeleton is nested: Frame -> Skeleton -> WorldPosition
                # Based on prompt: Frame has Skeleton struct.
                if hasattr(skel_frame, "Skeleton"):
                    skel = skel_frame.Skeleton
                    # Skeleton might be an array if multiple users, usually 1 for this dataset
                    if isinstance(skel, np.ndarray) and skel.size > 0:
                        skel = skel[0]  # Take first user

                    if hasattr(skel, "WorldPosition"):
                        # WorldPosition is likely an array of structs or a struct of arrays
                        # Prompt says: JointsType, WorldPosition...
                        # Let's assume we can iterate joints
                        pass
                        # Due to complex nesting variability in MAT files,
                        # we will infer dimensions from the 'PixelPosition' or 'WorldPosition' if accessible directly
                        # but strictly following prompt, let's just count joints if possible.
                        # Instead of deep parsing which is fragile, we'll check MaxDepth from Video struct

        max_depth = getattr(video, "MaxDepth", 0)

        return labels_data, max_depth

    except Exception:
        return None, None


def analyze_video_properties(rel_path):
    full_path = os.path.join(INPUT_DIR, rel_path)
    try:
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        return {"width": width, "height": height, "fps": fps, "frames": frame_count}
    except:
        return None


def get_pixel_stats(rel_path_list, num_samples=20, frames_per_sample=5):
    """Estimates mean and std of RGB video data."""
    sampled_paths = np.random.choice(
        rel_path_list, min(len(rel_path_list), num_samples), replace=False
    )

    pixel_sums = np.zeros(3)
    pixel_sq_sums = np.zeros(3)
    total_pixels = 0

    for p in sampled_paths:
        full_path = os.path.join(INPUT_DIR, p)
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        indices = np.linspace(0, total_frames - 1, frames_per_sample).astype(int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Frame is BGR
                frame = frame / 255.0
                pixel_sums += np.sum(frame, axis=(0, 1))
                pixel_sq_sums += np.sum(frame**2, axis=(0, 1))
                total_pixels += frame.shape[0] * frame.shape[1]
        cap.release()

    if total_pixels == 0:
        return np.zeros(3), np.zeros(3)

    mean = pixel_sums / total_pixels
    # std = sqrt(E[x^2] - (E[x])^2)
    std = np.sqrt((pixel_sq_sums / total_pixels) - (mean**2))

    # Convert BGR to RGB for reporting
    return mean[::-1], std[::-1]


def analyze_audio_properties(rel_path):
    full_path = os.path.join(INPUT_DIR, rel_path)
    try:
        info = sf.info(full_path)
        return {
            "duration": info.duration,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
        }
    except:
        return None


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print("Error: Metadata file not found.")
        return

    df = pd.read_csv(METADATA_PATH)

    # Parse labels from string "1,2,3" to list [1, 2, 3]
    df["labels_list"] = df["labels"].apply(
        lambda x: [int(i) for i in str(x).split(",")] if pd.notna(x) and x != "" else []
    )

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    all_labels = [lbl for sublist in df["labels_list"] for lbl in sublist]
    label_counts = Counter(all_labels)
    total_gestures = len(all_labels)

    # Sequence Lengths
    seq_lengths = df["labels_list"].apply(len)

    print("TARGET VARIABLE ANALYSIS")
    print(f"Total Gestures: {total_gestures}")
    print(f"Unique Classes: {len(label_counts)}")
    print(
        f"Most Frequent Class: {ID_TO_NAME.get(label_counts.most_common(1)[0][0], 'Unknown')} (Count: {label_counts.most_common(1)[0][1]})"
    )
    print(
        f"Least Frequent Class: {ID_TO_NAME.get(label_counts.most_common()[-1][0], 'Unknown')} (Count: {label_counts.most_common()[-1][1]})"
    )

    # Class Balance Ratio (Max / Min)
    max_freq = label_counts.most_common(1)[0][1]
    min_freq = label_counts.most_common()[-1][1]
    balance_ratio = max_freq / min_freq if min_freq > 0 else 0
    print(f"Class Balance Ratio (Max/Min): {balance_ratio:.4f}")

    print(f"Sequence Length Mean: {seq_lengths.mean():.4f}")
    print(f"Sequence Length Std: {seq_lengths.std():.4f}")
    print(f"Sequence Length Min: {seq_lengths.min()}")
    print(f"Sequence Length Max: {seq_lengths.max()}")
    print("-" * 30)

    # ==========================================
    # 3. Input Data Analysis
    # ==========================================

    # --- Video Analysis ---
    # We scan a subset for dimensions to ensure consistency and sample pixels
    video_widths = []
    video_heights = []
    video_fps = []

    # Check first 5 videos for dimensions consistency
    for _, row in df.head(5).iterrows():
        if pd.notna(row["color_path"]):
            props = analyze_video_properties(row["color_path"])
            if props:
                video_widths.append(props["width"])
                video_heights.append(props["height"])
                video_fps.append(props["fps"])

    # Pixel Stats (RGB)
    rgb_mean, rgb_std = get_pixel_stats(df["color_path"].dropna().tolist())

    print("INPUT DATA ANALYSIS: VIDEO (RGB)")
    if video_widths:
        print(
            f"Dimensions (Width x Height): {video_widths[0]} x {video_heights[0]} (Checked first 5 samples)"
        )
        print(f"Frame Rate: {video_fps[0]} fps")
    else:
        print("Dimensions: Could not determine.")

    print(
        f"Pixel Mean (RGB): [{rgb_mean[0]:.4f}, {rgb_mean[1]:.4f}, {rgb_mean[2]:.4f}]"
    )
    print(f"Pixel Std (RGB):  [{rgb_std[0]:.4f}, {rgb_std[1]:.4f}, {rgb_std[2]:.4f}]")
    print("-" * 30)

    # --- Audio Analysis ---
    audio_durations = []
    audio_fss = []
    audio_channels = []

    # Sample 20 audio files
    sample_audio_paths = (
        df["audio_path"].dropna().sample(min(20, len(df)), random_state=SEED)
    )
    for p in sample_audio_paths:
        props = analyze_audio_properties(p)
        if props:
            audio_durations.append(props["duration"])
            audio_fss.append(props["samplerate"])
            audio_channels.append(props["channels"])

    print("INPUT DATA ANALYSIS: AUDIO")
    if audio_durations:
        print(f"Duration Mean: {np.mean(audio_durations):.4f} sec")
        print(f"Duration Std: {np.std(audio_durations):.4f} sec")
        print(
            f"Sample Rate: {audio_fss[0]} Hz (Consistent in sample: {len(set(audio_fss))==1})"
        )
        print(
            f"Channels: {audio_channels[0]} (Consistent in sample: {len(set(audio_channels))==1})"
        )
    else:
        print("Audio stats unavailable.")
    print("-" * 30)

    # --- Structured / Skeleton Analysis (.mat) ---
    # We also extract gesture durations here
    gesture_durations = defaultdict(list)
    max_depths = []

    # Iterate over all training data to get robust class-specific stats
    for _, row in df.iterrows():
        if pd.notna(row["data_path"]):
            labels_data, max_d = load_mat_data(row["data_path"])
            if labels_data:
                for item in labels_data:
                    gesture_durations[item["id"]].append(item["duration"])
            if max_d:
                max_depths.append(max_d)

    print("INPUT DATA ANALYSIS: STRUCTURED (MAT/SKELETON)")
    if max_depths:
        print(f"Max Depth Value Mean: {np.mean(max_depths):.4f} mm")
        print(f"Max Depth Value Max: {np.max(max_depths):.4f} mm")
    else:
        print("Depth stats unavailable.")
    print("-" * 30)

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Correlation: Video Duration (Frames) vs Number of Gestures
    # df['num_frames'] is from metadata
    if "num_frames" in df.columns:
        corr = df["num_frames"].corr(seq_lengths)
        print(f"Correlation (NumFrames vs NumGestures): {corr:.4f}")

    # Duration per Class
    print("\nAverage Duration per Gesture Class (Frames):")
    durations_summary = []
    for gid in sorted(LABEL_MAP.values()):
        durs = gesture_durations.get(gid, [])
        if durs:
            avg_dur = np.mean(durs)
            durations_summary.append((gid, avg_dur))
        else:
            durations_summary.append((gid, 0))

    # Sort by duration to see short vs long gestures
    durations_summary.sort(key=lambda x: x[1], reverse=True)

    print("Top 3 Longest Gestures:")
    for gid, dur in durations_summary[:3]:
        print(f"  {ID_TO_NAME.get(gid)} (ID {gid}): {dur:.4f} frames")

    print("Top 3 Shortest Gestures:")
    for gid, dur in durations_summary[-3:]:
        print(f"  {ID_TO_NAME.get(gid)} (ID {gid}): {dur:.4f} frames")


if __name__ == "__main__":
    main()
