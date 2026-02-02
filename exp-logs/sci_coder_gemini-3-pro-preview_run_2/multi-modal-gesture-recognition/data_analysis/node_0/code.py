import os
import pandas as pd
import numpy as np
import scipy.io
import cv2
import soundfile as sf
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42

# Gesture Vocabulary Mapping
GESTURE_MAP = {
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_FILE}")

    df = pd.read_csv(METADATA_FILE)
    # Convert labels from space-separated string to list of ints
    df["labels"] = df["labels"].apply(
        lambda x: (
            [int(i) for i in str(x).split()]
            if pd.notna(x) and str(x).strip() != ""
            else []
        )
    )
    return df


def analyze_targets(df):
    print("TARGET VARIABLE ANALYSIS")

    # Flatten all labels
    all_labels = [label for sublist in df["labels"] for label in sublist]
    total_gestures = len(all_labels)

    if total_gestures == 0:
        print("No gestures found in training data.")
        return

    # Class distribution
    label_counts = pd.Series(all_labels).value_counts().sort_index()

    print(f"Total Gesture Instances: {total_gestures}")
    print(f"Unique Classes: {len(label_counts)}")

    # Imbalance
    min_class = label_counts.idxmin()
    min_count = label_counts.min()
    max_class = label_counts.idxmax()
    max_count = label_counts.max()

    print(f"Class Balance Ratio (Max/Min): {max_count/min_count:.4f}")
    print(
        f"Most Frequent Class: {max_class} (Count: {max_count}, {max_count/total_gestures*100:.2f}%)"
    )
    print(
        f"Least Frequent Class: {min_class} (Count: {min_count}, {min_count/total_gestures*100:.2f}%)"
    )

    # Sequence Length Analysis
    seq_lengths = df["labels"].apply(len)
    print(f"Sequence Length Mean: {seq_lengths.mean():.4f}")
    print(f"Sequence Length Std: {seq_lengths.std():.4f}")
    print(f"Sequence Length Min: {seq_lengths.min()}")
    print(f"Sequence Length Max: {seq_lengths.max()}")
    print("-" * 30)


def analyze_audio_data(df, sample_size=50):
    print("INPUT DATA ANALYSIS - AUDIO")

    audio_paths = df["audio_path"].dropna().tolist()
    if not audio_paths:
        print("No audio paths found.")
        return

    sampled_paths = random.sample(audio_paths, min(len(audio_paths), sample_size))

    durations = []
    sample_rates = []
    channels = []

    for p in sampled_paths:
        full_path = os.path.join(INPUT_DIR, p)
        try:
            info = sf.info(full_path)
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels.append(info.channels)
        except Exception:
            continue

    if durations:
        print(f"Audio Duration Mean: {np.mean(durations):.4f} sec")
        print(f"Audio Duration Std: {np.std(durations):.4f} sec")
        print(f"Audio Duration Min: {np.min(durations):.4f} sec")
        print(f"Audio Duration Max: {np.max(durations):.4f} sec")

        unique_sr = np.unique(sample_rates)
        print(f"Sampling Rates: {unique_sr}")

        unique_ch = np.unique(channels)
        print(f"Channels Distribution: {unique_ch}")
        if len(unique_ch) > 1:
            print("Warning: Inconsistent channel counts detected.")
    else:
        print("Could not analyze audio files.")
    print("-" * 30)


def analyze_video_data(df, sample_size=30):
    print("INPUT DATA ANALYSIS - IMAGE/VIDEO")

    rgb_paths = df["rgb_path"].dropna().tolist()
    if not rgb_paths:
        print("No RGB paths found.")
        return

    sampled_paths = random.sample(rgb_paths, min(len(rgb_paths), sample_size))

    widths = []
    heights = []
    fps_list = []
    pixel_means = []
    pixel_stds = []

    for p in sampled_paths:
        full_path = os.path.join(INPUT_DIR, p)
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        widths.append(width)
        heights.append(height)
        fps_list.append(fps)

        # Read first frame for pixel stats
        ret, frame = cap.read()
        if ret:
            # Frame is BGR
            pixel_means.append(np.mean(frame))
            pixel_stds.append(np.std(frame))

        cap.release()

    if widths:
        print(f"Widths Distribution: {np.unique(widths)}")
        print(f"Heights Distribution: {np.unique(heights)}")
        print(f"Aspect Ratios: {np.unique(np.array(widths)/np.array(heights))}")
        print(f"FPS Distribution: {np.unique(np.round(fps_list, 2))}")

        if pixel_means:
            print(f"Global Pixel Mean (Est): {np.mean(pixel_means):.4f}")
            print(f"Global Pixel Std (Est): {np.mean(pixel_stds):.4f}")
            print("Channel Count: 3 (RGB)")
    else:
        print("Could not analyze video files.")
    print("-" * 30)


def analyze_mat_structure_and_relationships(df):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # We will analyze the relationship between video length and gesture count
    # And also specific gesture durations from the .mat files

    gesture_durations = {i: [] for i in range(1, 21)}

    # Sample a larger subset for mat file analysis as it is fast
    mat_paths = df["data_path"].dropna().tolist()
    sampled_paths = random.sample(mat_paths, min(len(mat_paths), 500))

    valid_samples_count = 0

    for p in sampled_paths:
        full_path = os.path.join(INPUT_DIR, p)
        try:
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                continue

            video = mat["Video"]
            labels_raw = getattr(video, "Labels", [])

            # Helper to process label object
            def process_label(obj):
                try:
                    name = obj.Name
                    start = obj.Begin
                    end = obj.End
                    if name in GESTURE_MAP:
                        gid = GESTURE_MAP[name]
                        duration = end - start + 1
                        gesture_durations[gid].append(duration)
                except AttributeError:
                    pass

            if isinstance(labels_raw, np.ndarray):
                if labels_raw.ndim == 0:
                    process_label(labels_raw.item())
                else:
                    for l in labels_raw:
                        process_label(l)
            else:
                process_label(labels_raw)

            valid_samples_count += 1

        except Exception:
            continue

    # 1. Unstructured Relationship: Video Length vs Number of Gestures
    # We use the dataframe info for this
    correlation = df["num_frames"].corr(df["labels"].apply(len))
    print(f"Correlation (NumFrames vs NumGestures): {correlation:.4f}")

    # 2. Gesture Duration Analysis
    print("\nGesture Duration Analysis (in frames):")
    durations_summary = []
    for gid in range(1, 21):
        durs = gesture_durations[gid]
        if durs:
            durations_summary.append(
                {
                    "Gesture": gid,
                    "Mean_Dur": np.mean(durs),
                    "Std_Dur": np.std(durs),
                    "Count": len(durs),
                }
            )

    durations_df = pd.DataFrame(durations_summary)
    if not durations_df.empty:
        longest = durations_df.loc[durations_df["Mean_Dur"].idxmax()]
        shortest = durations_df.loc[durations_df["Mean_Dur"].idxmin()]

        print(
            f"Longest Avg Gesture: ID {int(longest['Gesture'])} ({longest['Mean_Dur']:.2f} frames)"
        )
        print(
            f"Shortest Avg Gesture: ID {int(shortest['Gesture'])} ({shortest['Mean_Dur']:.2f} frames)"
        )
        print(
            f"Overall Mean Gesture Duration: {durations_df['Mean_Dur'].mean():.2f} frames"
        )

    print("-" * 30)


def main():
    set_seed(SEED)

    try:
        df = load_metadata()
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    analyze_targets(df)
    analyze_video_data(df)
    analyze_audio_data(df)
    analyze_mat_structure_and_relationships(df)


if __name__ == "__main__":
    main()
