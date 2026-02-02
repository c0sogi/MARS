import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # --- Configuration ---
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"
    SAMPLE_SIZE = 2000  # Number of files to sample for pixel stats and meta-features
    SEED = 42

    set_seed(SEED)

    # --- 1. Data Integrity & Loading ---
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # --- 2. Target Variable Analysis ---
    print("TARGET VARIABLE ANALYSIS")
    target_counts = df_train["target"].value_counts().sort_index()
    total_samples = len(df_train)

    print(f"Target Distribution: {target_counts.to_dict()}")

    class_0_ratio = target_counts.get(0, 0) / total_samples
    class_1_ratio = target_counts.get(1, 0) / total_samples

    print(f"Class 0 Ratio: {class_0_ratio:.4f}")
    print(f"Class 1 Ratio: {class_1_ratio:.4f}")

    if abs(class_0_ratio - class_1_ratio) > 0.2:
        print("Imbalance Status: Imbalanced")
    else:
        print("Imbalance Status: Balanced")
    print("-" * 30)

    # --- 3. Input Data Analysis (Image/Spectrogram Modality) ---
    print("INPUT DATA ANALYSIS")

    # We treat the (6, 273, 256) arrays as 6-channel images.
    # Sampling for efficiency
    sample_df = df_train.sample(
        n=min(len(df_train), SAMPLE_SIZE), random_state=SEED
    ).copy()

    # Accumulators for global stats
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    global_min = float("inf")
    global_max = float("-inf")

    # Lists to store meta-features for Section 4
    meta_features = []

    # Shape tracking
    shapes = {}

    for _, row in sample_df.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        try:
            # Load numpy array
            # Shape is expected to be (6, 273, 256)
            # 6 positions (ABACAD), 273 freq bins, 256 time steps
            img = np.load(file_path).astype(np.float32)

            # Record shape
            current_shape = img.shape
            shapes[current_shape] = shapes.get(current_shape, 0) + 1

            # Update global pixel stats
            pixel_sum += np.sum(img)
            pixel_sq_sum += np.sum(img**2)
            pixel_count += img.size
            global_min = min(global_min, np.min(img))
            global_max = max(global_max, np.max(img))

            # Extract Meta-Features for Relationship Analysis
            # 'A' observations are at indices 0, 2, 4 (On-Target)
            # 'B', 'C', 'D' observations are at indices 1, 3, 5 (Off-Target)
            on_target = img[[0, 2, 4], :, :]
            off_target = img[[1, 3, 5], :, :]

            stats = {
                "mean_on_target": np.mean(on_target),
                "std_on_target": np.std(on_target),
                "max_on_target": np.max(on_target),
                "mean_off_target": np.mean(off_target),
                "std_off_target": np.std(off_target),
                "max_off_target": np.max(off_target),
                "mean_diff": np.mean(on_target) - np.mean(off_target),
                "target": row["target"],  # Keep target for correlation
            }
            meta_features.append(stats)

        except Exception as e:
            # Silent fail for individual files to ensure robustness, though not expected
            continue

    # Calculate final global stats
    if pixel_count > 0:
        global_mean = pixel_sum / pixel_count
        global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))
    else:
        global_mean = 0.0
        global_std = 0.0

    # Report Dimensions
    print("Dimensions Analysis (Shape Distribution):")
    for shape, count in shapes.items():
        print(f"  Shape {shape}: {count} samples")

    # Report Channels
    # Assuming shape is (Channels, Height, Width) or similar.
    # Based on description: (6, 273, 256) -> 6 channels (Cadence positions)
    print(
        f"Channel Distribution: Fixed at {list(shapes.keys())[0][0]} channels (Cadence positions) per sample."
    )

    # Report Pixel Stats
    print(f"Global Pixel Mean: {global_mean:.4f}")
    print(f"Global Pixel Std Dev: {global_std:.4f}")
    print(f"Global Pixel Min: {global_min:.4f}")
    print(f"Global Pixel Max: {global_max:.4f}")
    print("-" * 30)

    # --- 4. Feature/Signal Relationships ---
    print("FEATURE/SIGNAL RELATIONSHIPS")

    if not meta_features:
        print("No features extracted.")
        return

    df_meta = pd.DataFrame(meta_features)

    # 4.1 Structured Relationships (Meta-Features)
    # Correlation with Target
    print("Meta-Feature Correlations with Target:")
    correlations = df_meta.corr()["target"].drop("target").sort_values(ascending=False)
    print(correlations.apply(lambda x: f"{x:.4f}"))

    # Feature Importance (Random Forest)
    X = df_meta.drop(columns=["target"])
    y = df_meta["target"]

    # Simple RF
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    print("\nTop 5 Meta-Features (Random Forest Importance):")
    for name, imp in importances.head(5).items():
        print(f"  {name}: {imp:.4f}")

    # Redundancy check (Collinearity)
    print("\nHighly Collinear Feature Pairs (> 0.90):")
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

    pairs_found = False
    for col in to_drop:
        correlated_cols = upper_tri.index[upper_tri[col] > 0.90].tolist()
        for c_row in correlated_cols:
            print(f"  {col} & {c_row}: {upper_tri.loc[c_row, col]:.4f}")
            pairs_found = True

    if not pairs_found:
        print("  None found.")

    # 4.2 Unstructured Relationships
    # Here we analyze if specific signal properties (like high max intensity) correlate with the class.
    # We can infer this from the RF importance and correlations calculated above.
    print("\nSignal Property Analysis:")
    if importances.idxmax() in ["std_on_target", "max_on_target", "mean_diff"]:
        print(
            "  Observation: Intensity variations or maximums in On-Target panels are strong indicators."
        )
    elif "mean_diff" in importances.head(3).index:
        print(
            "  Observation: The contrast between On-Target and Off-Target panels is a key predictor."
        )
    else:
        print(
            "  Observation: Simple global statistics may not be sufficient; spatial patterns likely matter more."
        )


if __name__ == "__main__":
    main()
