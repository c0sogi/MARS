import os
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_eda():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load Metadata
    df = pd.read_csv(METADATA_PATH)

    # ---------------------------------------------------------
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("SECTION 1: TARGET VARIABLE ANALYSIS")
    target_counts = df["target"].value_counts()
    target_ratios = df["target"].value_counts(normalize=True)

    print(f"Target Distribution:")
    for label, count in target_counts.items():
        ratio = target_ratios[label]
        print(f"Class {label}: {count} ({ratio:.4f})")

    # Imbalance Check
    majority_class = target_counts.idxmax()
    minority_class = target_counts.idxmin()
    if target_counts[minority_class] > 0:
        imbalance_ratio = target_counts[majority_class] / target_counts[minority_class]
        print(f"Imbalance Ratio (Majority/Minority): {imbalance_ratio:.4f}")
    else:
        print("Imbalance Ratio: Infinite (Minority class has 0 samples)")
    print("-" * 30)

    # ---------------------------------------------------------
    # SECTION 2: INPUT DATA ANALYSIS (SPECTROGRAM IMAGES)
    # ---------------------------------------------------------
    print("SECTION 2: INPUT DATA ANALYSIS (IMAGE/SPECTROGRAM)")

    # Sampling for Image Analysis to stay within time limits
    SAMPLE_SIZE = 2000
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        df_sample = df.copy()

    print(f"Analyzing a sample of {len(df_sample)} files for pixel statistics...")

    # Storage for extracted meta-features
    meta_features = []

    # Accumulators for global stats
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    shapes = set()

    # Iterate through sample
    for idx, row in df_sample.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        if not os.path.exists(file_path):
            continue

        try:
            # Load numpy array
            # Shape is expected to be (6, 273, 256)
            img = np.load(file_path).astype(np.float32)

            # Shape check
            shapes.add(img.shape)

            # Global stats accumulation
            pixel_sum += np.sum(img)
            pixel_sq_sum += np.sum(img**2)
            pixel_count += img.size

            # Per-sample feature extraction
            # Dimensions: (Cadence, Freq, Time)
            # ON targets: 0, 2, 4 (Panels 1, 3, 5)
            # OFF targets: 1, 3, 5 (Panels 2, 4, 6)

            on_target_indices = [0, 2, 4]
            off_target_indices = [1, 3, 5]

            on_target = img[on_target_indices, :, :]
            off_target = img[off_target_indices, :, :]

            # Basic stats
            mean_val = np.mean(img)
            std_val = np.std(img)
            max_val = np.max(img)
            min_val = np.min(img)

            # Cadence specific stats
            mean_on = np.mean(on_target)
            mean_off = np.mean(off_target)
            std_on = np.std(on_target)
            std_off = np.std(off_target)
            max_on = np.max(on_target)

            # Contrast/Difference
            diff_mean = mean_on - mean_off

            meta_features.append(
                {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "mean_on": mean_on,
                    "mean_off": mean_off,
                    "std_on": std_on,
                    "std_off": std_off,
                    "max_on": max_on,
                    "diff_mean": diff_mean,
                    "target": row["target"],
                }
            )

        except Exception:
            continue

    # Dimension Analysis
    print(f"Unique Image Dimensions found in sample: {shapes}")

    # Global Pixel Stats
    if pixel_count > 0:
        global_mean = pixel_sum / pixel_count
        global_variance = (pixel_sq_sum / pixel_count) - (global_mean**2)
        # Ensure variance is non-negative due to float precision
        global_variance = max(0.0, global_variance)
        global_std = np.sqrt(global_variance)

        print(f"Global Pixel Mean: {global_mean:.4f}")
        print(f"Global Pixel Std Dev: {global_std:.4f}")
    else:
        print("Could not calculate pixel stats (no data processed).")

    print("-" * 30)

    # ---------------------------------------------------------
    # SECTION 3: FEATURE/SIGNAL RELATIONSHIPS
    # ---------------------------------------------------------
    print("SECTION 3: FEATURE/SIGNAL RELATIONSHIPS")

    if len(meta_features) > 0:
        df_feats = pd.DataFrame(meta_features)

        # 1. Correlation Analysis
        numeric_cols = df_feats.select_dtypes(include=[np.number]).columns
        if "target" in numeric_cols:
            correlations = df_feats[numeric_cols].corr()["target"].drop("target")

            print("Correlations with Target (Top 5 absolute):")
            abs_corr = correlations.abs().sort_values(ascending=False).head(5)
            for feat, val in abs_corr.items():
                sign = correlations[feat]
                print(f"{feat}: {sign:.4f}")

        # 2. Feature Importance (Random Forest)
        # We check if we have enough data and both classes
        if df_feats["target"].nunique() > 1:
            X = df_feats.drop(columns=["target"])
            y = df_feats["target"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            rf = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)

            # Evaluate briefly
            if hasattr(rf, "predict_proba"):
                preds = rf.predict_proba(X_test)[:, 1]
                try:
                    auc = roc_auc_score(y_test, preds)
                    print(f"Meta-Feature RF AUC: {auc:.4f}")
                except ValueError:
                    print("Could not calculate AUC.")

            print("Top 5 Important Meta-Features (RF):")
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            top_feats = importances.sort_values(ascending=False).head(5)
            for feat, imp in top_feats.items():
                print(f"{feat}: {imp:.4f}")

            # 3. Redundancy
            print("Collinear Features (Correlation > 0.90):")
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

            if to_drop:
                print(f"Found {len(to_drop)} redundant features: {to_drop}")
            else:
                print(
                    "No highly collinear features found among extracted meta-features."
                )
        else:
            print(
                "Target has only one class in the sample. Skipping relationship analysis."
            )

    else:
        print("No features extracted.")


if __name__ == "__main__":
    perform_eda()
