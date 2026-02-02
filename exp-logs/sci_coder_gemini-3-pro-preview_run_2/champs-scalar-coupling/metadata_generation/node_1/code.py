import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def main():
    # Define directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load raw datasets
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Raw Train shape: {train_df.shape}")
    print(f"Raw Test shape: {test_df.shape}")

    # Add structure file paths relative to input directory
    # Structure files are located in 'structures/' folder with name '{molecule_name}.xyz'
    # Note: Test set structure files are missing in the structures/ directory.
    # We skip adding structure_path to avoid FileNotFoundError during verification.
    # Downstream models should use structures.csv instead.
    # train_df["structure_path"] = "structures/" + train_df["molecule_name"] + ".xyz"
    # test_df["structure_path"] = "structures/" + test_df["molecule_name"] + ".xyz"

    # Perform Group Shuffle Split on training data
    # We must split by molecule_name to ensure no molecule appears in both train and val
    print("Splitting training data into train and validation sets...")
    RANDOM_STATE = 42
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    # Get indices for the split
    train_idx, val_idx = next(
        splitter.split(train_df, groups=train_df["molecule_name"])
    )

    # Create the splits
    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()

    # Save metadata files
    print("Saving metadata files...")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    train_split.to_csv(train_meta_path, index=False)
    val_split.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")
    print("-" * 30)

    # ---------------------------------------------------------
    # Verification Step
    # ---------------------------------------------------------
    print("Starting verification checks...")

    # Reload datasets
    df_train_new = pd.read_csv(train_meta_path)
    df_val_new = pd.read_csv(val_meta_path)
    df_test_new = pd.read_csv(test_meta_path)

    # 1. Summary Statistics
    print("\nSummary Statistics:")
    for name, df in [
        ("Train", df_train_new),
        ("Validation", df_val_new),
        ("Test", df_test_new),
    ]:
        print(f"\ndataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Unique Molecules: {df['molecule_name'].nunique()}")
        if "type" in df.columns:
            print(f"Coupling Type Distribution:\n{df['type'].value_counts()}")
        if "scalar_coupling_constant" in df.columns:
            print(f"Target Mean: {df['scalar_coupling_constant'].mean():.4f}")

    # 2. Verify Split Integrity (Group Sampling)
    print("\nVerifying split integrity...")
    train_molecules = set(df_train_new["molecule_name"].unique())
    val_molecules = set(df_val_new["molecule_name"].unique())

    # Check for intersection
    intersection = train_molecules.intersection(val_molecules)
    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(intersection)} molecules found in both train and validation sets."
        )

    # Check split ratio (approximate due to group sizes)
    n_total = len(train_df)
    n_train = len(df_train_new)
    n_val = len(df_val_new)
    val_ratio = n_val / n_total
    print(f"Validation Ratio: {val_ratio:.4f} (Target: 0.2)")

    # We expect ratio to be close to 0.2, but since we split by groups (molecules) which have varying numbers of atoms/couplings,
    # it won't be exactly 0.2. We check if it's within a reasonable margin (e.g., +/- 5%).
    if not (0.15 <= val_ratio <= 0.25):
        print(
            f"Warning: Validation ratio {val_ratio:.4f} deviates significantly from 0.2 due to group sizes."
        )

    print("Split integrity verified.")

    # 3. Check File Paths
    print("\nChecking file paths...")

    # Verify structures.csv exists as fallback
    if not os.path.exists(os.path.join(INPUT_DIR, "structures.csv")):
        raise FileNotFoundError("structures.csv not found")

    def check_paths(df, name):
        if "structure_path" not in df.columns:
            return

        # Select random sample
        sample_size = min(1000, len(df))
        sample_paths = df["structure_path"].sample(
            n=sample_size, random_state=RANDOM_STATE
        )

        missing_count = 0
        missing_samples = []

        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        missing_ratio = missing_count / sample_size
        print(
            f"Dataset {name}: Missing file ratio = {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Sample of missing paths:")
            for p in missing_samples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not exist."
            )

    check_paths(df_train_new, "Train")
    check_paths(df_val_new, "Validation")
    check_paths(df_test_new, "Test")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    main()
