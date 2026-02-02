import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit


def generate_metadata():
    # Configuration
    INPUT_DIR = "./input"
    META_DIR = "./metadata"
    STRUCTURES_DIR_REL = "structures"
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(META_DIR, exist_ok=True)

    print("Loading raw datasets...")
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Helper to generate relative structure path
    def get_structure_rel_path(mol_name):
        return os.path.join(STRUCTURES_DIR_REL, f"{mol_name}.xyz")

    # Add structure file paths
    print("Adding structure file paths...")
    train_df["structure_path"] = train_df["molecule_name"].apply(get_structure_rel_path)
    test_df["structure_path"] = test_df["molecule_name"].apply(get_structure_rel_path)

    # Shuffle training data before splitting
    print("Shuffling training data...")
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Split into Train and Validation using GroupShuffleSplit
    # We must group by molecule_name to ensure no molecule appears in both sets
    print("Splitting training data (Group Shuffle Split)...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)
    groups = train_df["molecule_name"]

    train_idx, val_idx = next(gss.split(train_df, groups=groups))

    train_meta = train_df.iloc[train_idx].copy()
    val_meta = train_df.iloc[val_idx].copy()

    # Save metadata
    print("Saving metadata files...")
    train_meta_path = os.path.join(META_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(META_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(META_DIR, "test_metadata.csv")

    train_meta.to_csv(train_meta_path, index=False)
    val_meta.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    print("Metadata generation complete.")

    # ==========================================
    # Verification Step
    # ==========================================
    print("\nStarting Verification...")

    # Reload datasets
    df_train_check = pd.read_csv(train_meta_path)
    df_val_check = pd.read_csv(val_meta_path)
    df_test_check = pd.read_csv(test_meta_path)

    # 1. Print Summary Statistics
    print("\n--- Summary Statistics ---")
    for name, df in [
        ("Train", df_train_check),
        ("Validation", df_val_check),
        ("Test", df_test_check),
    ]:
        print(f"\nDataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Unique Molecules: {df['molecule_name'].nunique()}")
        if "scalar_coupling_constant" in df.columns:
            print(f"Target Mean: {df['scalar_coupling_constant'].mean():.4f}")
            print(f"Target Std: {df['scalar_coupling_constant'].std():.4f}")
        if "type" in df.columns:
            print("Type Distribution:")
            print(df["type"].value_counts(normalize=True).head())

    # 2. Check File Paths
    print("\n--- Checking File Paths ---")
    for name, df in [
        ("Train", df_train_check),
        ("Validation", df_val_check),
        ("Test", df_test_check),
    ]:
        print(f"Checking {name} paths...")
        # Sample 1000 paths (or all if less than 1000)
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
        print(f"  Missing Ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})")

        if missing_ratio > 0.5:
            print("  Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths in {name} metadata do not resolve."
            )

    # 3. Verify Split Logic
    print("\n--- Verifying Split Logic ---")
    train_mols = set(df_train_check["molecule_name"].unique())
    val_mols = set(df_val_check["molecule_name"].unique())

    # Check for overlap
    overlap = train_mols.intersection(val_mols)
    if len(overlap) > 0:
        raise AssertionError(
            f"Data Leakage detected! {len(overlap)} molecules found in both Train and Validation sets."
        )
    else:
        print("  SUCCESS: No molecule overlap between Train and Validation.")

    # Check split ratio (approximate due to group splitting)
    n_train_mols = len(train_mols)
    n_val_mols = len(val_mols)
    total_mols = n_train_mols + n_val_mols
    train_ratio = n_train_mols / total_mols

    print(f"  Molecule Split Ratio (Train): {train_ratio:.4f}")

    # Allow small deviation because groups vary in size, but it should be close to 0.8
    if not (0.75 < train_ratio < 0.85):
        raise AssertionError(
            f"Split ratio deviation too high. Expected ~0.8, got {train_ratio:.4f}"
        )
    else:
        print("  SUCCESS: Split ratio is within acceptable bounds.")

    print("\nAll checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
