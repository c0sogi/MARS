import os
import pandas as pd
import numpy as np


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load train and test CSVs
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise

    # Function to generate relative file path for structure files
    # Structure files are located in input/structures/
    def get_structure_path(molecule_name):
        return os.path.join("structures", f"{molecule_name}.xyz")

    # Add file_path column
    train_df["file_path"] = train_df["molecule_name"].apply(get_structure_path)
    test_df["file_path"] = test_df["molecule_name"].apply(get_structure_path)

    print("Performing Group Split...")
    # Get unique molecules for group splitting
    unique_molecules = train_df["molecule_name"].unique()

    # Shuffle with fixed seed
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_molecules)

    # Split 80/20
    n_train = int(len(unique_molecules) * 0.8)
    train_molecules = set(unique_molecules[:n_train])
    val_molecules = set(unique_molecules[n_train:])

    # Create train and validation dataframes based on molecule groups
    train_meta = train_df[train_df["molecule_name"].isin(train_molecules)].copy()
    val_meta = train_df[train_df["molecule_name"].isin(val_molecules)].copy()
    test_meta = test_df.copy()

    # Save metadata files
    print("Saving metadata to ./metadata/ ...")
    train_meta.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_meta.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_meta.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # ---------------------------------------------------------
    # Verification Section
    # ---------------------------------------------------------
    print("\nStarting Verification...")

    # Reload metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Print Summary Statistics
    datasets = {"Train": df_train, "Validation": df_val, "Test": df_test}
    for name, df in datasets.items():
        print(f"\n[{name} Dataset]")
        print(f"Total Samples: {len(df)}")
        print(f"Unique Molecules: {df['molecule_name'].nunique()}")
        print(f"Data Shape: {df.shape}")
        if "scalar_coupling_constant" in df.columns:
            print(f"Target Mean: {df['scalar_coupling_constant'].mean():.4f}")
            print(f"Target Std: {df['scalar_coupling_constant'].std():.4f}")

    # 2. Check File Paths
    print("\nChecking file path resolution...")
    for name, df in datasets.items():
        # Sample 1000 paths (or all if less than 1000)
        sample_size = min(1000, len(df))
        sample_paths = df["file_path"].sample(n=sample_size, random_state=RANDOM_STATE)

        missing_paths = []
        for rel_path in sample_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)
            if not os.path.exists(full_path):
                missing_paths.append(rel_path)

        missing_ratio = len(missing_paths) / sample_size
        print(f"{name}: Missing Ratio = {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print(f"Sample missing paths in {name}: {missing_paths[:5]}")
            raise FileNotFoundError(
                f"Missing file ratio ({missing_ratio}) exceeds 0.5 for {name} dataset."
            )

    # 3. Verify Group Split
    print("\nVerifying group split integrity...")
    train_mols_set = set(df_train["molecule_name"].unique())
    val_mols_set = set(df_val["molecule_name"].unique())

    # Check for intersection
    intersection = train_mols_set.intersection(val_mols_set)
    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage: {len(intersection)} molecules found in both Train and Validation sets."
        )

    # Check split ratio
    total_mols = len(train_mols_set) + len(val_mols_set)
    actual_train_ratio = len(train_mols_set) / total_mols
    print(f"Actual Train Split Ratio (by molecule): {actual_train_ratio:.4f}")

    # Allow small tolerance for integer division
    if not (0.79 <= actual_train_ratio <= 0.81):
        raise AssertionError(
            f"Split ratio {actual_train_ratio:.4f} deviates significantly from expected 0.8"
        )

    print("\nMetadata generation and verification completed successfully.")


if __name__ == "__main__":
    main()
