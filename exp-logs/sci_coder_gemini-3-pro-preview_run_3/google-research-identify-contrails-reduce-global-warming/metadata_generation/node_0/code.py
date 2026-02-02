import os
import json
import pandas as pd
import numpy as np
import random


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Starting metadata generation...")

    # --- Helper Functions ---

    def load_metadata_json(filename):
        """Loads metadata JSON into a dictionary keyed by record_id."""
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Metadata file {filename} not found.")
            return {}

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            # Convert to dict: record_id -> metadata_dict
            return {str(item["record_id"]): item for item in data}
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return {}

    def get_records_from_dir(
        split_name, meta_json, expect_pixel_mask=False, expect_individual_mask=False
    ):
        """Scans a directory and returns a list of dictionaries with file paths."""
        dir_path = os.path.join(INPUT_DIR, split_name)
        if not os.path.exists(dir_path):
            return []

        records = []
        # List all subdirectories (record_ids)
        try:
            record_ids = [
                d
                for d in os.listdir(dir_path)
                if os.path.isdir(os.path.join(dir_path, d))
            ]
        except Exception as e:
            print(f"Error scanning {dir_path}: {e}")
            return []

        for rid in record_ids:
            # Path relative to ./input
            rel_dir_path = os.path.join(split_name, rid)

            record = {"record_id": rid, "split": split_name}

            # Add band paths
            for b in range(8, 17):
                band_filename = f"band_{b:02d}.npy"
                record[f"band_{b:02d}"] = os.path.join(rel_dir_path, band_filename)

            # Add mask paths
            if expect_pixel_mask:
                record["human_pixel_masks"] = os.path.join(
                    rel_dir_path, "human_pixel_masks.npy"
                )

            if expect_individual_mask:
                record["human_individual_masks"] = os.path.join(
                    rel_dir_path, "human_individual_masks.npy"
                )

            # Add metadata from JSON
            if rid in meta_json:
                for k, v in meta_json[rid].items():
                    if k != "record_id":  # Avoid duplicate
                        record[k] = v

            records.append(record)

        return records

    # --- Main Logic ---

    # Load JSON metadata
    train_meta_json = load_metadata_json("train_metadata.json")
    val_meta_json = load_metadata_json("validation_metadata.json")

    # Scan directories
    # Note: Validation set in this dataset typically has pixel masks but no individual masks
    train_data = get_records_from_dir(
        "train", train_meta_json, expect_pixel_mask=True, expect_individual_mask=True
    )
    val_data = get_records_from_dir(
        "validation",
        val_meta_json,
        expect_pixel_mask=True,
        expect_individual_mask=False,
    )
    test_data = get_records_from_dir(
        "test", {}, expect_pixel_mask=False, expect_individual_mask=False
    )

    # Create DataFrames
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_test = pd.DataFrame(test_data)

    # Handle Validation Split
    if df_val.empty and not df_train.empty:
        print("Validation set is empty. Creating split from training set...")

        # Shuffle
        df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(
            drop=True
        )

        # Split 80:20
        split_idx = int(len(df_train) * 0.8)

        df_val = df_train.iloc[split_idx:].copy()
        df_train = df_train.iloc[:split_idx].copy()

        # Update split column
        df_val["split"] = "validation"

        # Verification of split
        assert len(df_train) > 0, "Training set is empty after split"
        assert len(df_val) > 0, "Validation set is empty after split"
        # Check no overlap
        train_ids = set(df_train["record_id"])
        val_ids = set(df_val["record_id"])
        assert train_ids.isdisjoint(val_ids), "Training and Validation sets overlap!"

        print(f"Created validation split: {len(df_train)} train, {len(df_val)} val")

    # Save Metadata
    print("Saving metadata files...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "validation.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # --- Verification ---
    print("\n--- Verification ---")

    datasets = {"Train": df_train, "Validation": df_val, "Test": df_test}

    for name, df in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Number of samples: {len(df)}")
        if len(df) == 0:
            print(f"Warning: {name} dataset is empty.")
            continue

        print(f"Columns: {list(df.columns)}")

        # Check file paths
        # Identify path columns: start with 'band_' or contain 'mask'
        path_columns = [c for c in df.columns if c.startswith("band_") or "mask" in c]

        all_paths = []
        for col in path_columns:
            all_paths.extend(df[col].dropna().tolist())

        if not all_paths:
            print("No file paths found to check.")
            continue

        # Randomly check 1000 paths
        n_check = min(1000, len(all_paths))
        check_paths = random.sample(all_paths, n_check)

        missing_count = 0
        missing_examples = []

        for p in check_paths:
            full_path = os.path.join(INPUT_DIR, p)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        missing_ratio = missing_count / n_check
        print(
            f"Checked {n_check} paths. Missing: {missing_count} (Ratio: {missing_ratio:.4f})"
        )

        if missing_ratio > 0.5:
            print("Sample of missing paths:")
            for mp in missing_examples:
                print(f"  {mp}")
            raise FileNotFoundError(
                f"Missing file ratio {missing_ratio:.4f} exceeds threshold 0.5 for {name} dataset."
            )

    print("\nMetadata generation and verification complete.")


if __name__ == "__main__":
    main()
