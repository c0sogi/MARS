import os
import json
import pandas as pd
import random
import shutil

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def get_record_ids(split_dir):
    """
    Scans a directory to find all subdirectories, which correspond to record_ids.
    """
    if not os.path.exists(split_dir):
        return []
    # Directories in the split folder are the record_ids
    return [d.name for d in os.scandir(split_dir) if d.is_dir()]


def generate_metadata():
    """
    Generates metadata CSVs for train, validation, and test sets.
    """
    # Create metadata directory
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    os.makedirs(METADATA_DIR)

    # Define splits and their expected file properties
    splits = {
        "train": {
            "has_pixel_mask": True,
            "has_individual_mask": True,
            "metadata_file": "train_metadata.json",
        },
        "validation": {
            "has_pixel_mask": True,
            "has_individual_mask": False,  # Description says individual masks are removed
            "metadata_file": "validation_metadata.json",
        },
        "test": {
            "has_pixel_mask": False,
            "has_individual_mask": False,
            "metadata_file": None,  # Usually no metadata json for test
        },
    }

    generated_dfs = {}

    for split_name, props in splits.items():
        print(f"Processing {split_name} dataset...")
        split_dir = os.path.join(INPUT_DIR, split_name)
        record_ids = get_record_ids(split_dir)

        if not record_ids:
            print(f"  No records found for {split_name}.")
            continue

        data = []
        for rid in record_ids:
            # Ensure record_id is string to preserve precision and match JSON
            row = {"record_id": str(rid)}

            # Construct paths for Bands 08-16
            # Paths are relative to INPUT_DIR
            for b in range(8, 17):
                band_filename = f"band_{b:02d}.npy"
                # Format: split_name/record_id/band_xx.npy
                rel_path = os.path.join(split_name, rid, band_filename)
                row[f"band_{b:02d}"] = rel_path

            # Construct paths for Masks
            if props["has_pixel_mask"]:
                row["human_pixel_masks"] = os.path.join(
                    split_name, rid, "human_pixel_masks.npy"
                )

            if props["has_individual_mask"]:
                row["human_individual_masks"] = os.path.join(
                    split_name, rid, "human_individual_masks.npy"
                )

            data.append(row)

        df = pd.DataFrame(data)

        # Merge with provided JSON metadata if available
        if props["metadata_file"]:
            meta_path = os.path.join(INPUT_DIR, props["metadata_file"])
            if os.path.exists(meta_path):
                print(f"  Loading metadata from {meta_path}...")
                with open(meta_path, "r") as f:
                    meta_json = json.load(f)

                df_meta = pd.DataFrame(meta_json)
                # Ensure record_id is string for merging
                df_meta["record_id"] = df_meta["record_id"].astype(str)

                # Merge (left join to keep all scanned directories)
                df = df.merge(df_meta, on="record_id", how="left")

        # Save to metadata directory
        output_filename = f"{split_name}_metadata.csv"
        output_path = os.path.join(METADATA_DIR, output_filename)
        df.to_csv(output_path, index=False)

        generated_dfs[split_name] = df
        print(f"  Saved {output_path} with {len(df)} records.")

    return generated_dfs


def validate_paths(df, split_name):
    """
    Checks a random sample of file paths to ensure they exist.
    """
    print(f"Validating file paths for {split_name}...")

    # Identify columns that contain file paths
    path_cols = [c for c in df.columns if "band_" in c or "mask" in c]

    # Collect all paths from the dataframe
    all_paths = []
    for col in path_cols:
        # Drop NaNs just in case
        paths = df[col].dropna().tolist()
        all_paths.extend(paths)

    if not all_paths:
        print("  No paths to validate.")
        return

    # Sample 1000 paths (or all if less than 1000)
    sample_size = min(1000, len(all_paths))
    sampled_paths = random.sample(all_paths, sample_size)

    missing_count = 0
    missing_examples = []

    for rel_path in sampled_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(rel_path)

    ratio = missing_count / sample_size
    print(f"  Missing file ratio: {ratio:.4f} ({missing_count}/{sample_size})")

    if ratio > 0.5:
        print("  Sample missing files:")
        for mp in missing_examples:
            print(f"    {mp}")
        raise AssertionError(
            f"Missing file ratio {ratio:.4f} exceeds threshold 0.5 for {split_name} dataset."
        )


def print_summary(dfs):
    print("\n" + "=" * 30)
    print("DATASET SUMMARY")
    print("=" * 30)

    for split_name, df in dfs.items():
        print(f"Dataset: {split_name}")
        print(f"  Total Samples: {len(df)}")
        print(f"  Columns: {len(df.columns)}")

        # Print some distribution info if available
        if "timestamp" in df.columns:
            # timestamp is unix epoch
            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()
            print(f"  Timestamp Range: {min_ts} to {max_ts}")

        print("-" * 30)


def main():
    # Set random seed for reproducibility
    random.seed(RANDOM_STATE)

    try:
        # Generate metadata
        dfs = generate_metadata()

        # Print summary statistics
        print_summary(dfs)

        # Validate paths for each generated dataset
        for split_name, df in dfs.items():
            if not df.empty:
                validate_paths(df, split_name)

        print("\nMetadata generation and validation completed successfully.")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
