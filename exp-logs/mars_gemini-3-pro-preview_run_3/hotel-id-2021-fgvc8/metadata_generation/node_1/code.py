import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42
    VAL_SIZE = 0.2

    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    # Load raw data
    train_df_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))

    # Remove duplicate images to prevent overlap between splits
    train_df_raw = train_df_raw.drop_duplicates(subset=["image"]).reset_index(drop=True)

    test_df_raw = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    # ---------------------------------------------------------
    # Process Training Data
    # ---------------------------------------------------------
    # Construct relative file paths for training images
    # Structure: train_images/<chain>/<image>
    # Ensure chain is treated as integer/string correctly for path construction
    train_df_raw["file_path"] = (
        "train_images/"
        + train_df_raw["chain"].astype(str)
        + "/"
        + train_df_raw["image"]
    )

    # Split into Train and Validation
    # We need to handle classes with only 1 sample (cannot be stratified)
    hotel_counts = train_df_raw["hotel_id"].value_counts()
    singletons = hotel_counts[hotel_counts < 2].index

    # Separate data
    is_singleton = train_df_raw["hotel_id"].isin(singletons)
    df_singletons = train_df_raw[is_singleton].copy()
    df_multiples = train_df_raw[~is_singleton].copy()

    print(f"Total samples: {len(train_df_raw)}")
    print(f"Singleton classes samples (forced to train): {len(df_singletons)}")
    print(f"Stratifiable samples: {len(df_multiples)}")

    # Stratified split on samples with >= 2 instances
    train_split, val_split = train_test_split(
        df_multiples,
        test_size=VAL_SIZE,
        stratify=df_multiples["hotel_id"],
        random_state=RANDOM_STATE,
    )

    # Combine singletons into training set
    train_final = (
        pd.concat([df_singletons, train_split], axis=0)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    val_final = val_split.reset_index(drop=True)

    # ---------------------------------------------------------
    # Process Test Data
    # ---------------------------------------------------------
    # Construct relative file paths for test images
    # Structure: test_images/<image>
    # Note: sample_submission.csv contains the images we need to predict
    test_df = test_df_raw.copy()
    test_df["file_path"] = "test_images/" + test_df["image"]

    # ---------------------------------------------------------
    # Save Metadata
    # ---------------------------------------------------------
    print("Saving metadata...")
    train_final.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    val_final.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # ---------------------------------------------------------
    # Verification and Statistics
    # ---------------------------------------------------------
    print("\nVerifying generated metadata...")

    # Reload datasets
    train_loaded = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_loaded = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_loaded = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Summary Statistics
    print("\n--- Summary Statistics ---")
    print(f"Train set shape: {train_loaded.shape}")
    print(f"Validation set shape: {val_loaded.shape}")
    print(f"Test set shape: {test_loaded.shape}")

    print(f"Train unique hotels: {train_loaded['hotel_id'].nunique()}")
    print(f"Val unique hotels: {val_loaded['hotel_id'].nunique()}")

    # 2. File Path Verification
    def check_paths(df, name):
        print(f"\nChecking file paths for {name}...")
        sample_size = min(1000, len(df))
        samples = df.sample(n=sample_size, random_state=RANDOM_STATE)[
            "file_path"
        ].values

        missing_count = 0
        missing_samples = []

        for path in samples:
            full_path = os.path.join(INPUT_DIR, path)
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(path)

        missing_ratio = missing_count / sample_size
        print(
            f"Missing file ratio: {missing_ratio:.4f} ({missing_count}/{sample_size})"
        )

        if missing_ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"Too many missing files in {name} dataset! Ratio: {missing_ratio}"
            )

    check_paths(train_loaded, "Train")
    check_paths(val_loaded, "Validation")
    check_paths(test_loaded, "Test")

    # 3. Validation Split Verification
    print("\nVerifying split requirements...")

    # Assert no overlap
    train_imgs = set(train_loaded["image"])
    val_imgs = set(val_loaded["image"])
    assert (
        len(train_imgs.intersection(val_imgs)) == 0
    ), "Overlap detected between train and validation sets!"

    # Assert stratification logic (roughly)
    # We can't check exact distribution equality because singletons were forced to train,
    # but we can check that the split on 'multiples' was correct.
    # Re-identify multiples in the loaded data
    val_counts = val_loaded["hotel_id"].value_counts()

    # Check that validation set doesn't contain classes that were supposed to be singletons
    # (though in this logic, val set comes purely from multiples, so this is implicitly true)
    # Let's verify that the size is correct relative to the stratifiable portion
    expected_val_size = int(len(df_multiples) * VAL_SIZE)
    # Allow small off-by-one due to rounding in split
    assert abs(len(val_loaded) - expected_val_size) <= len(
        df_multiples["hotel_id"].unique()
    ), f"Validation set size {len(val_loaded)} deviates significantly from expected {expected_val_size}"

    print("Verification successful. Metadata generation complete.")


if __name__ == "__main__":
    main()
