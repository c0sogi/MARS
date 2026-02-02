import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_datasets
from library.model import DRModel
from library.engine import train_model, predict_and_submit


def create_mini_metadata(original_meta_dir, new_meta_dir, sample_size=16):
    """
    Creates a subset of the metadata csvs to speed up the demo.
    """
    os.makedirs(new_meta_dir, exist_ok=True)

    for filename in ["train.csv", "val.csv", "test.csv"]:
        src = os.path.join(original_meta_dir, filename)
        dst = os.path.join(new_meta_dir, filename)

        if os.path.exists(src):
            df = pd.read_csv(src)
            # Take a small subset
            df_mini = df.head(sample_size).copy()
            df_mini.to_csv(dst, index=False)
            print(f"Created mini metadata for {filename}: {len(df_mini)} rows")
        else:
            print(f"Warning: {src} not found.")


def run_demo():
    # 1. Configuration and Setup
    print(">>> 1. Setup Configuration")
    SEED = 42
    seed_everything(SEED)

    INPUT_DIR = "./input"
    ORIGINAL_META_DIR = "./metadata"

    # Working directories for this demo
    WORK_DIR = "./working/demo_execution"
    MINI_META_DIR = os.path.join(WORK_DIR, "metadata")
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")

    # Clean up previous run if exists
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    # 2. Prepare Data Subset
    print("\n>>> 2. Preparing Data Subset")
    # We create a mini version of metadata to avoid processing 3000+ images for this demo
    create_mini_metadata(ORIGINAL_META_DIR, MINI_META_DIR, sample_size=16)

    # 3. Data Loading
    print("\n>>> 3. Testing Data Loading (library.data)")
    IMG_SIZE = 224  # Small size for speed
    BATCH_SIZE = 4

    # Instantiate datasets
    # This will trigger process_images which reads from disk and saves .npy to CACHE_DIR
    train_ds, val_ds, test_ds = get_datasets(
        input_dir=INPUT_DIR,
        metadata_dir=MINI_META_DIR,
        cache_dir=CACHE_DIR,
        img_size=IMG_SIZE,
        load_cached_data=False,  # Force processing
    )

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    # Verify cache creation
    assert os.path.exists(
        os.path.join(CACHE_DIR, f"train_images_{IMG_SIZE}.npy")
    ), "Train images not cached"
    assert os.path.exists(
        os.path.join(CACHE_DIR, "train_labels.npy")
    ), "Train labels not cached"

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 4. Model Initialization
    print("\n>>> 4. Testing Model Initialization (library.model)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use resnet18 as a lightweight backbone for demonstration
    model = DRModel(model_name="resnet18", pretrained=True, drop_rate=0.2)
    model.to(device)

    # Verify Forward Pass with dummy data
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # 5. Training Loop
    print("\n>>> 5. Testing Training Loop (library.engine)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run training for 1 epoch
    best_qwk = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=1,
        accumulation_steps=1,
        patience=1,
        save_path=MODEL_SAVE_PATH,
    )

    print(f"Training finished. Best QWK: {best_qwk}")
    assert os.path.exists(MODEL_SAVE_PATH), "Model file was not saved."

    # 6. Inference and Submission
    print("\n>>> 6. Testing Inference and Submission (library.engine)")

    # We need a sample submission file for the engine to work correctly with IDs.
    # Since we are using a subset, we need to mock a sample_submission.csv that matches our test subset.
    # The engine falls back to metadata/test.csv if sample_submission isn't found,
    # but let's ensure the path passed to predict_and_submit aligns with our mini metadata.

    # Note: predict_and_submit logic in library.engine tries to load sample_submission.csv from input
    # OR metadata/test.csv. Since we can't write to input, we rely on it finding the metadata
    # or we point it to our mini metadata.
    # However, the engine code hardcodes paths in default args but allows overrides.
    # We will override the sample_sub_path to point to our mini test metadata to ensure ID alignment.

    predict_and_submit(
        model=model,
        test_loader=test_loader,
        device=device,
        submission_path=SUBMISSION_PATH,
        sample_sub_path=os.path.join(MINI_META_DIR, "test.csv"),
    )

    assert os.path.exists(SUBMISSION_PATH), "Submission file not created."

    # Verify submission content
    sub_df = pd.read_csv(SUBMISSION_PATH)
    print("Submission head:")
    print(sub_df.head())

    assert (
        "id_code" in sub_df.columns and "diagnosis" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) == len(
        test_ds
    ), f"Submission length mismatch. Expected {len(test_ds)}, got {len(sub_df)}"
    assert (
        sub_df["diagnosis"].dtype == int or sub_df["diagnosis"].dtype == np.int64
    ), "Diagnosis should be integer."

    # 7. Metric Utility Verification
    print("\n>>> 7. Verifying Metric Utility (library.utils)")
    # Case 1: Perfect agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    score = quadratic_weighted_kappa(y_true, y_pred)
    print(f"QWK (Perfect Agreement): {score}")
    assert np.isclose(score, 1.0), "QWK should be 1.0 for perfect agreement"

    # Case 2: Complete disagreement
    y_true_bad = np.array([0, 0, 0])
    y_pred_bad = np.array([4, 4, 4])
    score_bad = quadratic_weighted_kappa(y_true_bad, y_pred_bad)
    print(f"QWK (Bad Agreement): {score_bad}")
    # Should be 0 or negative
    assert score_bad <= 0.1, "QWK should be low for disagreement"

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
