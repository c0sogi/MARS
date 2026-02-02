import os
import shutil
import pandas as pd
import numpy as np
import torch
import library.config
import library.data
import library.model
import library.train
import library.predict
import library.utils


def run_demo():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    print(">>> [1/7] Setting up demo environment...")

    # Define demo-specific paths in the working directory
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    # Seed for reproducibility
    library.utils.seed_everything(42)

    # ==========================================
    # 2. Create Data Subset (for Speed)
    # ==========================================
    print(">>> [2/7] Creating dataset subset for rapid execution...")

    # Load original metadata
    orig_train_df = pd.read_parquet(library.config.TRAIN_METADATA_PATH)
    orig_val_df = pd.read_parquet(library.config.VAL_METADATA_PATH)
    orig_test_df = pd.read_parquet(library.config.TEST_METADATA_PATH)

    # Sample a small subset (e.g., 8 train, 4 val, 4 test) to verify pipeline functionality
    # We ensure we don't sample more than available
    n_train = min(8, len(orig_train_df))
    n_val = min(4, len(orig_val_df))
    n_test = min(4, len(orig_test_df))

    demo_train_df = orig_train_df.sample(n=n_train, random_state=42)
    demo_val_df = orig_val_df.sample(n=n_val, random_state=42)
    demo_test_df = orig_test_df.sample(n=n_test, random_state=42)

    # Save subset metadata
    demo_train_path = os.path.join(DEMO_META_DIR, "train.parquet")
    demo_val_path = os.path.join(DEMO_META_DIR, "val.parquet")
    demo_test_path = os.path.join(DEMO_META_DIR, "test.parquet")

    demo_train_df.to_parquet(demo_train_path, index=False)
    demo_val_df.to_parquet(demo_val_path, index=False)
    demo_test_df.to_parquet(demo_test_path, index=False)

    print(
        f"    Subset sizes - Train: {len(demo_train_df)}, Val: {len(demo_val_df)}, Test: {len(demo_test_df)}"
    )

    # ==========================================
    # 3. Monkey-Patch Library Configs
    # ==========================================
    print(">>> [3/7] Configuring library to use demo paths and parameters...")

    # We need to update the global variables in the imported modules because
    # they were bound at import time.

    # Update Paths in library.data
    library.data.TRAIN_METADATA_PATH = demo_train_path
    library.data.VAL_METADATA_PATH = demo_val_path
    library.data.TEST_METADATA_PATH = demo_test_path

    # Update Cache Paths to avoid conflicts with real run
    library.data.CACHE_TRAIN_X = os.path.join(DEMO_DIR, "cached_train_X.npy")
    library.data.CACHE_TRAIN_Y = os.path.join(DEMO_DIR, "cached_train_y.npy")
    library.data.CACHE_VAL_X = os.path.join(DEMO_DIR, "cached_val_X.npy")
    library.data.CACHE_VAL_Y = os.path.join(DEMO_DIR, "cached_val_y.npy")
    library.data.CACHE_TEST_X = os.path.join(DEMO_DIR, "cached_test_X.npy")
    library.data.CACHE_TEST_IDS = os.path.join(DEMO_DIR, "cached_test_ids.npy")

    # Update Paths in library.predict (it uses TEST_METADATA_PATH and CACHE paths)
    library.predict.TEST_METADATA_PATH = demo_test_path
    library.predict.CACHE_TEST_X = library.data.CACHE_TEST_X
    library.predict.CACHE_TEST_IDS = library.data.CACHE_TEST_IDS

    # Update Output Paths in library.train and library.predict
    demo_model_path = os.path.join(DEMO_DIR, "best_model.pth")
    demo_submission_path = os.path.join(DEMO_DIR, "submission.csv")

    library.train.MODEL_SAVE_PATH = demo_model_path
    library.train.SUBMISSION_FILE = demo_submission_path
    library.predict.MODEL_SAVE_PATH = demo_model_path

    # Update Hyperparameters for Speed
    # Reduce Epochs to 1
    library.train.NUM_EPOCHS = 1
    # Reduce Batch Size to 4 (since we only have 8 training samples)
    library.data.BATCH_SIZE = 4
    library.train.BATCH_SIZE = (
        4  # Though train.py uses data.get_dataloaders, good to be safe
    )
    library.predict.BATCH_SIZE = 4

    # ==========================================
    # 4. Model Verification
    # ==========================================
    print(">>> [4/7] Verifying Model Architecture...")

    device = library.utils.get_device()
    model = library.model.SliceGroupedFusionNet()
    model.to(device)
    model.eval()

    # Create dummy input: (Batch=2, Channels=128, H=256, W=256)
    # Channels = 32 slices * 4 modalities = 128
    dummy_input = torch.randn(2, 128, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Assert output shape is (Batch, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("    Model forward pass successful. Output shape verified.")

    # ==========================================
    # 5. Run Training Pipeline
    # ==========================================
    print(">>> [5/7] Executing Training Pipeline (1 Epoch)...")

    # load_cached_data=False ensures we process our new demo subset
    # This will generate the .npy cache files in DEMO_DIR
    library.train.run_training(load_cached_data=False)

    # Verify model was saved
    assert os.path.exists(demo_model_path), "Model file was not saved!"
    print(f"    Training complete. Model saved to {demo_model_path}")

    # ==========================================
    # 6. Run Inference Pipeline
    # ==========================================
    print(">>> [6/7] Executing Inference Pipeline...")

    # We use the predict module which loads the saved model and generates predictions
    # We override the internal directory creation in library.predict via monkey-patching
    # library.predict.run_inference_and_save is imported as generate_submission in predict.py
    # However, predict.py has its own generate_submission function.

    # We need to ensure library.predict saves to our DEMO_DIR.
    # library.predict.generate_submission hardcodes "./submission/submission.csv" inside
    # the function body via `submission_dir = "./submission"`.
    # We cannot easily patch local variables.
    # However, library.train.run_training already calls generate_submission at the end.
    # Let's verify the submission generated by run_training first.

    assert os.path.exists(
        demo_submission_path
    ), "Submission file from training pipeline missing!"

    # Now let's explicitly test the library.predict module functionality.
    # Since we can't change the hardcoded path inside library.predict.generate_submission easily,
    # we will rely on the fact that we patched MODEL_SAVE_PATH and data paths.
    # It will write to ./submission/submission.csv. We will check that.

    print("    Running library.predict.generate_submission()...")
    library.predict.generate_submission(
        load_cached_data=True
    )  # Use cache generated by train step

    default_submission_path = "./submission/submission.csv"
    assert os.path.exists(default_submission_path), "Default submission file missing!"

    # ==========================================
    # 7. Final Validation
    # ==========================================
    print(">>> [7/7] Validating Results...")

    # Check content of the submission file generated by training loop
    df_sub = pd.read_csv(demo_submission_path)
    print(f"    Submission Head:\n{df_sub.head()}")

    assert len(df_sub) == n_test, f"Expected {n_test} predictions, found {len(df_sub)}"
    assert "BraTS21ID" in df_sub.columns
    assert "MGMT_value" in df_sub.columns
    assert df_sub["MGMT_value"].dtype == float

    print("\n>>> DEMO COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    run_demo()
