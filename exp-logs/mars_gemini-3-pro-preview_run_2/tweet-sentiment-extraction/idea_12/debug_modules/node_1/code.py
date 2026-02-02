import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data_factory import get_loaders
from library.modeling import TweetModel
from library.training_engine import train_fn, eval_fn
import library.inference_engine as inference_engine


def run_demo():
    print("--- Starting Demo Script ---")

    # ==========================================
    # 1. Configuration Override
    # ==========================================
    print("[1] Configuring environment for rapid demonstration...")

    # Set paths to a custom working directory to avoid conflicts with existing runs
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config parameters for speed and demonstration purposes
    Config.ARTIFACTS_DIR = os.path.join(DEMO_DIR, "artifacts")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 1
    Config.N_FOLDS = 1  # Only train 1 fold
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    # Use 'roberta-base' for speed; the library supports generic HF models
    Config.MODEL_BACKBONES = ["roberta-base"]

    os.makedirs(Config.ARTIFACTS_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Artifacts Dir: {Config.ARTIFACTS_DIR}")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[2] Verifying Data Loading...")

    # We use debug=True to create small cached parquet files.
    # load_cached_data=False forces the creation of these new debug files.
    train_loader, val_loader, test_loader = get_loaders(
        model_name="roberta-base",
        batch_size=Config.TRAIN_BATCH_SIZE,
        load_cached_data=False,
        debug=True,
    )

    # Verify Train Batch Structure
    try:
        batch = next(iter(train_loader))
        print("    Train batch fetched successfully.")
        print(f"    Keys: {list(batch.keys())}")

        # Assertions to verify data integrity
        required_keys = [
            "ids",
            "mask",
            "token_type_ids",
            "targets_start",
            "targets_end",
        ]
        for key in required_keys:
            if key not in batch:
                raise AssertionError(f"Missing key in batch: {key}")

        assert batch["ids"].shape[0] <= Config.TRAIN_BATCH_SIZE
        assert batch["ids"].shape[1] == Config.MAX_LEN
    except StopIteration:
        raise AssertionError("Train loader is empty!")
    except Exception as e:
        raise AssertionError(f"Data loading failed: {e}")

    # ==========================================
    # 3. Model Initialization Verification
    # ==========================================
    print("\n[3] Verifying Model Initialization...")
    model = TweetModel("roberta-base")
    model.to(device)

    # Test Forward Pass
    ids = batch["ids"].to(device)
    mask = batch["mask"].to(device)
    tt_ids = batch["token_type_ids"].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(ids, mask, tt_ids)

    print(f"    Logits Shape: {start_logits.shape}")

    # Verify output shapes match (Batch_Size, Max_Len)
    assert start_logits.shape == (ids.shape[0], Config.MAX_LEN)
    assert end_logits.shape == (ids.shape[0], Config.MAX_LEN)
    print("    Forward pass successful.")

    # ==========================================
    # 4. Training Loop Verification
    # ==========================================
    print("\n[4] Verifying Training Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch (on the 50 sample subset)
    train_loss = train_fn(train_loader, model, optimizer, device)
    print(f"    Training Loss: {train_loss:.4f}")

    if np.isnan(train_loss):
        raise AssertionError("Training loss is NaN.")

    # ==========================================
    # 5. Evaluation Verification
    # ==========================================
    print("\n[5] Verifying Evaluation Step...")
    val_loss, val_jaccard = eval_fn(val_loader, model, device)
    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val Jaccard: {val_jaccard:.4f}")

    if np.isnan(val_loss) or np.isnan(val_jaccard):
        raise AssertionError("Validation metrics are NaN.")

    # ==========================================
    # 6. Saving Model for Inference
    # ==========================================
    print("\n[6] Saving Model for Inference...")
    # The inference engine looks for specific filenames: {model_name}_fold_{fold}.pth
    # We are using 'roberta-base' and fold 0.
    model_filename = "roberta-base_fold_0.pth"
    save_path = os.path.join(Config.ARTIFACTS_DIR, model_filename)
    torch.save(model.state_dict(), save_path)

    if not os.path.exists(save_path):
        raise AssertionError(f"Failed to save model to {save_path}")
    print(f"    Model saved to {save_path}")

    # ==========================================
    # 7. Inference Engine Verification
    # ==========================================
    print("\n[7] Verifying Inference Engine...")

    # generate_submission will:
    # 1. Read metadata/test.csv (full test set IDs)
    # 2. Call get_loaders(..., load_cached_data=True)
    #    -> This will load the cached 'test' parquet we created in step 2 (which is DEBUG size, 50 samples)
    # 3. Predict on those 50 samples
    # 4. Fill the rest of the IDs with default predictions (neutral heuristic or full text)
    # 5. Save submission.csv

    inference_engine.generate_submission(device=device, load_cached_data=True)

    if not os.path.exists(Config.SUBMISSION_FILE):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_FILE}")

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission generated with {len(df_sub)} rows.")

    # Check format
    expected_cols = ["textID", "selected_text"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {df_sub.columns.tolist()}"
        )

    # Verify that we have non-empty predictions
    non_empty = df_sub["selected_text"].astype(str).str.len() > 0
    print(f"    Non-empty predictions: {non_empty.sum()}/{len(df_sub)}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
