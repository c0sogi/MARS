import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.data import prepare_data, PhraseDataset
from library.model import DebertaV3Regressor
from library.trainer import run_fold
from library.inference import inference_fn


def main():
    # =========================================================================
    # 1. Configuration Overrides for Rapid Demo
    # =========================================================================
    print("[1/7] Configuring environment for rapid demonstration...")

    # Override Config parameters to ensure the script runs quickly (within minutes)
    Config.debug = True
    Config.epochs = 1
    Config.n_folds = 2  # We will only utilize Fold 0 for this demo
    Config.batch_size = 4
    Config.gradient_accumulation_steps = 1
    Config.max_length = 64  # Reduced sequence length for speed

    # Redirect working directories to a separate demo folder to avoid overwriting real work
    Config.working_dir = "./working/demo_run_script"
    Config.models_dir = os.path.join(Config.working_dir, "models")
    Config.predictions_dir = os.path.join(Config.working_dir, "predictions")
    Config.submission_dir = os.path.join(Config.working_dir, "submission")
    Config.cache_dir = os.path.join(Config.working_dir, "cache")

    # Re-run setup to create these new directories
    Config.setup()

    # Set seeds
    seed_everything(Config.seed)

    print(f"Working Directory: {Config.working_dir}")
    print(f"Device: {Config.device}")

    # =========================================================================
    # 2. Data Loading and Subsetting
    # =========================================================================
    print("\n[2/7] Preparing and subsetting data...")

    # Load data (force processing to ensure we aren't reading old caches incompatible with this run)
    # Note: prepare_data loads the full metadata. We will slice it immediately after.
    train_full, test_full = prepare_data(load_cached_data=False)

    # Create tiny subsets for the demo
    # We take 50 samples for training, 20 for validation (simulated), 20 for testing
    train_subset = train_full.iloc[:50].reset_index(drop=True)
    val_subset = train_full.iloc[50:70].reset_index(drop=True)
    test_subset = test_full.iloc[:20].reset_index(drop=True)

    print(f"Train subset shape: {train_subset.shape}")
    print(f"Val subset shape: {val_subset.shape}")
    print(f"Test subset shape: {test_subset.shape}")

    # Validation: Check essential columns
    required_cols = ["id", "anchor", "target", "context_text"]
    for col in required_cols:
        assert col in train_subset.columns, f"Missing column {col} in train data"
        assert col in test_subset.columns, f"Missing column {col} in test data"
    assert "score" in train_subset.columns, "Missing 'score' column in train data"

    # =========================================================================
    # 3. Dataset & DataLoader Verification
    # =========================================================================
    print("\n[3/7] Verifying Dataset logic...")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Instantiate dataset
    ds_demo = PhraseDataset(
        train_subset, tokenizer, max_length=Config.max_length, is_train=True
    )

    # Fetch one item
    item = ds_demo[0]

    # Assertions to verify structure
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "token_type_ids" in item
    assert "labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["input_ids"].shape[0] == Config.max_length

    print("Dataset structure verified.")

    # =========================================================================
    # 4. Model Architecture Verification
    # =========================================================================
    print("\n[4/7] Verifying Model architecture...")

    model = DebertaV3Regressor(Config.model_name, pretrained=True)
    model.to(Config.device)
    model.eval()

    # Prepare a dummy batch
    dummy_input_ids = item["input_ids"].unsqueeze(0).to(Config.device)
    dummy_mask = item["attention_mask"].unsqueeze(0).to(Config.device)
    dummy_token_type = item["token_type_ids"].unsqueeze(0).to(Config.device)

    # Run forward pass
    with torch.no_grad():
        output = model(dummy_input_ids, dummy_mask, dummy_token_type)

    # Assert output shape (Batch Size,)
    assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"
    print("Model forward pass successful.")

    # Clean up memory
    del model, output
    torch.cuda.empty_cache()

    # =========================================================================
    # 5. Training Loop Simulation
    # =========================================================================
    print("\n[5/7] Running training loop (Fold 0)...")

    # run_fold performs training and validation, and saves the best model
    # We pass our subsets directly
    best_score = run_fold(fold=0, train_df=train_subset, valid_df=val_subset)

    print(f"Training completed. Best Validation Score: {best_score:.4f}")

    # Verify model file was created
    model_path = os.path.join(Config.models_dir, "model_fold_0.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    # =========================================================================
    # 6. Inference Simulation
    # =========================================================================
    print("\n[6/7] Running inference on test subset...")

    # Load the trained model
    model = DebertaV3Regressor(Config.model_name, pretrained=False)
    state_dict = torch.load(model_path, map_location=Config.device)
    model.load_state_dict(state_dict)
    model.to(Config.device)

    # Create test dataloader
    test_ds = PhraseDataset(
        test_subset, tokenizer, max_length=Config.max_length, is_train=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )

    # Run inference
    predictions = inference_fn(model, test_loader, Config.device)

    # Verify predictions
    assert len(predictions) == len(test_subset), "Prediction count mismatch"
    assert np.all(
        (predictions >= -0.5) & (predictions <= 1.5)
    ), "Predictions wildly out of range"

    # Clip for submission
    predictions = np.clip(predictions, 0, 1)
    print(f"Inference successful. Generated {len(predictions)} predictions.")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    print("\n[7/7] Generating submission file...")

    submission = pd.DataFrame({"id": test_subset["id"], "score": predictions})

    submission_path = os.path.join(Config.submission_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created"

    print("Head of submission:")
    print(submission.head())
    print(f"\nDemo completed successfully. Output saved to {submission_path}")


if __name__ == "__main__":
    main()
