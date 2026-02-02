import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import JigsawTransformer
from library.engine import train_fn, eval_fn, inference_fn


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # --------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set DEBUG to True to use a small subset (5000 samples) for speed
    Config.DEBUG = True

    # Reduce epochs to 1 for demonstration purposes
    Config.EPOCHS = 1

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Working Dir: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    print("\n=== Data Preparation ===")
    # We set load_cached_data=False to force reprocessing with DEBUG=True (subsampling)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verification: Check if dataloaders are loaded and have correct batch size
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Verify we are working with the subsampled dataset
    # 5000 samples / 64 batch_size ~= 78 batches
    if Config.DEBUG:
        assert (
            len(train_loader.dataset) <= 5000
        ), "Train dataset should be subsampled in DEBUG mode"

    # Inspect one batch to verify structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "target" in sample_batch
    assert "identities" in sample_batch

    print("DataLoaders initialized and verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n=== Model Initialization ===")
    model = JigsawTransformer()
    model.to(Config.DEVICE)

    # Verification: Check if model has the expected heads
    assert hasattr(model, "toxicity_head"), "Model missing toxicity_head"
    assert hasattr(model, "identity_head"), "Model missing identity_head"

    # Check LoRA parameters if enabled
    if Config.USE_LORA:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Total Parameters:     {total_params:,}")
        # LoRA should have significantly fewer trainable params than total
        assert trainable_params < total_params, "LoRA should not train all parameters"

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print("\n=== Starting Training (1 Epoch) ===")

    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Setup Scheduler
    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Run Training
    avg_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)
    print(f"Epoch 1/1 - Average Loss: {avg_loss:.4f}")

    # Verification: Loss should be a finite float
    assert isinstance(avg_loss, float)
    assert np.isfinite(avg_loss)

    # Save the model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 5. Evaluation
    # --------------------------------------------------------------------------
    print("\n=== Running Evaluation ===")
    metrics = eval_fn(val_loader, model, Config.DEVICE)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Verification: Check for required metric keys
    required_keys = ["score", "overall_auc", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"

    # Note: In DEBUG mode with very few samples, some AUCs might be NaN if a subgroup
    # isn't present in the validation subset. This is handled by the metric class (returns NaN/0),
    # but the key should still exist.

    # --------------------------------------------------------------------------
    # 6. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n=== Running Inference ===")
    predictions = inference_fn(test_loader, model, Config.DEVICE)

    print(f"Predictions generated: {len(predictions)}")

    # Verification: Predictions count should match test dataset size
    assert len(predictions) == len(
        test_loader.dataset
    ), f"Mismatch: {len(predictions)} preds vs {len(test_loader.dataset)} inputs"

    # Create Submission DataFrame
    # Note: In DEBUG mode, we must ensure we align with the IDs processed.
    # The tokenize_and_cache function saves IDs to .npy files.
    # We load the test IDs directly from the cache to ensure alignment.
    test_ids_path = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    # The cache saves token IDs (input_ids), not the original CSV IDs.
    # However, the metadata file used for 'test' in DEBUG mode is subsampled
    # in tokenize_and_cache, but the function doesn't return the original CSV IDs directly.
    # For this demo, we will read the test metadata again, applying the same debug logic
    # to retrieve the correct CSV IDs.

    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)
    if Config.DEBUG:
        # Match the slicing logic in library.data.tokenize_and_cache
        test_meta_df = test_meta_df.iloc[:5000].copy()

    submission_df = pd.DataFrame({"id": test_meta_df["id"], "prediction": predictions})

    # Save Submission
    submission_df.to_csv(Config.PREDICTION_SAVE_PATH, index=False)
    print(f"Submission saved to {Config.PREDICTION_SAVE_PATH}")

    # Final Verification
    assert os.path.exists(Config.PREDICTION_SAVE_PATH)
    print("\n=== Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
