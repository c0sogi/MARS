import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_dataloaders
from library.model import ContextualDualEncoder
from library.train import train_epoch, validate_epoch, predict


def main():
    print("=== Starting Demonstration of StackExchange QA Library ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    seed_everything(42)

    # Override Config for speed
    Config.MAX_LEN = 32  # Reduce sequence length for faster tokenization/inference
    Config.TRAIN_BATCH_SIZE = 4
    Config.VAL_BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_run/"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    MAX_LEN set to: {Config.MAX_LEN}")
    print(f"    BATCH_SIZE set to: {Config.TRAIN_BATCH_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Verify Metric Calculation
    print("\n[2] Verifying Metric Calculation (Spearman's Rank Correlation)...")
    # Create synthetic data: 2 samples, 3 targets
    # Col 0: Perfect positive correlation
    # Col 1: Perfect negative correlation
    # Col 2: Random/Constant
    preds = torch.tensor([[0.1, 0.9, 0.5], [0.2, 0.8, 0.5], [0.3, 0.7, 0.5]])
    targets = torch.tensor([[0.1, 0.9, 0.0], [0.2, 0.8, 0.0], [0.3, 0.7, 0.0]])

    score = compute_spearman_metric(preds, targets)
    print(f"    Computed Spearman Score: {score:.4f}")

    # Expected: (1.0 + (-1.0) + NaN/Undefined) / 3 or handled gracefully.
    # Note: scipy's spearmanr returns NaN for constant input.
    # The utils function uses np.nanmean, so if the constant column returns nan, it is ignored.
    # Avg of 1.0 and -1.0 is 0.0.

    # Let's test a simpler case to be strictly deterministic with assertions
    # 2 columns, both perfect correlation
    p_simple = torch.tensor([[0.1, 0.5], [0.2, 0.6]])
    t_simple = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    score_simple = compute_spearman_metric(p_simple, t_simple)
    assert np.isclose(score_simple, 1.0), f"Expected 1.0, got {score_simple}"
    print(
        "    Assertion Passed: Metric calculation is correct for perfect correlation."
    )

    # 3. Data Loading
    print("\n[3] Initializing DataLoaders (Debug Mode)...")
    # debug=True loads a small subset (100 rows)
    # load_cached_data=False forces re-tokenization with our new MAX_LEN=32
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    q_ids = batch["q_input_ids"]
    a_ids = batch["a_input_ids"]
    labels = batch["labels"]

    print(f"    Batch keys: {list(batch.keys())}")
    print(f"    Q Input Shape: {q_ids.shape}")
    print(f"    Labels Shape:  {labels.shape}")

    # Assertions
    assert q_ids.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), "Incorrect Q input shape"
    assert a_ids.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), "Incorrect A input shape"
    assert labels.shape == (Config.TRAIN_BATCH_SIZE, 30), "Incorrect Labels shape"
    assert labels.dtype == torch.float32, "Labels should be float32"
    print("    Assertion Passed: DataLoader outputs correct shapes and types.")

    # 4. Model Instantiation & Forward Pass
    print("\n[4] Instantiating ContextualDualEncoder...")
    model = ContextualDualEncoder()
    model.to(Config.DEVICE)

    print("    Performing Forward Pass...")
    # Move batch to device
    q_ids = q_ids.to(Config.DEVICE)
    q_mask = batch["q_attention_mask"].to(Config.DEVICE)
    a_ids = a_ids.to(Config.DEVICE)
    a_mask = batch["a_attention_mask"].to(Config.DEVICE)

    logits = model(q_ids, q_mask, a_ids, a_mask)

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (Config.TRAIN_BATCH_SIZE, 30), "Logits shape mismatch"
    print("    Assertion Passed: Model forward pass produces correct output shape.")

    # Verify Freezing Logic
    print("    Verifying Backbone Freezing...")
    model.freeze_backbone()
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            raise AssertionError(f"Backbone parameter {name} should be frozen!")
    print("    Assertion Passed: Backbone successfully frozen.")

    model.unfreeze_backbone()
    # Check one parameter to ensure unfreeze worked
    first_param = next(model.backbone.parameters())
    assert first_param.requires_grad, "Backbone should be unfrozen!"
    print("    Assertion Passed: Backbone successfully unfrozen.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Epoch (1 Epoch, Head Only)...")
    model.freeze_backbone()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # Run training epoch
    avg_loss = train_epoch(
        model, train_loader, optimizer, None, criterion, Config.DEVICE
    )
    print(f"    Training Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # Run validation epoch
    print("    Running Validation...")
    val_loss, val_spearman = validate_epoch(model, val_loader, criterion, Config.DEVICE)
    print(f"    Val Loss: {val_loss:.4f}, Val Spearman: {val_spearman:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    # Spearman might be NaN if model outputs constant values initially, which is possible with random init.
    # However, validate_epoch handles NaNs internally in compute_spearman_metric (returns mean of valid cols).

    # 6. Inference Demonstration
    print("\n[6] Running Inference on Test Set...")
    predictions = predict(model, test_loader, Config.DEVICE)

    print(f"    Predictions Shape: {predictions.shape}")
    print(f"    Sample Prediction (First row, first 5 cols): {predictions[0, :5]}")

    # Load test metadata to check count
    import pandas as pd

    test_df_len = len(pd.read_csv(Config.TEST_PATH).iloc[:100])  # Debug mode uses 100

    # Note: The test loader in debug mode might drop the last batch if configured,
    # but usually validation/test loaders don't drop last.
    # Let's verify the number of predictions matches the number of samples in the loader's dataset
    assert predictions.shape[0] == len(
        test_loader.dataset
    ), f"Prediction count {predictions.shape[0]} mismatch with dataset size {len(test_loader.dataset)}"
    assert predictions.shape[1] == 30, "Prediction column count mismatch"
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions not in [0, 1] range"
    print("    Assertion Passed: Inference output valid.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
