import os
import sys
import torch
import numpy as np
import logging
import transformers
import warnings

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders
from library.model import SharedBottomMultiBranchModel
from library.engine import (
    get_optimizer_params,
    get_scheduler,
    train_one_epoch,
    validate,
    predict,
    set_backbone_freezing,
)

# Suppress warnings and verbose logs for clean output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("Initializing Demo...")

    # 1. Setup and Configuration Overrides for Speed
    # We override Config attributes to run a fast demo
    Config.EPOCHS = 2  # Run 2 epochs to test both frozen (warmup) and unfrozen states
    Config.WARMUP_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.ACCUMULATION_STEPS = 1  # Simplify for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny debug data

    # Setup directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # 2. Data Loading (Debug Mode)
    print("Loading Data (Debug Mode)...")
    # debug=True loads a very small subset (e.g., 100 rows)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Verify DataLoaders
    batch = next(iter(train_loader))
    print(f"Train Batch Keys: {batch.keys()}")
    assert "input_ids_q" in batch
    assert "targets" in batch
    assert batch["input_ids_q"].shape[0] == Config.TRAIN_BATCH_SIZE
    assert batch["targets"].shape[1] == Config.NUM_TARGETS
    print("Data loading verification passed.")

    # 3. Model Initialization
    print("Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = SharedBottomMultiBranchModel()
    model.to(device)

    # Verify Model Output Shape
    with torch.no_grad():
        dummy_q = batch["input_ids_q"].to(device)
        dummy_mask_q = batch["attention_mask_q"].to(device)
        dummy_a = batch["input_ids_a"].to(device)
        dummy_mask_a = batch["attention_mask_a"].to(device)

        outputs = model(dummy_q, dummy_mask_q, dummy_a, dummy_mask_a)
        assert outputs.shape == (Config.TRAIN_BATCH_SIZE, Config.NUM_TARGETS)
        print(f"Model output shape verified: {outputs.shape}")

    # 4. Optimizer and Scheduler
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    # Calculate training steps for 2 epochs on debug data
    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_scheduler(optimizer, num_train_steps)

    # 5. Training Loop
    print(f"Starting Training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Apply freezing logic explicitly before assertion
        set_backbone_freezing(model, epoch)

        # Check freezing logic
        if epoch < Config.WARMUP_EPOCHS:
            # Backbone should be frozen
            assert (
                not model.embeddings.word_embeddings.weight.requires_grad
            ), "Backbone should be frozen in warmup"
            assert model.head_final.weight.requires_grad, "Head should be trainable"
            print(f"Epoch {epoch+1}: Backbone Frozen (Warmup)")
        else:
            # We need to call train_one_epoch first, which handles the unfreezing logic internally
            # based on the epoch index passed to it.
            pass

        loss = train_one_epoch(model, optimizer, scheduler, train_loader, device, epoch)

        # Verify loss is valid
        assert not np.isnan(loss), "Training loss returned NaN"
        assert loss > 0, "Training loss should be positive"

        # Post-epoch check for unfreezing (if we just finished the warmup epoch and started the next)
        # Note: train_one_epoch sets the requires_grad flags at the START of the function.
        # So for epoch 1 (second epoch), it should have unfrozen them.
        if epoch >= Config.WARMUP_EPOCHS:
            print(f"Epoch {epoch+1}: Backbone Unfrozen")

    # 6. Validation
    print("Running Validation...")
    val_loss, spearman_score = validate(model, val_loader, device)

    assert isinstance(val_loss, float)
    assert -1.0 <= spearman_score <= 1.0, "Spearman correlation out of range"
    print(f"Validation passed. Score: {spearman_score:.4f}")

    # 7. Prediction
    print("Running Prediction on Test Set...")
    predictions = predict(model, test_loader, device)

    # Verify Predictions
    # debug=True slices test set to 50 samples
    expected_rows = 50
    assert predictions.shape == (expected_rows, Config.NUM_TARGETS)
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions must be in [0, 1]"
    print("Prediction verification passed.")

    # 8. Metric Logic Verification (Unit Test)
    print("Verifying Metric Logic...")
    # Create synthetic perfect correlation
    t_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    t_pred = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    score = compute_spearmanr(t_pred, t_true)
    assert np.isclose(score, 1.0), f"Expected 1.0 for perfect correlation, got {score}"

    # Create synthetic inverse correlation
    t_pred_inv = np.array([[0.5, 0.6], [0.3, 0.4], [0.1, 0.2]])
    score_inv = compute_spearmanr(t_pred_inv, t_true)
    assert np.isclose(
        score_inv, -1.0
    ), f"Expected -1.0 for inverse correlation, got {score_inv}"
    print("Metric logic verified.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
