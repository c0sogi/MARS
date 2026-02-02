import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MultiTaskRoBERTa
from library.loss import HybridContrastiveLoss
from library.metrics import JigsawEvaluator
from library.engine import train_one_epoch, valid_fn, inference_fn


def run_demonstration():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for speed (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 1000  # Small subset for demonstration
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[2] Loading Data (forcing reprocessing for demo)...")

    # We set load_cached_data=False to demonstrate the processing pipeline
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Verification: Check Batch Structure
    sample_batch = next(iter(train_loader))
    required_keys = ["input_ids", "attention_mask", "target", "identities"]
    for key in required_keys:
        assert key in sample_batch, f"Missing key {key} in batch"

    print("    Batch structure verified.")
    print(f"    Input IDs Shape: {sample_batch['input_ids'].shape}")
    print(f"    Targets Shape:   {sample_batch['target'].shape}")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = MultiTaskRoBERTa(Config)
    model.to(device)

    # Verification: Forward Pass
    print("    Running dummy forward pass...")
    input_ids = sample_batch["input_ids"].to(device)
    mask = sample_batch["attention_mask"].to(device)

    with torch.no_grad():
        tox_logits, ident_logits = model(input_ids, mask)

    # Assert Output Shapes
    # Toxicity: (Batch, 1)
    assert tox_logits.shape == (
        input_ids.shape[0],
        1,
    ), f"Toxicity logits shape mismatch: {tox_logits.shape}"
    # Identity: (Batch, Num_Identities)
    assert ident_logits.shape == (
        input_ids.shape[0],
        len(Config.IDENTITY_COLUMNS),
    ), f"Identity logits shape mismatch: {ident_logits.shape}"

    print("    Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 4. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Loss Function...")
    loss_fn = HybridContrastiveLoss()

    targets = sample_batch["target"].to(device)
    identities = sample_batch["identities"].to(device)

    # Calculate loss on dummy batch (requires grad usually, but here just checking computation)
    # We use the logits from the previous step (detached, so we re-run with grad enabled for completeness)
    model.train()
    tox_logits, ident_logits = model(input_ids, mask)
    loss = loss_fn(tox_logits, ident_logits, targets, identities)

    assert torch.is_tensor(loss), "Loss is not a tensor"
    assert not torch.isnan(loss).item(), "Loss returned NaN"
    print(f"    Calculated Loss: {loss.item():.4f}")

    # --------------------------------------------------------------------------
    # 5. Metric Logic Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Metric Calculation Logic...")
    # Create synthetic data to test JigsawEvaluator
    evaluator = JigsawEvaluator()

    # Synthetic Batch 1
    # Logits: Positive values -> Prob > 0.5
    syn_logits = torch.tensor(
        [[2.0], [-2.0], [2.0], [-2.0]]
    )  # Preds: High, Low, High, Low
    syn_targets = torch.tensor([1.0, 0.0, 0.0, 1.0])  # True:  1,    0,    0,    1
    # Identities: 9 columns. Let's set the first identity (Male) for some samples
    syn_ident = torch.zeros((4, 9))
    syn_ident[0, 0] = 1.0  # Toxic, Male
    syn_ident[1, 0] = 1.0  # Non-Toxic, Male

    evaluator.update(syn_logits, syn_targets, syn_ident)
    metrics = evaluator.compute()

    assert "score" in metrics, "Metric dict missing 'score'"
    assert "overall_auc" in metrics, "Metric dict missing 'overall_auc'"
    print(f"    Synthetic Check Score: {metrics['score']:.4f}")
    print("    Metric calculation logic verified.")

    # --------------------------------------------------------------------------
    # 6. Training & Validation Loop
    # --------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    # Setup Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Setup Scheduler
    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(Config.WARMUP_RATIO * num_train_steps),
        num_training_steps=num_train_steps,
    )

    # Train
    avg_train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, loss_fn
    )
    print(f"    Epoch 1 Training Loss: {avg_train_loss:.4f}")

    # Validate
    print("    Running Validation...")
    avg_val_loss, val_metrics = valid_fn(model, val_loader, device, loss_fn)

    print(f"    Validation Loss: {avg_val_loss:.4f}")
    print(f"    Validation Score: {val_metrics['score']:.4f}")
    print(f"    Overall AUC:      {val_metrics['overall_auc']:.4f}")

    # Save Model (Mock)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"    Model saved to {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 7. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[7] Running Inference on Test Set...")

    test_preds = inference_fn(model, test_loader, device)

    assert len(test_preds) == len(
        test_ids
    ), f"Prediction count ({len(test_preds)}) matches ID count ({len(test_ids)})"

    print("    Generating Submission File...")
    submission_df = pd.DataFrame({"id": test_ids, "prediction": test_preds})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # Verify file content
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    First 3 rows of submission:\n{saved_df.head(3)}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
