import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import HybridDeberta
from library.trainer import train_fn, eval_fn, predict_fn


def main():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    print("1. Configuring environment for demonstration...")

    # Override Config for speed and isolation
    Config.debug = True  # Limits dataset to 100 rows for speed
    Config.epochs = 1  # Run only 1 epoch
    Config.train_batch_size = 4  # Small batch size
    Config.valid_batch_size = 8  # Small batch size
    Config.working_dir = "./working/demo_run"  # Isolated working directory
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set random seeds for reproducibility
    seed_everything(Config.seed)
    print(f"   Debug Mode: {Config.debug}")
    print(f"   Working Directory: {Config.working_dir}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n2. Initializing DataLoaders...")
    # load_cached_data=False forces reprocessing to demonstrate the pipeline
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify DataLoader output
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("DataLoader is empty!")

    print(f"   Batch keys: {list(batch.keys())}")

    # Assertions to verify batch structure
    expected_keys = [
        "view1_input_ids",
        "view1_attention_mask",
        "view2_input_ids",
        "view2_attention_mask",
        "view2_q_mask",
        "view2_a_mask",
        "labels",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key {key} in batch"

    # Verify shapes
    B = Config.train_batch_size
    assert (
        batch["view1_input_ids"].shape[0] == B
    ), f"Batch size mismatch. Expected {B}, got {batch['view1_input_ids'].shape[0]}"
    assert batch["labels"].shape == (
        B,
        30,
    ), f"Label shape mismatch. Expected ({B}, 30), got {batch['labels'].shape}"
    print("   DataLoader structure and shapes verified.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n3. Initializing Model...")
    device = Config.device
    model = HybridDeberta()
    model.to(device)
    print(f"   Model initialized and moved to {device}.")

    # ==========================================
    # 4. Forward Pass Verification
    # ==========================================
    print("\n4. Verifying Forward Pass...")
    model.eval()
    with torch.no_grad():
        # Prepare inputs
        inputs = {
            "view1_input_ids": batch["view1_input_ids"].to(device),
            "view1_attention_mask": batch["view1_attention_mask"].to(device),
            "view2_input_ids": batch["view2_input_ids"].to(device),
            "view2_attention_mask": batch["view2_attention_mask"].to(device),
            "view2_q_mask": batch["view2_q_mask"].to(device),
            "view2_a_mask": batch["view2_a_mask"].to(device),
        }
        if "view2_token_type_ids" in batch:
            inputs["view2_token_type_ids"] = batch["view2_token_type_ids"].to(device)

        logits = model(**inputs)

    assert logits.shape == (
        B,
        30,
    ), f"Logits shape mismatch. Expected ({B}, 30), got {logits.shape}"
    print("   Forward pass successful. Output shape matches expected targets.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n5. Running Training Loop (1 Epoch)...")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)

    # Run training function
    train_loss = train_fn(
        train_loader, model, criterion, optimizer, scheduler, device, epoch=0
    )

    print(f"   Train Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN."

    # ==========================================
    # 6. Validation Loop Demonstration
    # ==========================================
    print("\n6. Running Validation Loop...")
    val_loss, val_score = eval_fn(val_loader, model, criterion, device)

    print(f"   Val Loss: {val_loss:.6f}")
    print(f"   Val Spearman Correlation: {val_score:.6f}")
    assert not np.isnan(val_loss), "Validation loss resulted in NaN."

    # ==========================================
    # 7. Prediction & Submission
    # ==========================================
    print("\n7. Running Prediction on Test Set...")
    test_preds = predict_fn(test_loader, model, device)

    # In debug mode, the test set is also truncated to 100 rows (or fewer if total < 100)
    # The original test.csv has 608 rows. Debug mode takes head(100).
    expected_test_rows = 100
    assert test_preds.shape == (
        expected_test_rows,
        30,
    ), f"Prediction shape mismatch. Expected ({expected_test_rows}, 30), got {test_preds.shape}"

    print("   Predictions generated successfully.")

    print("\n8. Generating Submission File...")
    # Load test metadata to get qa_ids matching the predictions
    test_df = pd.read_csv(Config.test_path)
    if Config.debug:
        test_df = test_df.head(100)

    qa_ids = test_df["qa_id"].values

    # Create submission DataFrame
    submission_df = pd.DataFrame(test_preds, columns=Config.target_cols)
    submission_df.insert(0, "qa_id", qa_ids)

    # Save submission
    submission_path = os.path.join(Config.working_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file was not created."
    print(f"   Submission saved to {submission_path}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
