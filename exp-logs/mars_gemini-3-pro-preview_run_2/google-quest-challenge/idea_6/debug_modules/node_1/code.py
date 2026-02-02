import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import GranularSiameseModel, predict_fn
from library.engine import train_one_epoch, evaluate


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    seed_everything(42)

    print("--- Configuring for Fast Demonstration ---")
    # Override Config parameters to ensure quick execution
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 rows per split for demo
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.WORKING_DIR = "./working/demo_run"  # Separate dir for this demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n--- Loading Data (Debug Mode) ---")
    # debug=True forces re-processing of a small subset of data
    train_loader, val_loader, test_loader, cat_dims = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches:   {len(val_loader)}")
    print(f"Test Batches:  {len(test_loader)}")
    print(f"Categorical Dimensions: {cat_dims}")

    # Verify Data Structure
    # Fetch one batch to inspect
    batch = next(iter(train_loader))
    # Batch structure: [qa_ids, q_input_ids, q_att_mask, q_type_ids, q_title_mask, q_body_mask,
    #                   a_input_ids, a_att_mask, a_type_ids, cat_feats, targets]
    assert len(batch) == 11, "DataLoader batch should contain 11 elements."

    # Unpack for shape verification
    (
        qa_ids,
        q_input_ids,
        q_attention_mask,
        q_token_type_ids,
        q_title_mask,
        q_body_mask,
        a_input_ids,
        a_attention_mask,
        a_token_type_ids,
        cat_feats,
        targets,
    ) = batch

    assert q_input_ids.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Input IDs shape mismatch. Expected ({Config.TRAIN_BATCH_SIZE}, {Config.MAX_LEN}), got {q_input_ids.shape}"
    assert targets.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch. Expected ({Config.TRAIN_BATCH_SIZE}, 30), got {targets.shape}"

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- Initializing Model ---")
    device = Config.DEVICE
    print(f"Device: {device}")

    model = GranularSiameseModel(cat_dims=cat_dims).to(device)

    # Verify Forward Pass
    print("Verifying forward pass...")
    # Move batch elements to device
    batch_device = [b.to(device) for b in batch]

    # Perform inference (no grad)
    with torch.no_grad():
        preds = model(
            batch_device[1],  # q_input_ids
            batch_device[2],  # q_attention_mask
            batch_device[3],  # q_token_type_ids
            batch_device[4],  # q_title_mask
            batch_device[5],  # q_body_mask
            batch_device[6],  # a_input_ids
            batch_device[7],  # a_attention_mask
            batch_device[8],  # a_token_type_ids
            batch_device[9],  # cat_feats
        )

    assert preds.shape == (
        Config.TRAIN_BATCH_SIZE,
        30,
    ), "Prediction output shape mismatch."
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities in [0, 1]."
    print("Forward pass successful.")

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n--- Starting Training (1 Epoch) ---")
    # Initialize Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss resulted in NaN."
    assert train_loss > 0, "Training loss should be positive."

    # ==========================================
    # 5. Evaluation
    # ==========================================
    print("\n--- Evaluating on Validation Set ---")
    val_loss, val_spearman = evaluate(model, val_loader, device, criterion)
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Spearman Correlation: {val_spearman:.4f}")

    assert -1.0 <= val_spearman <= 1.0, "Spearman correlation must be between -1 and 1."

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n--- Generating Test Predictions ---")
    test_preds, test_ids = predict_fn(model, test_loader, device)

    print(f"Predictions shape: {test_preds.shape}")
    assert test_preds.shape[1] == 30, "Must predict 30 target columns."
    assert (
        len(test_ids) == test_preds.shape[0]
    ), "Mismatch between IDs and predictions count."

    print("Creating submission file...")
    sub_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    sub_df.insert(0, "qa_id", test_ids)

    # Save to working directory
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    sub_df.to_csv(submission_path, index=False)

    # Verify file existence
    if os.path.exists(submission_path):
        print(f"Successfully saved submission to {submission_path}")

        # Verify content format
        saved_df = pd.read_csv(submission_path)
        assert saved_df.shape == (
            len(test_loader.dataset),
            31,
        ), "Saved CSV has incorrect shape."
        assert "qa_id" in saved_df.columns, "qa_id column missing."
        assert (
            list(saved_df.columns[1:]) == Config.TARGET_COLS
        ), "Target columns mismatch."
    else:
        raise AssertionError("Submission file was not created.")

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
