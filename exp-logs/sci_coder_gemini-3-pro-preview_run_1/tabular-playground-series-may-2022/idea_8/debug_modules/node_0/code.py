import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DeGUTModel
from library.engine import train_one_epoch, evaluate, predict_and_submit


def main():
    print("Starting DeGUT Library Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")
    # Override Config parameters to ensure the demo runs quickly (< 1 hour)
    # We use DEBUG=True to sample a small subset of the data
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 32  # Small batch size for demo

    # Reduce model size to speed up initialization and forward passes for this demo
    Config.D_MODEL = 64
    Config.N_HEADS = 4
    Config.N_LAYERS = 2
    Config.DIM_FEEDFORWARD = 128

    # Setup directories (creates ./working/degut_model/cache etc.)
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, Reduced Model Size.")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data (Debug Mode)...")
    # debug=True forces sampling of the dataset and reprocessing (skips large cache)
    # load_cached_data=False ensures we don't try to load full-size cached files if they exist
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False, debug=Config.DEBUG
    )

    print(f"Vocab Size: {len(vocab)}")
    print(f"Train Batches: {len(train_loader)}")

    # Verify Data Structure
    batch = next(iter(train_loader))
    required_keys = [
        "num_features",
        "seq_features",
        "mask_num",
        "mask_seq",
        "target_cls",
        "target_num",
        "target_seq",
        "ids",
    ]

    print("Verifying batch keys...")
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Dimensions
    B, N_num = batch["num_features"].shape
    _, N_seq = batch["seq_features"].shape

    assert B <= Config.BATCH_SIZE, f"Batch size {B} exceeds config {Config.BATCH_SIZE}"
    assert batch["target_cls"].shape[0] == B, "Target shape mismatch"

    print(
        f"Batch verified. Batch Size: {B}, Num Features: {N_num}, Seq Length: {N_seq}"
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing DeGUT Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model using dimensions derived from data
    model = DeGUTModel(num_feats=N_num, vocab_size=len(vocab))
    model.to(device)

    # Basic check of model structure
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params} parameters.")
    assert total_params > 0, "Model has no parameters!"

    # -------------------------------------------------------------------------
    # 4. Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Forward Pass...")
    # Move inputs to device
    num_features = batch["num_features"].to(device)
    seq_features = batch["seq_features"].to(device)
    mask_num = batch["mask_num"].to(device)
    mask_seq = batch["mask_seq"].to(device)

    # Run forward pass
    outputs = model(
        num_features=num_features,
        seq_features=seq_features,
        mask_num=mask_num,
        mask_seq=mask_seq,
    )

    # Check outputs dictionary
    assert "logits_cls" in outputs
    assert "pred_num" in outputs
    assert "pred_seq" in outputs

    logits_cls = outputs["logits_cls"]
    pred_num = outputs["pred_num"]
    pred_seq = outputs["pred_seq"]

    # Verify output shapes
    assert logits_cls.shape == (B, 1), f"Logits shape mismatch: {logits_cls.shape}"
    assert pred_num.shape == (B, N_num), f"Pred Num shape mismatch: {pred_num.shape}"
    assert pred_seq.shape == (
        B,
        N_seq,
        len(vocab),
    ), f"Pred Seq shape mismatch: {pred_seq.shape}"

    print("Forward pass successful. Output shapes verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, total_steps=len(train_loader)
    )

    # Run one epoch (which is short in debug mode)
    avg_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, epoch=0
    )

    print(f"Training Epoch 1 Loss: {avg_loss:.5f}")
    assert not np.isnan(avg_loss), "Loss is NaN"
    assert avg_loss > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 6. Evaluation Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Evaluation...")
    val_auc = evaluate(model, val_loader, device)
    print(f"Validation AUC: {val_auc:.5f}")

    # AUC should be between 0 and 1
    assert 0.0 <= val_auc <= 1.0, f"Invalid AUC: {val_auc}"

    # -------------------------------------------------------------------------
    # 7. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Submission Generation...")

    # Save the model first (as predict_and_submit loads from disk)
    print(f"Saving model to {Config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file not found after saving"

    # Generate submission
    predict_and_submit(model, test_loader, device)

    # Verify output file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    assert "id" in df_sub.columns
    assert "target" in df_sub.columns
    assert len(df_sub) == len(test_loader.dataset), "Submission row count mismatch"

    # Check probability range
    probs = df_sub["target"].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of [0, 1] range"

    print("\nDemo Complete. All checks passed.")


if __name__ == "__main__":
    main()
