import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library import config
from library import dataset
from library import model_components
from library import model as lib_model
from library import loss
from library import train_utils


def main():
    print("Initializing Demonstration...")

    # 1. Configuration & Setup
    # Set random seed for reproducibility
    config.set_seed(42)
    device = config.get_device()
    print(f"Device: {device}")

    # Modify config for rapid demonstration
    config.EPOCHS = 2
    config.DEBUG = True
    config.BATCH_SIZE = 8  # Small batch size for demo
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading Verification
    print("\n[Step 1] Verifying Data Loading...")
    # Load training dataset in debug mode
    train_ds = dataset.RNADataset(split="train", debug=True)
    print(f"Train Dataset Length (Debug): {len(train_ds)}")

    # Validate a single sample
    x, p, y = train_ds[0]
    print(f"Sample Shapes -> Input: {x.shape}, Partner: {p.shape}, Target: {y.shape}")

    # Assertions for data shapes
    # Input: (Channels=18, Length=107)
    assert x.shape == (18, 107), f"Incorrect input shape: {x.shape}"
    # Partner: (Length=107)
    assert p.shape == (107,), f"Incorrect partner shape: {p.shape}"
    # Target: (Channels=5, Length=107)
    assert y.shape == (5, 107), f"Incorrect target shape: {y.shape}"

    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    # 3. Model Component Verification
    print("\n[Step 2] Verifying Model Components...")
    # Create dummy batch
    batch_x = x.unsqueeze(0).to(device)  # (1, 18, 107)
    batch_p = p.unsqueeze(0).to(device)  # (1, 107)
    batch_y = y.unsqueeze(0).to(device)  # (1, 5, 107)

    # A. HybridInputStem
    stem = model_components.HybridInputStem(
        in_channels=18, context_channels=config.GROWTH_RATE
    ).to(device)
    out_stem = stem(batch_x)
    expected_stem_channels = 18 + config.GROWTH_RATE
    assert out_stem.shape == (
        1,
        expected_stem_channels,
        107,
    ), "HybridInputStem output shape mismatch"
    print("HybridInputStem: OK")

    # B. DenseDilatedBackbone
    backbone = model_components.DenseDilatedBackbone(
        in_channels=expected_stem_channels,
        growth_rate=config.GROWTH_RATE,
        dilations=config.DILATIONS,
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT,
    ).to(device)
    z = backbone(out_stem)
    assert z.shape == (
        1,
        config.LATENT_DIM,
        107,
    ), "DenseDilatedBackbone output shape mismatch"
    print("DenseDilatedBackbone: OK")

    # C. FeedbackStem
    fb_stem = model_components.FeedbackStem(in_channels=5, hidden_dim=32).to(device)
    e_fb = fb_stem(batch_y)
    assert e_fb.shape == (1, 32, 107), "FeedbackStem output shape mismatch"
    print("FeedbackStem: OK")

    # D. InteractionModule
    interaction = model_components.InteractionModule(
        dim_z=config.LATENT_DIM, dim_fb=32, rnn_hidden=config.RNN_HIDDEN, num_targets=5
    ).to(device)
    y_pred = interaction(z, e_fb, batch_p)
    assert y_pred.shape == (1, 5, 107), "InteractionModule output shape mismatch"
    print("InteractionModule: OK")

    # E. Full HI-GFDN Model
    full_model = lib_model.HIGFDN().to(device)
    # Test forward pass with no previous prediction (initial state)
    y_out_initial = full_model(batch_x, batch_p)
    assert y_out_initial.shape == (1, 5, 107), "Full Model forward pass mismatch"
    print("Full HI-GFDN Model: OK")

    # 4. Loss Function Verification
    print("\n[Step 3] Verifying Loss Function...")
    loss_fn = loss.MaskedMCRMSELoss()

    # Test Case: Predictions = 1.0, Targets = 0.0
    # Scored columns: 0, 1, 3. Scored length: 68.
    # Error = 1.0, Squared Error = 1.0, RMSE = 1.0, MCRMSE = 1.0

    # Create dummy tensors
    # Shape (Batch=2, Channels=5, Length=107)
    dummy_preds = torch.ones((2, 5, 107), dtype=torch.float32)
    dummy_targets = torch.zeros((2, 5, 107), dtype=torch.float32)

    calculated_loss = loss_fn(dummy_preds, dummy_targets)
    print(f"Calculated Loss: {calculated_loss.item():.6f}")

    # Assert with small tolerance
    assert (
        abs(calculated_loss.item() - 1.0) < 1e-5
    ), "Loss function calculation incorrect"
    print("MaskedMCRMSELoss: OK")

    # 5. Training Loop Execution
    print("\n[Step 4] Executing Training Loop (Demo)...")

    # Load Validation Data
    val_ds = dataset.RNADataset(split="val", debug=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Run Training using utility function
    # This saves the best model to config.MODEL_PATH
    train_utils.run_training(
        full_model,
        train_loader,
        val_loader,
        device,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
    )

    # Verify Model Artifact
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_PATH}")
    print("Training completed and model saved.")

    # 6. Submission Generation
    print("\n[Step 5] Generating Submission...")

    # Run the submission generation function from library
    # This loads the test set (debug mode) and the saved model
    lib_model.generate_submission()

    # Verify Submission File
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {df_sub.shape}")

    # Verify row count
    # Debug subset size is 100 (or length of test file if smaller)
    # Test file has 240 lines, so debug=100 applies.
    # Rows = 100 samples * 107 positions = 10700
    test_ds_len = len(dataset.RNADataset(split="test", debug=True))
    expected_rows = test_ds_len * 107

    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, found {len(df_sub)}"

    # Verify columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("Submission generated and verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
