import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.layers import SinusoidalPairingEmbedding, ScalarMixture
from library.model import ZoneoutWideResBiGRU
from library.engine import train_model, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration of RNA Degradation Prediction Pipeline ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1/5] Configuring environment for rapid demonstration...")

    # Modify Config for speed
    Config.DEBUG = True  # Use small subset of data
    Config.DEBUG_SUBSET_SIZE = 50  # Only 50 samples
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"      Device: {device}")
    print(f"      Debug Mode: {Config.DEBUG}")
    print(f"      Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Component Verification (Unit Tests)
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying individual components...")

    # A. Verify Metric (MCRMSE)
    y_true = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y_pred_perfect = y_true.copy()
    y_pred_off = y_true + 1.0  # RMSE should be 1.0 for each column -> MCRMSE = 1.0

    score_perfect = calculate_mcrmse(y_true, y_pred_perfect)
    score_off = calculate_mcrmse(y_true, y_pred_off)

    assert score_perfect == 0.0, f"Metric failed: Expected 0.0, got {score_perfect}"
    assert abs(score_off - 1.0) < 1e-6, f"Metric failed: Expected 1.0, got {score_off}"
    print("      Metric calculation verified.")

    # B. Verify SinusoidalPairingEmbedding
    # Input shape: (Batch, Seq_Len) -> Output: (Batch, Seq_Len, Embed_Dim)
    dummy_dists = torch.randn(4, 107)
    embed_layer = SinusoidalPairingEmbedding(embed_dim=128)
    out_embed = embed_layer(dummy_dists)

    assert out_embed.shape == (
        4,
        107,
        128,
    ), f"Embedding shape mismatch: {out_embed.shape}"
    print("      SinusoidalPairingEmbedding verified.")

    # C. Verify ScalarMixture
    # Input: List of tensors -> Output: Single tensor (weighted sum)
    num_layers = 3
    dummy_inputs = [torch.ones(4, 107, 16) * i for i in range(num_layers)]
    mixture_layer = ScalarMixture(num_layers=num_layers)
    out_mix = mixture_layer(dummy_inputs)

    assert out_mix.shape == (
        4,
        107,
        16,
    ), f"ScalarMixture shape mismatch: {out_mix.shape}"
    # Check that weights sum to 1 (softmax is applied internally)
    # Since initialized uniformly, output should be roughly mean of inputs if weights were equal,
    # but they are learnable. We just check it runs without error and preserves shape.
    print("      ScalarMixture verified.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[3/5] Initializing DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=Config.DEBUG  # Force processing for demo
    )

    # Inspect a batch
    batch = next(iter(train_loader))
    seq = batch["sequence"]
    targets = batch["targets"]

    assert seq.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Input sequence shape mismatch: {seq.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch: {targets.shape}"

    print(f"      Train batches: {len(train_loader)}")
    print(f"      Val batches:   {len(val_loader)}")
    print(f"      Test batches:  {len(test_loader)}")

    # -------------------------------------------------------------------------
    # 4. Model Training & Evaluation
    # -------------------------------------------------------------------------
    print("\n[4/5] Training Model...")

    model = ZoneoutWideResBiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Run the training engine
    train_model(model, train_loader, val_loader, optimizer, device, scheduler)

    # Verify model checkpoint exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model checkpoint was not saved.")
    print("      Training loop completed and model saved.")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating Submission...")

    generate_submission(model, test_loader, device)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"      Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    # Check row count: num_test_samples * seq_len
    # In debug mode, we have Config.DEBUG_SUBSET_SIZE samples in test set
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check that unscored columns are 0.0
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("      Submission verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
