import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# Suppress verbose warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, get_score
from library.dataset import get_data, PhraseDataset
from library.model import PhraseModel
from library.engine import train_fn, valid_fn
from library.awp import AWP


def run_demo():
    print(">>> Starting Phrase Similarity Task Demo")

    # ==========================================
    # 1. Configuration Override for Speed/Demo
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable debug mode to use a small subset of data (1000 rows)
    CFG.debug = True

    # Override model with a tiny BERT to ensure very fast execution and low memory footprint
    # The original config uses 'microsoft/deberta-v3-large' which is heavy.
    CFG.model_name = "prajjwal1/bert-tiny"

    # Reduce training parameters for speed
    CFG.epochs = 1
    CFG.batch_size = 8
    CFG.n_fold = 2
    CFG.print_freq = 10

    # Enable AWP from the first epoch to verify its logic immediately
    CFG.awp = True
    CFG.awp_start_epoch = 0

    # Ensure reproducibility
    seed_everything(CFG.seed)

    print(f"Debug Mode: {CFG.debug}")
    print(f"Model Architecture: {CFG.model_name}")
    print(f"Device: {CFG.device}")

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    print("\n[2] Loading and processing data...")

    # We set load_cached_data=False to force the execution of the raw data processing logic
    # (merging metadata with CPC descriptions) rather than loading pre-computed Parquet files.
    train_df, test_df = get_data(CFG, load_cached_data=False)

    # Validation assertions
    assert (
        len(train_df) == 1000
    ), f"Expected 1000 training samples in debug mode, got {len(train_df)}"
    assert (
        len(test_df) == 100
    ), f"Expected 100 test samples in debug mode, got {len(test_df)}"
    assert (
        "context_text" in train_df.columns
    ), "Context text mapping failed: column missing"
    assert (
        not train_df["context_text"].isnull().any()
    ), "Found null context descriptions after mapping"
    assert "fold" in train_df.columns, "Fold column missing in training data"

    print("Data loading verified successfully.")

    # ==========================================
    # 3. Dataset & Tokenizer
    # ==========================================
    print("\n[3] Preparing Dataset and Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # Instantiate Datasets
    # We use mode='train' for both to ensure we can verify label retrieval
    train_dataset = PhraseDataset(CFG, train_df, tokenizer, mode="train")

    # Verify Dataset Item Structure
    sample_item = train_dataset[0]
    required_keys = ["input_ids", "attention_mask", "label"]
    for key in required_keys:
        assert key in sample_item, f"Dataset item missing required key: {key}"

    # Verify Shapes and Types
    assert (
        sample_item["input_ids"].shape[0] == CFG.max_len
    ), f"Input IDs length mismatch. Expected {CFG.max_len}"
    assert isinstance(
        sample_item["label"].item(), float
    ), "Label should be a float value"

    print("Dataset and Tokenizer verified successfully.")

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("\n[4] Initializing Model...")

    model = PhraseModel(CFG, pretrained=True)
    model.to(CFG.device)

    # Verify Forward Pass
    # Create a dummy batch (batch_size=1)
    dummy_input = sample_item["input_ids"].unsqueeze(0).to(CFG.device)
    dummy_mask = sample_item["attention_mask"].unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        output = model(dummy_input, dummy_mask)

    # Expect output shape [1] (regression score)
    assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"

    print("Model initialized and forward pass verified.")

    # ==========================================
    # 5. AWP Logic Verification (Unit Test)
    # ==========================================
    print("\n[5] Verifying Adversarial Weight Perturbation (AWP)...")

    # Setup for AWP unit test
    model.train()
    optimizer_awp = AdamW(model.parameters(), lr=1e-3)
    awp = AWP(model, optimizer_awp, adv_lr=0.1, adv_eps=0.5, start_epoch=0)

    # Identify a suitable parameter to track (must have gradients)
    target_param = None
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name and param.dim() > 1:
            target_param = param
            break

    if target_param is not None:
        # Step A: Compute Gradients via backward pass
        optimizer_awp.zero_grad()
        out = model(dummy_input, dummy_mask)
        loss = nn.MSELoss()(out, torch.tensor([1.0], device=CFG.device))
        loss.backward()

        # Step B: Save original weight
        original_weight = target_param.data.clone()

        # Step C: Attack (Perturb weights)
        awp.attack(epoch=0)
        perturbed_weight = target_param.data.clone()

        # Check if weights changed
        diff = torch.norm(original_weight - perturbed_weight).item()
        assert diff > 0, "AWP Attack failed: Weights did not change after attack"

        # Step D: Restore
        awp._restore()
        restored_weight = target_param.data

        # Check if weights restored exactly
        assert torch.equal(
            original_weight, restored_weight
        ), "AWP Restore failed: Weights not restored to original"

        print(f"AWP logic verified. Weight perturbation magnitude: {diff:.6f}")
    else:
        raise AssertionError("Could not find a suitable parameter to test AWP.")

    # Clear gradients after test
    optimizer_awp.zero_grad()

    # ==========================================
    # 6. Training & Validation Loop (Engine)
    # ==========================================
    print("\n[6] Running Training and Validation Loops...")

    # Create DataLoader
    # drop_last=True ensures consistent batch sizes for assertions
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Use main thread for demo stability
    )

    # Setup Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=CFG.encoder_lr, weight_decay=CFG.weight_decay
    )
    num_train_steps = len(train_loader) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    criterion = nn.MSELoss()

    # Run Training Function (1 Epoch)
    print("Executing train_fn (1 epoch)...")
    train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=CFG.device,
        cfg=CFG,
    )

    # Run Validation Function
    # We reuse train_loader as the validation set for this speed demo
    print("Executing valid_fn...")
    val_loss, preds, score = valid_fn(
        valid_loader=train_loader,
        model=model,
        criterion=criterion,
        device=CFG.device,
        cfg=CFG,
    )

    # Verify Results
    assert not np.isnan(val_loss), "Validation loss returned NaN"

    # Check predictions count
    # Since drop_last=True, count is (total // batch_size) * batch_size
    expected_count = (len(train_dataset) // CFG.batch_size) * CFG.batch_size
    assert (
        len(preds) == expected_count
    ), f"Prediction count mismatch. Expected {expected_count}, got {len(preds)}"

    # Check Score (Pearson Correlation)
    assert isinstance(score, float), "Score should be a float"
    assert (
        -1.0 <= score <= 1.0
    ), f"Pearson correlation score {score} is out of valid range [-1, 1]"

    print(f"Engine execution verified. Final Validation Score: {score:.4f}")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
