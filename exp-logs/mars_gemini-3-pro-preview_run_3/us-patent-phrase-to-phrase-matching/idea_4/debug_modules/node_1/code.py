import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

# Import components from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_data, PearsonDataset, DataCollator
from library.model import CustomModel
from library.awp import AWP
from library.engine import get_optimizer_params, train_fn, valid_fn


def run_demo():
    print(">>> Starting Library Usage Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config attributes to optimize for speed and demonstration purposes
    Config.debug = True  # Use a tiny subset of data (100 rows)
    Config.epochs = 1  # Run only 1 epoch
    Config.train_batch_size = 4  # Small batch size
    Config.valid_batch_size = 8
    Config.awp_start_epoch = 0  # Enable AWP immediately for this demo
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo
    Config.output_dir = "./working/demo_output"

    # Ensure reproducibility
    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")
    print("Configuration configured for fast demo execution (Debug Mode).")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n>>> Loading and preparing data...")
    # load_cached_data=False forces loading from CSVs to verify that logic
    df_train, df_val, df_test = prepare_data(load_cached_data=False)

    # Verify that Debug mode correctly reduced the dataset size
    print(f"Train set size: {len(df_train)}")
    print(f"Val set size:   {len(df_val)}")
    if len(df_train) > 100:
        raise AssertionError("Debug mode failed: Train set is too large.")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    train_dataset = PearsonDataset(df_train, tokenizer)
    val_dataset = PearsonDataset(df_val, tokenizer)

    # Create Data Collator
    collator = DataCollator(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=Config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=Config.num_workers,
    )

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    required_keys = ["input_ids", "attention_mask", "score", "label_idx"]
    for key in required_keys:
        if key not in sample_batch:
            raise AssertionError(f"Batch missing key: {key}")

    print("Data Pipeline verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model...")
    model = CustomModel()
    model.to(device)

    # Verify Forward Pass dimensions
    model.eval()
    with torch.no_grad():
        input_ids = sample_batch["input_ids"].to(device)
        mask = sample_batch["attention_mask"].to(device)
        outputs = model(input_ids, mask)

    # Check output shapes
    # Score: (Batch, 1), Logits: (Batch, Num_Classes)
    batch_size = input_ids.size(0)
    if outputs["score"].shape != (batch_size, 1):
        raise AssertionError(
            f"Expected score shape ({batch_size}, 1), got {outputs['score'].shape}"
        )
    if outputs["logits"].shape != (batch_size, Config.num_classes):
        raise AssertionError(
            f"Expected logits shape ({batch_size}, {Config.num_classes}), got {outputs['logits'].shape}"
        )

    print("Model architecture and forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Optimizer, Scheduler & AWP Setup
    # -------------------------------------------------------------------------
    print("\n>>> Setting up Optimizer and AWP...")
    # Use the LLRD (Layer-wise Learning Rate Decay) utility
    optimizer_params = get_optimizer_params(model, encoder_lr=1e-5, decoder_lr=1e-4)
    optimizer = torch.optim.AdamW(optimizer_params)

    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Initialize Adversarial Weight Perturbation
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-4)

    # -------------------------------------------------------------------------
    # 5. Manual AWP Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying AWP mechanism...")
    model.train()
    # We need populated gradients to perform an AWP attack
    optimizer.zero_grad()
    outputs = model(input_ids, mask)
    dummy_loss = outputs["score"].mean()
    dummy_loss.backward()

    # Select a parameter to monitor (e.g., from the regressor head)
    param_monitor = list(model.fc_regressor.parameters())[0]
    weight_before = param_monitor.data.clone()

    # Perform Attack (Perturb weights)
    awp.attack()
    weight_after = param_monitor.data.clone()

    # Verify weights changed
    diff = torch.norm(weight_after - weight_before).item()
    print(f"AWP Perturbation Norm: {diff:.6f}")
    if diff == 0.0:
        print("Warning: AWP did not change weights (gradients might be zero).")

    # Restore weights
    awp._restore()
    weight_restored = param_monitor.data

    # Verify restoration
    if not torch.allclose(weight_before, weight_restored):
        raise AssertionError("AWP failed to restore original weights.")

    print("AWP logic verified (Attack & Restore).")
    optimizer.zero_grad()  # Clear gradients before actual training

    # -------------------------------------------------------------------------
    # 6. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n>>> Executing Training Loop (1 Epoch)...")
    train_loss = train_fn(
        train_loader,
        model,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
    )
    print(f"Training completed. Average Loss: {train_loss:.4f}")

    if np.isnan(train_loss):
        raise AssertionError("Training loss returned NaN.")

    # -------------------------------------------------------------------------
    # 7. Validation Loop Execution
    # -------------------------------------------------------------------------
    print("\n>>> Executing Validation Loop...")
    val_loss, val_pearson = valid_fn(val_loader, model, device)
    print(f"Validation completed. Loss: {val_loss:.4f}, Pearson: {val_pearson:.4f}")

    if not (-1.0 <= val_pearson <= 1.0):
        raise AssertionError(
            f"Pearson score {val_pearson} is out of valid range [-1, 1]."
        )

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
