import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    compute_qwk,
    optimize_thresholds,
    apply_thresholds,
)
from library.dataset import get_dataloaders
from library.models import EssayDebertaModel
from library.lexical_model import train_lexical_model
from library.engine import run_training


def main():
    print("Starting demonstration script...")

    # --- 1. Configuration Overrides for Speed and Demo ---
    # We override the Config class attributes directly.
    # Since the other modules import the 'Config' class, these changes will propagate.

    print("Overriding Config for fast demonstration...")
    Config.debug = True
    Config.debug_subset_size = 50  # Small subset for quick execution
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.num_workers = 0  # Disable multiprocessing for simple demo

    # Use a smaller model to ensure the demo runs quickly without downloading a massive file
    # DeBERTa-v3-xsmall is architecturally compatible with the EssayDebertaModel class
    Config.model_name = "microsoft/deberta-v3-xsmall"
    Config.output_dir = "./working/demo_run"
    Config.model_dir = os.path.join(Config.output_dir, "models")

    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.model_dir, exist_ok=True)

    seed_everything(Config.seed)

    # --- 2. Demonstrate Lexical Model (TF-IDF + Ridge) ---
    print("\n[Demo] Running Lexical Model Pipeline...")
    # This function handles data loading, vectorization, training, and evaluation
    lexical_results = train_lexical_model(load_cached_data=False, debug=True)

    # Verify results
    if "val_score" not in lexical_results or "test_preds" not in lexical_results:
        raise AssertionError("Lexical model did not return expected keys.")

    print(f"Lexical Model Validation QWK: {lexical_results['val_score']:.4f}")
    print("Lexical model demonstration successful.")

    # --- 3. Demonstrate Deep Learning Data Loading ---
    print("\n[Demo] Loading DataLoaders for Deep Learning Model...")
    # get_dataloaders handles tokenization and batching
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verify DataLoader
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    required_keys = ["input_ids", "attention_mask", "labels"]
    for k in required_keys:
        if k not in batch:
            raise AssertionError(f"Batch missing key: {k}")

    print(
        f"Batch loaded. Input shape: {batch['input_ids'].shape}, Labels shape: {batch['labels'].shape}"
    )
    print("Data loading demonstration successful.")

    # --- 4. Demonstrate Model Initialization & Forward Pass ---
    print("\n[Demo] Initializing EssayDebertaModel...")
    device = Config.device
    model = EssayDebertaModel(pretrained=True)
    model.to(device)

    print(f"Model moved to {device}.")

    # Run a single forward pass to verify architecture
    print("Running forward pass on a single batch...")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Output should be [batch_size, 1]
    expected_shape = (batch["input_ids"].size(0), 1)
    if outputs.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"
        )

    print(f"Forward pass successful. Output shape: {outputs.shape}")

    # --- 5. Demonstrate Training Loop (Engine) ---
    print("\n[Demo] Running Training Loop (1 Epoch)...")

    # Setup Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Total steps calculation
    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(Config.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    save_path = os.path.join(Config.model_dir, "best_model_demo.bin")

    # Run training
    best_weights, best_score = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=Config.epochs,
        patience=1,
        save_path=save_path,
    )

    print(f"Training loop finished. Best Validation QWK: {best_score}")

    # --- 6. Demonstrate Utility Functions (Thresholding) ---
    print("\n[Demo] Demonstrating Threshold Optimization...")

    # Create synthetic data for demonstration
    # Scenario: Model predicts continuous values, we need to map to 1-6 integers
    np.random.seed(42)
    dummy_targets = np.random.randint(1, 7, size=100)
    # Create predictions that are somewhat correlated but continuous
    dummy_preds_continuous = dummy_targets + np.random.normal(0, 0.4, size=100)

    # Optimize thresholds
    print("Optimizing thresholds...")
    opt_thresholds = optimize_thresholds(dummy_targets, dummy_preds_continuous)
    print(f"Optimized Thresholds: {opt_thresholds}")

    # Apply thresholds
    dummy_preds_int = apply_thresholds(dummy_preds_continuous, opt_thresholds)

    # Compute Score
    score = compute_qwk(dummy_targets, dummy_preds_int)
    print(f"Computed QWK on dummy data: {score:.4f}")

    # Verify logic
    if not (1 <= dummy_preds_int.min() and dummy_preds_int.max() <= 6):
        raise AssertionError(
            "Threshold application resulted in values outside 1-6 range."
        )

    print("Utility demonstration successful.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
