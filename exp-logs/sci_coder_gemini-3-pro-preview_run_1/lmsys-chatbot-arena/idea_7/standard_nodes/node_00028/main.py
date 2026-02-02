import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import SiameseDebertaGated
from library.engine import train_model, predict
from library.features import generate_all_features


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print("Generating features...")
    generate_all_features(load_cached_data=True)

    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_features=True)

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing model...")
    model = SiameseDebertaGated()
    model.to(device)

    # ---------------------------------------------------------
    # 4. Optimization Setup
    # ---------------------------------------------------------
    # Differential Learning Rates
    # Backbone parameters get a lower LR, Head parameters get a higher LR
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    optimizer_grouped_parameters = [
        {
            "params": backbone_params,
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": head_params,
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Scheduler: Linear with Warmup
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUM_STEPS
    max_train_steps = num_update_steps_per_epoch * Config.NUM_EPOCHS
    # Warmup for 10% of steps
    num_warmup_steps = int(0.1 * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    # train_model handles the loop, validation, early stopping, and loading best model
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    )

    # ---------------------------------------------------------
    # 6. Final Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\nRunning final validation evaluation...")
    model.eval()
    val_probs = []
    val_targets = []
    val_structural_features = []

    with torch.no_grad():
        for data in val_loader:
            # Move data to device
            ids_a = data["input_ids_a"].to(device)
            mask_a = data["attention_mask_a"].to(device)
            ids_b = data["input_ids_b"].to(device)
            mask_b = data["attention_mask_b"].to(device)
            struct_feats = data["structural_features"].to(device)
            targets = data["labels"].to(device)

            # Inference
            outputs = model(ids_a, mask_a, ids_b, mask_b, struct_feats)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Store results
            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_structural_features.append(struct_feats.cpu().numpy())

    # Concatenate batches
    val_probs = np.concatenate(val_probs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_structural_features = np.concatenate(val_structural_features, axis=0)

    # Calculate Log Loss
    # eps='auto' is not a valid string for sklearn log_loss eps parameter in older versions,
    # but the task metric description says 'eps=auto'.
    # Standard sklearn log_loss default eps is 1e-15.
    # We will use the default behavior which is robust.
    metric = log_loss(val_targets, val_probs)
    print(f"Final Validation Metric: {metric}")

    # ---------------------------------------------------------
    # 7. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude per sample (Cross Entropy)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    # Cross Entropy = - sum(y_true * log(y_pred))
    sample_errors = -np.sum(val_targets * np.log(val_probs_clipped), axis=1)

    # Feature names corresponding to StructuralFeatureGenerator
    feature_names = [
        "char_len_diff",
        "char_len_ratio",
        "word_len_diff",
        "word_len_ratio",
        "newline_diff",
        "newline_ratio",
    ]

    print("Correlation between Error Magnitude and Input Features:")
    for i, name in enumerate(feature_names):
        feature_values = val_structural_features[:, i]
        # Calculate Pearson correlation
        correlation = np.corrcoef(sample_errors, feature_values)[0, 1]
        print(f"Feature '{name}': {correlation:.4f}")

    # ---------------------------------------------------------
    # 8. Submission
    # ---------------------------------------------------------
    THRESHOLD = 1.0026075514615997

    if metric < THRESHOLD:
        print(f"\nValidation metric {metric} < {THRESHOLD}. Generating submission...")
        predict(model, test_loader, device)
    else:
        print(f"\nValidation metric {metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
