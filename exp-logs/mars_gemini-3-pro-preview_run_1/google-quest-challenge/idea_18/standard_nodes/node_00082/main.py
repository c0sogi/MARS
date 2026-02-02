import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr

from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders
from library.model import SymmetricDualEncoder
from library.engine import train_fn, eval_fn


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, target_cols = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Setup
    print("Initializing model...")
    model = SymmetricDualEncoder()
    model.to(device)

    # 4. Optimization Configuration
    # Separate parameters for differential learning rates
    backbone_params = list(model.q_backbone.parameters()) + list(
        model.a_backbone.parameters()
    )
    head_params = (
        list(model.head_proj_1.parameters())
        + list(model.head_proj_2.parameters())
        + list(model.layer_norm.parameters())
    )

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": Config.LR_BACKBONE},
        {"params": head_params, "lr": Config.LR_HEAD},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=Config.WEIGHT_DECAY)

    # Phantom Scheduling: Calculate steps for 7 epochs, but we will stop at 3
    num_update_steps_per_epoch = len(train_loader) // Config.ACCUMULATION_STEPS
    max_train_steps = num_update_steps_per_epoch * Config.EPOCHS_SCHEDULE

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # We handle warmup via freezing in Epoch 1
        num_training_steps=max_train_steps,
    )

    # 5. Training Loop
    print(
        f"Starting training for {Config.EPOCHS_ACTUAL} epochs (Phantom Schedule: {Config.EPOCHS_SCHEDULE} epochs)..."
    )

    for epoch in range(Config.EPOCHS_ACTUAL):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS_ACTUAL}")

        # Epoch 1: Warmup - Freeze Backbones
        if epoch == 0:
            print("Warmup Phase: Freezing backbones, training head only.")
            for param in backbone_params:
                param.requires_grad = False
        else:
            # Epoch 2+: Fine-tuning - Unfreeze Backbones
            if epoch == 1:
                print("Fine-tuning Phase: Unfreezing backbones.")
                for param in backbone_params:
                    param.requires_grad = True

        # Train
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)

        # Validate
        val_score, _ = eval_fn(model, val_loader, device)
        print(f"Epoch {epoch + 1} Validation Score: {val_score:.4f}")

    # 6. Final Evaluation
    print("\nComputing Final Validation Metric...")
    final_val_score, val_preds = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Retrieve targets and metadata from validation dataset
    val_df = val_loader.dataset.df
    val_targets = val_loader.dataset.labels

    # Calculate Mean Absolute Error per sample (averaged across 30 targets)
    # val_preds is (N, 30), val_targets is (N, 30)
    sample_mae = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Calculate text lengths
    q_body_lens = val_df["question_body"].fillna("").str.len().values
    ans_lens = val_df["answer"].fillna("").str.len().values

    # Compute correlations
    corr_q, _ = spearmanr(sample_mae, q_body_lens)
    corr_a, _ = spearmanr(sample_mae, ans_lens)

    print(f"Correlation between Error and Question Body Length: {corr_q:.4f}")
    print(f"Correlation between Error and Answer Length: {corr_a:.4f}")

    # 8. Submission
    THRESHOLD = 0.4113257391193607
    if final_val_score > THRESHOLD:
        print(
            f"\nScore ({final_val_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        _, test_preds = eval_fn(model, test_loader, device)

        # Create Submission DataFrame
        submission = pd.DataFrame(test_preds, columns=target_cols)
        submission.insert(0, "qa_id", test_loader.dataset.qa_ids)

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nScore ({final_val_score}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
