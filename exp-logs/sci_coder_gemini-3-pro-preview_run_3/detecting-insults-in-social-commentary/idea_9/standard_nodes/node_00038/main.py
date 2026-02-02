import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import ModelConfig
from library.utils import seed_everything
from library.data import load_processed_data, create_augmented_dataset, get_dataloaders
from library.model import InsultModel
from library.engine import train_one_epoch, valid_fn, inference_fn


def run():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override epochs for fast baseline execution while maintaining ensemble size
    ModelConfig.epochs = 2

    # Fix for OOM on Deberta-v3-large with AWP
    # Reduce batch size and increase accumulation to fit in ~16GB VRAM
    ModelConfig.train_batch_size = 8
    ModelConfig.accumulation_steps = 4

    # Ensure working directory exists
    os.makedirs(ModelConfig.working_dir, exist_ok=True)

    seed_everything(ModelConfig.seed)
    device = ModelConfig.device
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    train_df, val_df, test_df = load_processed_data(load_cached_data=True)

    # Idea Requirement: Explicitly concatenate Train and Validation sets
    full_train_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"Initial Training Set Size (Train+Val): {len(full_train_df)}")

    # ==========================================
    # 3. Stage 1: Teacher Ensemble Training
    # ==========================================
    print("\n=== Stage 1: Teacher Ensemble Training ===")

    stage1_test_preds = []

    # Loop over backbones and seeds
    for backbone in ModelConfig.backbones:
        for seed in ModelConfig.seeds:
            print(f"\nTraining Teacher | Backbone: {backbone} | Seed: {seed}")
            seed_everything(seed)

            # Get DataLoaders
            # We use full_train_df for training.
            # val_loader is just for monitoring, though metric will be biased.
            train_loader, val_loader, test_loader = get_dataloaders(
                full_train_df, val_df, test_df, backbone
            )

            # Initialize Model
            model = InsultModel(backbone).to(device)

            # Optimization
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=ModelConfig.learning_rate,
                weight_decay=ModelConfig.weight_decay,
            )

            num_training_steps = len(train_loader) * ModelConfig.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_training_steps * ModelConfig.warmup_ratio),
                num_training_steps=num_training_steps,
            )

            # Training Loop
            for epoch in range(ModelConfig.epochs):
                avg_loss = train_one_epoch(
                    model, optimizer, scheduler, train_loader, device, epoch
                )
                # We skip validation logging for speed in this baseline script
                # but we could call valid_fn here.
                print(f"  Epoch {epoch+1}/{ModelConfig.epochs} - Loss: {avg_loss:.4f}")

            # Inference on Test
            preds = inference_fn(model, test_loader, device)
            stage1_test_preds.append(preds)

            # Cleanup to save memory
            del model, optimizer, scheduler, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

    # ==========================================
    # 4. Pseudo-Labeling
    # ==========================================
    print("\n=== Generating Pseudo-Labels ===")
    # Average predictions from all teachers
    avg_stage1_preds = np.mean(stage1_test_preds, axis=0)

    # Create Augmented Dataset
    augmented_train_df = create_augmented_dataset(
        full_train_df, test_df, avg_stage1_preds
    )
    print(f"Augmented Training Set Size: {len(augmented_train_df)}")

    # ==========================================
    # 5. Stage 2: Student Ensemble (with AWP)
    # ==========================================
    print("\n=== Stage 2: Student Ensemble Training (AWP Enabled) ===")

    stage2_val_preds = []
    stage2_test_preds = []

    # AWP is enabled in config (use_awp=True) and starts at epoch 1 (awp_start_epoch=1)
    # Since we run 2 epochs (0, 1), AWP will be active in the second epoch.

    for backbone in ModelConfig.backbones:
        for seed in ModelConfig.seeds:
            print(f"\nTraining Student | Backbone: {backbone} | Seed: {seed}")
            seed_everything(seed)

            # Get DataLoaders with Augmented Data
            train_loader, val_loader, test_loader = get_dataloaders(
                augmented_train_df, val_df, test_df, backbone
            )

            model = InsultModel(backbone).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=ModelConfig.learning_rate,
                weight_decay=ModelConfig.weight_decay,
            )

            num_training_steps = len(train_loader) * ModelConfig.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(num_training_steps * ModelConfig.warmup_ratio),
                num_training_steps=num_training_steps,
            )

            for epoch in range(ModelConfig.epochs):
                avg_loss = train_one_epoch(
                    model, optimizer, scheduler, train_loader, device, epoch
                )
                print(f"  Epoch {epoch+1}/{ModelConfig.epochs} - Loss: {avg_loss:.4f}")

            # Inference on Validation (for metric) and Test (for submission)
            # Note: Validation is on the original hold-out set (val_df)
            _, val_auc = valid_fn(
                model, val_loader, device
            )  # Check individual model perf

            # We need raw probabilities for ensemble averaging
            # Re-run inference to get preds (valid_fn returns metrics)
            # Actually valid_fn computes AUC but doesn't return preds.
            # We need to modify valid_fn? No, we can't modify library.
            # We will use inference_fn on val_loader.

            val_probs = inference_fn(model, val_loader, device)
            test_probs = inference_fn(model, test_loader, device)

            stage2_val_preds.append(val_probs)
            stage2_test_preds.append(test_probs)

            del model, optimizer, scheduler, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

    # ==========================================
    # 6. Evaluation & Failure Analysis
    # ==========================================
    print("\n=== Evaluation ===")
    final_val_preds = np.mean(stage2_val_preds, axis=0)
    final_val_targets = val_df["Insult"].values

    final_auc = roc_auc_score(final_val_targets, final_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    val_df_analysis = val_df.copy()
    val_df_analysis["pred"] = final_val_preds
    val_df_analysis["error"] = np.abs(
        val_df_analysis["Insult"] - val_df_analysis["pred"]
    )

    # Calculate lengths
    val_df_analysis["char_len"] = val_df_analysis["Comment"].apply(len)
    val_df_analysis["word_len"] = val_df_analysis["Comment"].apply(
        lambda x: len(str(x).split())
    )

    corr_char = val_df_analysis["error"].corr(val_df_analysis["char_len"])
    corr_word = val_df_analysis["error"].corr(val_df_analysis["word_len"])

    print(f"Correlation between Error and Char Length: {corr_char:.4f}")
    print(f"Correlation between Error and Word Length: {corr_word:.4f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    threshold = 0.9660591133004925
    if final_auc > threshold:
        print(
            f"\nValidation metric {final_auc} > {threshold}. Generating submission..."
        )

        final_test_preds = np.mean(stage2_test_preds, axis=0)

        # Load sample submission to preserve format
        sample_sub_path = os.path.join("./input", "sample_submission_null.csv")
        if os.path.exists(sample_sub_path):
            sub_df = pd.read_csv(sample_sub_path)
            # Ensure alignment (assuming test_df and sample_sub are aligned by index as per standard)
            # The test.csv and sample_submission usually match row-for-row.
            sub_df["Insult"] = final_test_preds

            submission_path = os.path.join(ModelConfig.submission_dir, "submission.csv")
            sub_df.to_csv(submission_path, index=False)
            print(f"Submission saved to {submission_path}")
        else:
            print("Error: sample_submission_null.csv not found.")
    else:
        print(f"\nValidation metric {final_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    run()
