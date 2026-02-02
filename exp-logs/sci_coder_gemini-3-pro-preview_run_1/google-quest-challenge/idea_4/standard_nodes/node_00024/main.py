import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import load_data, QuestDataset, CollateFactory, get_tokenizer
from library.model import QuestModel
from library.engine import train_fn, eval_fn


def main():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    Config.setup()
    seed_everything(Config.seed)
    device = Config.device

    # Ensure we use the full metadata splits provided (approx 4.4k samples)
    # This is small enough for a fast baseline on A100.
    Config.debug = False

    print(f"Device: {device}")
    print("Initializing pipeline...")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # Load dataframes from metadata/cache
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=Config.debug)

    # Initialize Tokenizer and Collator
    tokenizer = get_tokenizer()
    collate_fn = CollateFactory(tokenizer)

    # Create Datasets
    train_dataset = QuestDataset(train_df, is_test=False)
    val_dataset = QuestDataset(val_df, is_test=False)

    # Create Dataloaders
    # drop_last=True ensures batch sizes are consistent for gradient accumulation
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train steps per epoch: {len(train_loader)}")

    # ==========================================
    # 3. Model & Optimizer Setup
    # ==========================================
    model = QuestModel()
    model.to(device)

    # Configure Optimizer with Differential Learning Rates
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        # Backbone parameters (lower LR)
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" in n and not any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_backbone,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" in n and any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_backbone,
            "weight_decay": 0.0,
        },
        # Head/Pooler parameters (higher LR)
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" not in n and not any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_head,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" not in n and any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_head,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters)

    # Configure Scheduler
    num_update_steps_per_epoch = len(train_loader) // Config.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_score = -1.0

    print("\nStarting training...")
    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)

        # Evaluate
        val_loss, val_score, _ = eval_fn(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{Config.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Spearman: {val_score:.5f}"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.5f} -> {val_score:.5f}). Saving Model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)

    # ==========================================
    # 5. Validation & Failure Analysis
    # ==========================================
    print("\nLoading best model for analysis...")
    if os.path.exists(Config.model_save_path):
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    else:
        print("Warning: No model file found. Using current weights.")

    model.to(device)
    model.eval()

    # Run Inference on Validation Set
    _, final_score, val_preds = eval_fn(model, val_loader, device)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error (MAE) per sample across all 30 targets
    val_targets = val_df[Config.target_cols].values
    mae_per_row = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Compute sequence lengths for correlation analysis
    # We use the raw text length as a proxy for token length
    val_df["q_len"] = val_df["question_text"].str.len()
    val_df["a_len"] = val_df["answer_text"].str.len()

    # Calculate Spearman correlation between Error and Lengths
    corr_q, _ = spearmanr(mae_per_row, val_df["q_len"])
    corr_a, _ = spearmanr(mae_per_row, val_df["a_len"])

    print(f"Correlation between Error (MAE) and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error (MAE) and Answer Length: {corr_a:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.40802662717842303

    if final_score > THRESHOLD:
        print(
            f"\nMetric ({final_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Prepare Test Loader
        test_dataset = QuestDataset(test_df, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Inference
        test_preds = []
        with torch.no_grad():
            for data in test_loader:
                q_ids = data["q_input_ids"].to(device)
                q_mask = data["q_attention_mask"].to(device)
                a_ids = data["a_input_ids"].to(device)
                a_mask = data["a_attention_mask"].to(device)

                logits = model(q_ids, q_mask, a_ids, a_mask)
                batch_preds = torch.sigmoid(logits)
                test_preds.append(batch_preds.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Create Submission DataFrame
        submission = pd.DataFrame(test_preds, columns=Config.target_cols)
        submission.insert(0, "qa_id", test_df["qa_id"].values)

        # Save
        os.makedirs(Config.submission_dir, exist_ok=True)
        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nMetric ({final_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
