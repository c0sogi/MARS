import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from scipy.stats import spearmanr
import warnings

# Import from library
from library.utils import set_seed, compute_spearman_metric
from library.trainer import Trainer

# Suppress warnings
warnings.filterwarnings("ignore")


def validate_model(model, loader, device):
    """
    Runs validation and returns score, predictions, and labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Single forward pass
            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    score = compute_spearman_metric(all_labels, all_preds)
    return score, all_preds, all_labels


def generate_submission_file(model, loader, device, target_cols, output_path):
    """
    Generates submission file for test set.
    """
    model.eval()
    all_preds = []
    all_qa_ids = []

    with torch.no_grad():
        for batch in loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            qa_ids = batch["qa_id"]

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_qa_ids.extend(qa_ids.numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    sub_df = pd.DataFrame(all_preds, columns=target_cols)
    sub_df.insert(0, "qa_id", all_qa_ids)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # ==========================================
    # 1. Setup and Initialization
    # ==========================================
    SEED = 42
    set_seed(SEED)

    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    THRESHOLD = 0.40826991743732177

    # Initialize Trainer to get components (DataLoaders, Model, Device)
    print("Initializing components...")
    trainer_ctx = Trainer(
        model_name="distilroberta-base",
        batch_size=16,
        max_length=512,
        seed=SEED,
        working_dir=WORKING_DIR,
        submission_dir=SUBMISSION_DIR,
    )

    model = trainer_ctx.model
    device = trainer_ctx.device
    train_loader = trainer_ctx.train_loader
    val_loader = trainer_ctx.val_loader
    test_loader = trainer_ctx.test_loader
    criterion = trainer_ctx.criterion
    target_cols = trainer_ctx.target_cols

    best_score = -float("inf")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # 2. Training Loop
    # ==========================================

    # --- Phase 1: Head Warmup (Epoch 1) ---
    print("\nPhase 1: Head Warmup")

    # Freeze Backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Optimizer for Head
    head_params = list(model.head.named_parameters()) + list(
        model.fusion_norm.named_parameters()
    )
    optimizer_head = AdamW(head_params, lr=1e-3, weight_decay=0.01)

    model.train()
    for step, batch in enumerate(train_loader):
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)
        a_ids = batch["a_input_ids"].to(device)
        a_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer_head.zero_grad()

        # R-Drop: Two forward passes
        logits1 = model(q_ids, q_mask, a_ids, a_mask)
        logits2 = model(q_ids, q_mask, a_ids, a_mask)
        loss = criterion(logits1, logits2, labels)

        loss.backward()
        optimizer_head.step()

    # Validation
    val_score, _, _ = validate_model(model, val_loader, device)
    print(f"Epoch 1 Validation Score: {val_score:.6f}")

    if val_score > best_score:
        best_score = val_score
        torch.save(model.state_dict(), best_model_path)

    # --- Phase 2: Fine-tuning (Epochs 2-8) ---
    print("\nPhase 2: Fine-tuning")

    # Unfreeze Backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Differential LR
    backbone_params = list(model.backbone.named_parameters())
    head_params = list(model.head.named_parameters()) + list(
        model.fusion_norm.named_parameters()
    )

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    optimizer_grouped_parameters = [
        # Backbone (lr=2e-5)
        {
            "params": [
                p for n, p in backbone_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
            "lr": 2e-5,
        },
        {
            "params": [
                p for n, p in backbone_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": 2e-5,
        },
        # Head (lr=1e-3)
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
            "lr": 1e-3,
        },
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": 1e-3,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    # Scheduler
    epochs = 8
    remaining_epochs = epochs - 1
    total_steps = len(train_loader) * remaining_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(2, epochs + 1):
        model.train()
        for batch in train_loader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits1 = model(q_ids, q_mask, a_ids, a_mask)
            logits2 = model(q_ids, q_mask, a_ids, a_mask)
            loss = criterion(logits1, logits2, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation
        val_score, _, _ = validate_model(model, val_loader, device)
        print(f"Epoch {epoch} Validation Score: {val_score:.6f}")

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # ==========================================
    # 3. Final Evaluation & Failure Analysis
    # ==========================================
    print(f"Final Validation Metric: {best_score}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get predictions on validation set
    val_score, val_preds, val_labels = validate_model(model, val_loader, device)

    # Failure Analysis
    # Metric: Mean Absolute Error per sample (averaged over 30 targets)
    mae_per_sample = np.mean(np.abs(val_preds - val_labels), axis=1)

    # Get Input Features (Lengths)
    val_dataset = val_loader.dataset

    # Calculate lengths (char length approximation)
    q_lengths = [
        len(str(t)) + len(str(b))
        for t, b in zip(val_dataset.titles, val_dataset.bodies)
    ]
    a_lengths = [len(str(a)) for a in val_dataset.answers]

    # Ensure alignment
    if len(q_lengths) == len(mae_per_sample):
        corr_q, _ = spearmanr(mae_per_sample, q_lengths)
        corr_a, _ = spearmanr(mae_per_sample, a_lengths)

        print("\nFailure Analysis:")
        print(f"Correlation (Error vs Question Length): {corr_q:.6f}")
        print(f"Correlation (Error vs Answer Length): {corr_a:.6f}")
    else:
        print("Warning: Mismatch in validation set size for failure analysis.")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    if best_score > THRESHOLD:
        print(
            f"\nMetric ({best_score:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission_file(
            model, test_loader, device, target_cols, SUBMISSION_PATH
        )
    else:
        print(
            f"\nMetric ({best_score:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
