import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from transformers import get_linear_schedule_with_warmup

# Import from provided library files
from library.utils import seed_everything, compute_spearmanr
from library.data import prepare_loaders
from library.model import DistilRobertaDualEncoder
from library.train import train_fn, eval_fn, predict_fn

# Configuration
# Cite solution_lesson_node_00012: Small Batch Size (16) enhances generalization
BATCH_SIZE = 16
ACCUMULATION_STEPS = 1
# Cite solution_lesson_node_00015: Differential Learning Rates
HEAD_LR = 1e-3
BACKBONE_LR = 2e-5
# Cite solution_lesson_node_00011: 5+ Epochs for convergence
EPOCHS = 6
WEIGHT_DECAY = 0.01
SEED = 42
THRESHOLD = 0.40826991743732177

WORKING_DIR = "./working/idea_16/"
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission/"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


def main():
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader, target_cols = prepare_loaders(
        load_cached_data=True, batch_size=BATCH_SIZE, seed=SEED
    )

    # 3. Model Initialization
    print("Initializing DistilRoBERTa Dual Encoder...")
    model = DistilRobertaDualEncoder(num_labels=len(target_cols))
    model.to(device)

    # Cite solution_lesson_node_00050: Freeze backbone for the first epoch
    print("Freezing backbone for initial epoch...")
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 4. Optimizer and Scheduler
    # Cite solution_lesson_node_00018: Segregate parameters for weight decay
    no_decay = ["bias", "LayerNorm.weight"]

    # We define groups for Head and Backbone separately to apply Differential LR
    # Cite solution_lesson_node_00015: High LR for Head, Low LR for Backbone

    # Head Parameters
    head_params_decay = [
        p
        for n, p in model.head.named_parameters()
        if not any(nd in n for nd in no_decay)
    ]
    head_params_no_decay = [
        p for n, p in model.head.named_parameters() if any(nd in n for nd in no_decay)
    ]

    # Backbone Parameters
    # Note: Even though requires_grad=False initially, we add them to optimizer.
    # PyTorch optimizer will skip them if grad is None, which is fine.
    # Cite solution_lesson_node_00019: Continuous schedule (no reset)
    backbone_params_decay = [
        p
        for n, p in model.backbone.named_parameters()
        if not any(nd in n for nd in no_decay)
    ]
    backbone_params_no_decay = [
        p
        for n, p in model.backbone.named_parameters()
        if any(nd in n for nd in no_decay)
    ]

    optimizer_grouped_parameters = [
        # Head Groups
        {"params": head_params_decay, "weight_decay": WEIGHT_DECAY, "lr": HEAD_LR},
        {"params": head_params_no_decay, "weight_decay": 0.0, "lr": HEAD_LR},
        # Backbone Groups
        {
            "params": backbone_params_decay,
            "weight_decay": WEIGHT_DECAY,
            "lr": BACKBONE_LR,
        },
        {"params": backbone_params_no_decay, "weight_decay": 0.0, "lr": BACKBONE_LR},
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)
    loss_fn = nn.BCEWithLogitsLoss()

    # Adjust num_training_steps
    num_update_steps_per_epoch = len(train_loader) // ACCUMULATION_STEPS
    num_training_steps = num_update_steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    best_score = -1.0
    print(f"Starting training for {EPOCHS} epochs with Batch Size {BATCH_SIZE}...")

    for epoch in range(EPOCHS):
        # Cite solution_lesson_node_00050: Unfreeze backbone after epoch 1
        if epoch == 1:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True

        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_fn,
            accumulation_steps=ACCUMULATION_STEPS,
        )
        val_loss, val_score = eval_fn(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Spearman: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"New best model saved with score: {best_score:.4f}")

    # 6. Final Validation Assessment
    print("\nPerforming Final Validation Assessment...")
    # Load best model
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Get predictions on validation set for analysis
    # We need to manually run inference to get raw predictions and labels
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            preds = torch.sigmoid(logits)
            val_preds.append(preds.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    # Compute Final Metric
    final_metric = compute_spearmanr(val_preds, val_labels)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(val_preds - val_labels), axis=1)

    # Get metadata features for correlation
    # Access the dataframe from the dataset
    val_df = val_loader.dataset.df

    # Calculate lengths
    q_lengths = val_df["question_text"].str.len().values
    a_lengths = val_df["answer"].str.len().values

    # Calculate correlations
    corr_q, _ = spearmanr(mae_per_sample, q_lengths)
    corr_a, _ = spearmanr(mae_per_sample, a_lengths)

    print(f"Correlation between Error and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error and Answer Length: {corr_a:.4f}")

    # 8. Conditional Submission
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds, test_ids = predict_fn(model, test_loader, device)

        submission_df = pd.DataFrame(test_preds, columns=target_cols)
        submission_df.insert(0, "qa_id", test_ids)

        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
