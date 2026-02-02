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
from library.model import LoRADebertaDualEncoder
from library.train import train_fn, eval_fn, predict_fn

# Configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-3  # Higher LR for LoRA
EPOCHS = 5  # Fast baseline
WEIGHT_DECAY = 0.01
SEED = 42
THRESHOLD = 0.40826991743732177

WORKING_DIR = "./working/idea_15/"
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
    print("Initializing LoRA DeBERTa Dual Encoder...")
    model = LoRADebertaDualEncoder(num_labels=len(target_cols))
    model.to(device)

    # 4. Optimizer and Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    best_score = -1.0
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device, loss_fn
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
