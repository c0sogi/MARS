import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from scipy.stats import spearmanr

# Import from provided library files
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_loaders, ALL_TARGETS
from library.model import ContextualDualEncoder
from library.engine import Engine


def main():
    # 1. Setup and Configuration
    SEED = 42
    BATCH_SIZE = 16  # Cite solution_lesson_node_00012
    EPOCHS = 5  # Cite solution_lesson_node_00011
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-3  # Cite solution_lesson_node_00015
    WEIGHT_DECAY = 0.01
    THRESHOLD = 0.4033583081786146

    seed_everything(SEED)

    # Directories
    WORK_DIR = "./working/idea_3"
    SUB_DIR = "./submission"
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    checkpoint_path = os.path.join(WORK_DIR, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Using load_cached_data=True as requested to use preprocessed parquet files
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    model = ContextualDualEncoder()
    model.to(device)

    # Optimizer Setup (Differential Learning Rates)
    # Group 1: Backbone (Low LR)
    # Group 2: Head (High LR)
    backbone_params = list(model.backbone.named_parameters())
    head_params = (
        list(model.head.named_parameters())
        + list(model.pooler.named_parameters())
        + list(model.fusion_norm.named_parameters())
    )

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in backbone_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
            "lr": LR_BACKBONE,
        },
        {
            "params": [
                p for n, p in backbone_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": LR_BACKBONE,
        },
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
            "lr": LR_HEAD,
        },
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": LR_HEAD,
        },
    ]

    optimizer = AdamW(optimizer_parameters)

    # Scheduler Setup
    num_train_steps = int(len(train_loader) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    engine = Engine(model, device, optimizer, scheduler)

    # 4. Training Loop
    best_score = -1.0

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss = engine.train_one_epoch(train_loader, epoch)
        val_score = engine.validate(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Spearman: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Training complete. Best Validation Score: {best_score}")

    # 5. Validation & Failure Analysis
    print("\nRunning Failure Analysis on Best Model...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # We need predictions and targets for the validation set
    # Engine.predict returns qa_ids and preds
    val_qa_ids, val_preds = engine.predict(val_loader)

    # Retrieve targets from dataset
    # Note: val_loader.dataset is QuestDataset
    val_ds = val_loader.dataset
    val_targets = np.concatenate([val_ds.q_labels, val_ds.a_labels], axis=1)

    # Calculate Final Metric explicitly to ensure precision
    final_metric = compute_spearman_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Input Lengths
    # Calculate MAE per sample (average across 30 targets)
    abs_error = np.abs(val_preds - val_targets)
    mean_abs_error = np.mean(abs_error, axis=1)  # Shape: (N_val,)

    # Extract lengths from dataframe
    val_df = val_ds.df
    q_lengths = val_df["question_input"].str.len().values
    a_lengths = val_df["answer_input"].str.len().values

    # Compute correlations
    # Handle potential constant arrays if dataset is tiny (unlikely here)
    try:
        corr_q, _ = spearmanr(mean_abs_error, q_lengths)
        corr_a, _ = spearmanr(mean_abs_error, a_lengths)
    except Exception:
        corr_q, corr_a = 0.0, 0.0

    print("\n--- Failure Analysis ---")
    print(f"Correlation between Error and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error and Answer Length: {corr_a:.4f}")

    # 6. Submission
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_qa_ids, test_preds = engine.predict(test_loader)

        sub_df = pd.DataFrame(test_preds, columns=ALL_TARGETS)
        sub_df.insert(0, "qa_id", test_qa_ids)

        submission_path = os.path.join(SUB_DIR, "submission.csv")
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
