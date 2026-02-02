import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import spearmanr

# Ensure the current directory is in the path for imports
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, compute_spearmanr_score
from library.dataset import load_data, QuestDataset, Collate
from library.model import QuestModel
from library.train import get_optimizer_params, train_fn, valid_fn, inference_fn


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # 3. Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    collate_fn = Collate(tokenizer)

    train_dataset = QuestDataset(train_df, tokenizer, is_test=False)
    val_dataset = QuestDataset(val_df, tokenizer, is_test=False)
    test_dataset = QuestDataset(test_df, tokenizer, is_test=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 4. Model Initialization
    model = QuestModel()
    model.to(device)

    # 5. Optimizer & Scheduler
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.lr_backbone,
        decoder_lr=Config.lr_head,
        weight_decay=Config.weight_decay,
    )

    optimizer = optim.AdamW(optimizer_parameters, eps=Config.eps, betas=Config.betas)

    num_train_steps = int(len(train_df) / Config.train_batch_size * Config.epochs)
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_score = -1.0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.output_model_path), exist_ok=True)

    for epoch in range(Config.epochs):
        _ = train_fn(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        _, val_preds, val_targets = valid_fn(model, val_loader, criterion, device)

        val_score = compute_spearmanr_score(val_preds, val_targets)

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.output_model_path)

    # 7. Final Validation & Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(Config.output_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Re-run validation inference to get best predictions
    _, val_preds, val_targets = valid_fn(model, val_loader, criterion, device)
    final_metric = compute_spearmanr_score(val_preds, val_targets)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Construct analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["error"] = mae_per_sample

    # Feature extraction for correlation
    analysis_df["q_len"] = (
        analysis_df["question_title"].str.len() + analysis_df["question_body"].str.len()
    )
    analysis_df["a_len"] = analysis_df["answer"].str.len()

    # Calculate correlations
    corr_q_len, _ = spearmanr(analysis_df["error"], analysis_df["q_len"])
    corr_a_len, _ = spearmanr(analysis_df["error"], analysis_df["a_len"])

    print(f"Failure Analysis - Correlation with Error:")
    print(f"  Question Length: {corr_q_len}")
    print(f"  Answer Length: {corr_a_len}")

    # 8. Submission
    threshold = 0.40802662717842303
    if final_metric > threshold:
        test_preds = inference_fn(model, test_loader, device)

        # Create submission DataFrame
        submission = pd.DataFrame(test_preds, columns=Config.target_cols)
        submission.insert(0, "qa_id", test_df["qa_id"])

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

        # Save submission
        submission.to_csv(Config.submission_path, index=False)


if __name__ == "__main__":
    main()
