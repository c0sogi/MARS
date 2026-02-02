import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from scipy.stats import spearmanr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_optimizer_params, compute_metric
from library.dataset import load_data, get_target_columns, StackExchangeDataset
from library.model import ContextualizedDualEncoder
from library.engine import run_training, eval_fn


def failure_analysis(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()
    preds = []
    targets = []

    # Get predictions
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            ids_q = data["input_ids_q"].to(device, dtype=torch.long)
            mask_q = data["attention_mask_q"].to(device, dtype=torch.long)
            ids_a = data["input_ids_a"].to(device, dtype=torch.long)
            mask_a = data["attention_mask_a"].to(device, dtype=torch.long)
            pool_mask_a = data["pooling_mask_a"].to(device, dtype=torch.float)
            target_batch = data["labels"].to(device, dtype=torch.float)

            outputs = model(ids_q, mask_q, ids_a, mask_a, pool_mask_a)
            outputs = torch.sigmoid(outputs)

            preds.append(outputs.cpu().numpy())
            targets.append(target_batch.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Calculate Mean Absolute Error per sample (averaged across 30 targets)
    # Shape: (N_samples,)
    sample_mae = np.mean(np.abs(preds - targets), axis=1)

    # Extract meta-features from dataframe
    # We need to ensure the order matches the loader.
    # Since we didn't shuffle the val_loader (usually), it should match val_df.
    # However, to be safe, let's assume val_loader is sequential as per standard eval.

    val_df["title_len"] = val_df["question_title"].fillna("").astype(str).apply(len)
    val_df["body_len"] = val_df["question_body"].fillna("").astype(str).apply(len)
    val_df["answer_len"] = val_df["answer"].fillna("").astype(str).apply(len)

    # Calculate correlations
    features = ["title_len", "body_len", "answer_len"]
    print("Correlation between Model Error (MAE) and Input Features:")
    for feat in features:
        if feat in val_df.columns:
            corr, _ = spearmanr(sample_mae, val_df[feat].values[: len(sample_mae)])
            print(f"  {feat}: {corr:.4f}")


def generate_submission(model, test_loader, test_df, device, target_cols):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\nGenerating submission...")
    model.eval()
    preds = []

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            ids_q = data["input_ids_q"].to(device, dtype=torch.long)
            mask_q = data["attention_mask_q"].to(device, dtype=torch.long)
            ids_a = data["input_ids_a"].to(device, dtype=torch.long)
            mask_a = data["attention_mask_a"].to(device, dtype=torch.long)
            pool_mask_a = data["pooling_mask_a"].to(device, dtype=torch.float)

            outputs = model(ids_q, mask_q, ids_a, mask_a, pool_mask_a)
            outputs = torch.sigmoid(outputs)

            preds.append(outputs.cpu().numpy())

    preds = np.concatenate(preds)

    # Create submission DataFrame
    sub_df = pd.DataFrame(preds, columns=target_cols)
    sub_df.insert(0, "qa_id", test_df["qa_id"].values)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading data...")
    train_df = load_data("train", load_cached_data=True)
    val_df = load_data("val", load_cached_data=True)
    test_df = load_data("test", load_cached_data=True)

    target_cols = get_target_columns()
    print(
        f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    # 3. Prepare Datasets and Loaders
    print("Initializing Tokenizer and Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    train_dataset = StackExchangeDataset(
        train_df, tokenizer, target_cols, Config.MAX_LEN, is_test=False
    )
    val_dataset = StackExchangeDataset(
        val_df, tokenizer, target_cols, Config.MAX_LEN, is_test=False
    )
    test_dataset = StackExchangeDataset(
        test_df, tokenizer, target_cols, Config.MAX_LEN, is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    print("Initializing Model...")
    model = ContextualizedDualEncoder()
    model.to(device)

    # 5. Optimizer and Scheduler
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.LEARNING_RATE,
        head_lr=Config.HEAD_LR,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )

    optimizer = AdamW(optimizer_parameters)

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 6. Training
    print("Starting Training...")
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
    )

    # 7. Final Evaluation
    print("Loading Best Model for Evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    val_loss, val_score = eval_fn(val_loader, model, device)
    print(f"Final Validation Metric: {val_score}")

    # 8. Failure Analysis
    failure_analysis(model, val_loader, val_df, device)

    # 9. Submission
    THRESHOLD = 0.40802662717842303
    if val_score > THRESHOLD:
        print(
            f"Validation score ({val_score}) > Threshold ({THRESHOLD}). Generating submission."
        )
        generate_submission(model, test_loader, test_df, device, target_cols)
    else:
        print(
            f"Validation score ({val_score}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
