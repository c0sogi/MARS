import sys
import os
import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data_processing import load_data, ChatbotDataset, CollateFn
from library.model_components import SiameseDeberta
from library.engine import train_fn, eval_fn, inference_fn

# --- Configuration Override for Fast Baseline ---
# We modify the Config class attributes directly to control the run
Config.EPOCHS = 1
Config.TRAIN_BATCH_SIZE = 4  # Fit in A100 memory
Config.TARGET_EFFECTIVE_BATCH_SIZE = (
    32  # Reduced for smaller dataset/faster convergence
)
Config.GRAD_ACCUM_STEPS = max(
    1, Config.TARGET_EFFECTIVE_BATCH_SIZE // Config.TRAIN_BATCH_SIZE
)
# We will manually slice the training data, so we keep DEBUG=False to get full Val/Test
Config.DEBUG = False


def perform_failure_analysis(val_df, preds, targets):
    """
    Analyzes the correlation between error magnitude and input features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Compute Error Magnitude (Cross Entropy per sample)
    # Clip predictions for numerical stability
    epsilon = 1e-15
    preds = np.clip(preds, epsilon, 1 - epsilon)

    # targets is (N, 3), preds is (N, 3)
    # Cross Entropy = - sum(target * log(pred))
    loss_per_sample = -np.sum(targets * np.log(preds), axis=1)

    # 2. Extract Features
    # We work on a copy to avoid SettingWithCopy warnings
    df_analysis = val_df.copy()

    df_analysis["prompt_len"] = df_analysis["prompt"].fillna("").str.len()
    df_analysis["response_a_len"] = df_analysis["response_a"].fillna("").str.len()
    df_analysis["response_b_len"] = df_analysis["response_b"].fillna("").str.len()
    df_analysis["len_diff"] = (
        df_analysis["response_a_len"] - df_analysis["response_b_len"]
    ).abs()

    features = ["prompt_len", "response_a_len", "response_b_len", "len_diff"]

    print("Correlation between Error Magnitude (Log Loss) and Input Features:")
    for feat in features:
        if feat in df_analysis.columns:
            # Compute Pearson correlation
            if df_analysis[feat].std() > 0:
                corr = np.corrcoef(df_analysis[feat].values, loss_per_sample)[0, 1]
                print(f"{feat}: {corr:.4f}")
            else:
                print(f"{feat}: NaN (No variance)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    # We load the full datasets first
    train_df, val_df, test_df = load_data(load_cached_data=Config.LOAD_CACHED_DATA)

    # --- Fast Baseline Optimization ---
    # Slice training data to 5000 samples to ensure execution < 1 hour
    # We keep Validation and Test full to meet requirements
    TRAIN_SAMPLE_LIMIT = 5000
    if len(train_df) > TRAIN_SAMPLE_LIMIT:
        print(
            f"Limiting training data to {TRAIN_SAMPLE_LIMIT} samples for fast baseline."
        )
        train_df = train_df.iloc[:TRAIN_SAMPLE_LIMIT].reset_index(drop=True)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Test Set Size: {len(test_df)}")

    # 3. Prepare DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    collate_fn = CollateFn(tokenizer)

    train_dataset = ChatbotDataset(
        train_df, tokenizer, max_length=Config.MAX_LENGTH, mode="train"
    )
    val_dataset = ChatbotDataset(
        val_df, tokenizer, max_length=Config.MAX_LENGTH, mode="val"
    )
    test_dataset = ChatbotDataset(
        test_df, tokenizer, max_length=Config.MAX_LENGTH, mode="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 4. Model & Optimization
    print("Initializing Siamese DeBERTa model...")
    model = SiameseDeberta()
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate steps
    num_training_steps = (len(train_loader) // Config.GRAD_ACCUM_STEPS) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    print("Starting Training...")
    model.train()

    # We manually run the loop here to control it tightly, or reuse train_fn
    # reusing train_fn is cleaner
    for epoch in range(Config.EPOCHS):
        avg_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("Training complete. Model saved.")

    # 6. Validation Assessment
    print("Running Validation...")
    # eval_fn returns the log loss on the full validation set
    val_log_loss, val_ce_loss = eval_fn(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_log_loss}")

    # 7. Failure Analysis
    # We need predictions on the validation set to analyze errors
    # We run a quick inference pass (without TTA) to match the metric calculation
    print("Generating validation predictions for failure analysis...")
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                logits = model(input_ids_a, mask_a, input_ids_b, mask_b, scalars)
                probs = torch.softmax(logits, dim=1)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    perform_failure_analysis(val_df, val_preds, val_targets)

    # 8. Submission
    SUBMISSION_THRESHOLD = 1.0005665522536111

    if val_log_loss < SUBMISSION_THRESHOLD:
        print(
            f"Validation Metric ({val_log_loss}) is better than threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Use inference_fn which applies Test-Time Augmentation (TTA)
        ids, test_preds = inference_fn(model, test_loader, device)

        submission_df = pd.DataFrame(
            {
                "id": ids,
                "winner_model_a": test_preds[:, 0],
                "winner_model_b": test_preds[:, 1],
                "winner_tie": test_preds[:, 2],
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation Metric ({val_log_loss}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
