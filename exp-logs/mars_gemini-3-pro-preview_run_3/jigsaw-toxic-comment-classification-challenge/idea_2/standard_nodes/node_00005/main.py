import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library functions
from library.utils import seed_everything, calculate_roc_auc
from library.data_processing import (
    load_data_from_metadata,
    get_tfidf_features,
    make_dataloaders,
)
from library.model_definitions import NBSVM, ToxicRoBERTa
from library.training_engine import train_roberta_epoch, infer_roberta

# Configuration
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
THRESHOLD = 0.9802773432234382
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Hyperparameters
ROBERTA_MODEL = "roberta-base"
BATCH_SIZE = 32
MAX_LEN = 128
EPOCHS = 1  # Limited to 1 epoch for fast baseline execution
LR = 2e-5
ENSEMBLE_ALPHA = 0.6  # Weight for RoBERTa


def main():
    # 1. Setup
    seed_everything(SEED)

    # 2. Load Data
    print("Loading data...")
    train_df, val_df, test_df = load_data_from_metadata()

    # Extract targets
    y_train = train_df[LABEL_COLS].values
    y_val = val_df[LABEL_COLS].values

    # 3. NBSVM Pipeline
    print("--- NBSVM Pipeline ---")
    # Generate/Load TF-IDF features
    train_feats, val_feats, test_feats = get_tfidf_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Train NBSVM
    print("Training NBSVM...")
    nbsvm = NBSVM(C=1.0, dual=True, n_jobs=-1, random_state=SEED)
    nbsvm.fit(train_feats, y_train)

    # Inference NBSVM
    print("NBSVM Inference...")
    nbsvm_val_probs = nbsvm.predict_proba(val_feats)
    nbsvm_test_probs = nbsvm.predict_proba(test_feats)

    nbsvm_auc = calculate_roc_auc(y_val, nbsvm_val_probs)
    print(f"NBSVM Validation AUC: {nbsvm_auc}")

    # 4. RoBERTa Pipeline
    print("--- RoBERTa Pipeline ---")
    # Prepare DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=BATCH_SIZE, max_len=MAX_LEN
    )

    # Initialize Model
    model = ToxicRoBERTa(model_name=ROBERTA_MODEL, num_classes=len(LABEL_COLS))
    model.to(DEVICE)

    # Optimizer & Scheduler
    # Only decay weight for non-bias/LayerNorm parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=LR)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    print(f"Training RoBERTa for {EPOCHS} epoch(s)...")
    for epoch in range(EPOCHS):
        avg_loss = train_roberta_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, criterion
        )
        print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_loss:.4f}")

    # Inference RoBERTa
    print("RoBERTa Inference...")
    # infer_roberta returns probabilities (sigmoid applied)
    roberta_val_probs = infer_roberta(model, val_loader, DEVICE)
    roberta_test_probs = infer_roberta(model, test_loader, DEVICE)

    roberta_auc = calculate_roc_auc(y_val, roberta_val_probs)
    print(f"RoBERTa Validation AUC: {roberta_auc}")

    # 5. Ensemble
    print("--- Ensemble ---")
    final_val_probs = (ENSEMBLE_ALPHA * roberta_val_probs) + (
        (1 - ENSEMBLE_ALPHA) * nbsvm_val_probs
    )
    final_test_probs = (ENSEMBLE_ALPHA * roberta_test_probs) + (
        (1 - ENSEMBLE_ALPHA) * nbsvm_test_probs
    )

    final_auc = calculate_roc_auc(y_val, final_val_probs)

    # Required Output Format
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("--- Failure Analysis ---")
    # Calculate error magnitude (Mean Absolute Error per sample across all labels)
    error_magnitude = np.mean(np.abs(y_val - final_val_probs), axis=1)

    # Calculate input feature: Word count
    # Handle potential non-string values just in case, though data loader handles it
    word_counts = (
        val_df["comment_text"].astype(str).apply(lambda x: len(x.split())).values
    )

    # Calculate correlation
    correlation = np.corrcoef(error_magnitude, word_counts)[0, 1]
    print(f"Correlation between error magnitude and word count: {correlation}")

    # 7. Submission
    if final_auc > THRESHOLD:
        print(
            f"Validation metric passed threshold ({THRESHOLD}). Generating submission..."
        )
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        sub_df = pd.DataFrame(final_test_probs, columns=LABEL_COLS)
        sub_df.insert(0, "id", test_df["id"])

        sub_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
    else:
        print(
            f"Validation metric {final_auc} did not pass threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
