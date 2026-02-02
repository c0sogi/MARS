import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import (
    VAL_PATH,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSION_PATH,
    CACHE_DIR,
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EMBED_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    SEED,
)
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.vocabulary import build_vocabulary
from library.model import BiLSTM, train_model, predict, evaluate


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    # We use load_cached_data=True to leverage any pre-processing done in working dir
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    # Build vocabulary to get size for model initialization
    vocab = build_vocabulary(load_cached_data=True)
    vocab_size = len(vocab.stoi)
    print(f"Vocabulary Size: {vocab_size}")

    # 3. Model Initialization
    print("Initializing BiLSTM Model...")
    model = BiLSTM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
    )

    # 4. Training
    # The config specifies 5 epochs, which is fast enough for ~3k samples.
    print("Starting Training...")
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_path=best_model_path,
    )

    # 5. Validation Assessment
    print("Performing Validation Assessment...")
    criterion = torch.nn.BCELoss()
    val_loss, val_auc = evaluate(trained_model, val_loader, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    # Load the raw validation dataframe to extract text features
    # We assume the order in val_loader matches val.csv (shuffle=False)
    val_df = pd.read_csv(VAL_PATH)

    # Get predictions
    val_probs = predict(trained_model, val_loader, DEVICE)

    if len(val_probs) != len(val_df):
        print(
            f"Warning: Mismatch in validation set size. Preds: {len(val_probs)}, DF: {len(val_df)}"
        )
        min_len = min(len(val_probs), len(val_df))
        val_probs = val_probs[:min_len]
        val_df = val_df.iloc[:min_len]

    # Calculate Error Magnitude
    # Target is 'Insult' column
    y_true = val_df["Insult"].values
    error_magnitude = np.abs(y_true - val_probs)

    # Extract Features: Character Length and Word Length
    # Handle potential non-string entries in 'Comment'
    val_df["Comment"] = val_df["Comment"].astype(str)
    char_lens = val_df["Comment"].apply(len)
    word_lens = val_df["Comment"].apply(lambda x: len(x.split()))

    # Calculate Correlations
    corr_char = np.corrcoef(error_magnitude, char_lens)[0, 1]
    corr_word = np.corrcoef(error_magnitude, word_lens)[0, 1]

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (Error vs Char Length): {corr_char:.4f}")
    print(f"Correlation (Error vs Word Length): {corr_word:.4f}")
    print("-" * 30)

    # 7. Submission Generation
    if val_auc > 0.861867816091954:
        print("Generating Submission...")
        test_probs = predict(trained_model, test_loader, DEVICE)

        # Load sample submission
        if not os.path.exists(SAMPLE_SUBMISSION_PATH):
            raise FileNotFoundError(
                f"Sample submission not found at {SAMPLE_SUBMISSION_PATH}"
            )

        submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        # Check length
        if len(test_probs) != len(submission_df):
            print(
                f"Warning: Prediction length ({len(test_probs)}) matches test set but differs from sample submission ({len(submission_df)})."
            )

        # Assign predictions
        submission_df["Insult"] = test_probs

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation AUC ({val_auc}) did not meet the threshold. Skipping submission."
        )


if __name__ == "__main__":
    main()
