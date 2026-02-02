import sys
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model as model_lib
from library import engine


def main():
    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    # Override defaults for a fast but effective baseline
    config.NUM_EPOCHS = 5  # Sufficient for DAN to converge on this data size
    config.BATCH_SIZE = 2048  # Increase batch size for A100 efficiency

    # Ensure reproducibility
    utils.set_seed(config.SEED)

    print(
        f"Configuration: Epochs={config.NUM_EPOCHS}, Batch Size={config.BATCH_SIZE}, Device={config.DEVICE}"
    )

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading and preprocessing data...")
    # get_dataloaders handles caching automatically.
    # First run will process raw CSVs; subsequent runs will load Parquet.
    train_loader, val_loader, test_loader, vocab, tag_encoder = data.get_dataloaders(
        debug=config.DEBUG, load_cached_data=True
    )

    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Number of Tags: {len(tag_encoder)}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing DAN Model...")
    dan_model = model_lib.DAN(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.HIDDEN_DIMS,
        output_dim=len(tag_encoder),
        dropout_rate=config.DROPOUT_RATE,
        padding_idx=0,  # Assuming 0 is PAD token based on Vocabulary class
    )
    dan_model.to(config.DEVICE)

    # ==========================================
    # 4. Training
    # ==========================================
    print("\nStarting Training...")
    engine.train_model(dan_model, train_loader, val_loader, config.DEVICE)

    # ==========================================
    # 5. Validation & Threshold Optimization
    # ==========================================
    print("\nOptimizing Prediction Threshold...")

    # Load the best model saved during training
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading best model from {config.MODEL_SAVE_PATH}")
        dan_model.load_state_dict(
            torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        )
    else:
        print("Warning: No saved model found. Using current weights.")

    # Run inference on validation set to get probabilities
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, val_probs, val_targets = engine.evaluate(
        dan_model, val_loader, config.DEVICE, criterion
    )

    # Find optimal threshold
    best_thresh, best_f1 = utils.optimize_threshold(val_targets, val_probs, steps=50)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {best_f1}")
    print(f"Optimal Threshold: {best_thresh}")

    # Update config for submission generation
    config.PREDICTION_THRESHOLD = best_thresh

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample F1 score
    # Binarize predictions using the optimal threshold
    val_preds_bin = (val_probs > best_thresh).astype(int)

    # Compute F1 for each sample manually
    # F1 = 2 * TP / (2 * TP + FP + FN) = 2 * TP / (Total_Pred + Total_True)
    tp = (val_preds_bin * val_targets).sum(axis=1)
    total_pred = val_preds_bin.sum(axis=1)
    total_true = val_targets.sum(axis=1)

    denominator = total_pred + total_true
    # Handle division by zero (if both pred and true are empty, F1 is 1, else 0)
    # However, in this dataset, tags are always present.
    f1_scores = np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0
    )

    # Error magnitude
    errors = 1.0 - f1_scores

    # Get input lengths (features)
    # We reload the validation dataframe to access the 'input_ids' length directly.
    # Using load_cached_data=True ensures this is fast.
    val_df, _, _ = data.prepare_data(
        "val",
        vocab=vocab,
        tag_encoder=tag_encoder,
        load_cached_data=True,
        debug=config.DEBUG,
    )

    # Calculate length of each sequence (number of tokens)
    # Note: val_df order matches val_loader because shuffle=False in get_dataloaders for val
    input_lengths = val_df["input_ids"].apply(len).values

    # Calculate correlation
    if len(errors) == len(input_lengths):
        correlation = np.corrcoef(errors, input_lengths)[0, 1]
        print(f"Correlation between Error (1-F1) and Input Length: {correlation}")
    else:
        print("Warning: Mismatch in validation set sizes for analysis.")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    print("\nGenerating Submission...")
    engine.generate_submission(dan_model, test_loader, tag_encoder, config.DEVICE)
    print("Process Complete.")


if __name__ == "__main__":
    main()
