import pandas as pd
import numpy as np
import torch
import os
import sys

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import ToxicityRoBERTa
from library.engine import run_training, evaluate, predict, set_seed


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for optimal trade-off between speed and performance.
    # Config.EPOCHS is set to 3 in config.py
    Config.DEBUG = False

    # Set random seed for reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading Data...")
    # load_cached_data=True allows using pre-processed npy files if they exist in ./working
    train_loader, val_loader, tokenizer = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing Model...")
    model = ToxicityRoBERTa()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("Starting Training...")
    # run_training handles the loop, validation, and saves the best model to Config.MODEL_SAVE_PATH
    run_training(model, train_loader, val_loader)

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    print("Loading best model for final evaluation...")
    # Load the best state dict saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    else:
        print("Warning: Model save file not found. Using current model state.")

    model.to(Config.DEVICE)

    # Calculate metric on validation set
    val_loss, val_auc = evaluate(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("Performing Failure Analysis...")
    model.eval()

    all_targets = []
    all_preds = []
    all_lengths = []

    # Collect predictions, targets, and metadata (lengths) manually to compute sample-wise errors
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            targets = batch["labels"].to(Config.DEVICE)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs)

            # Store data
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            # Length is the sum of the attention mask (1 for token, 0 for pad)
            all_lengths.append(attention_mask.sum(dim=1).cpu().numpy())

    # Concatenate batches
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    all_lengths = np.concatenate(all_lengths)

    # Calculate Mean Absolute Error (MAE) per sample across all 6 labels
    # Shape: (N_samples,)
    sample_errors = np.mean(np.abs(all_targets - all_preds), axis=1)

    # Calculate correlation between error magnitude and input length
    corr_matrix = np.corrcoef(sample_errors, all_lengths)
    correlation = corr_matrix[0, 1]

    print(f"Correlation between Model Error and Input Length: {correlation:.6f}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    # Threshold defined in task description
    THRESHOLD = 0.9920650979347099

    if val_auc > THRESHOLD:
        print(
            f"Validation metric ({val_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Get Test Loader
        test_loader = get_test_dataloader(tokenizer, load_cached_data=True)

        # Predict on Test Set
        test_probs = predict(model, test_loader, Config.DEVICE)

        # Load Sample Submission to preserve IDs and structure
        submission_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Assign predictions to the label columns
        # Ensure columns are in the correct order as per Config.LABEL_COLS
        submission_df[Config.LABEL_COLS] = test_probs

        # Save submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric ({val_auc}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
