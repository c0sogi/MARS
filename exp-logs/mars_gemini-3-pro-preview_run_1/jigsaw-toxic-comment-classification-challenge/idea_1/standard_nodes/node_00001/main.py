import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.trainer import Trainer
from library.model import BiGRU_Pool_Net


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Adjust Config for a fast baseline run
    # 127k samples is small enough for A100 to run full data quickly (approx 30s/epoch)
    # We limit epochs to ensure quick execution.
    Config.EPOCHS = 3

    print("=" * 40)
    print("STARTING PIPELINE")
    print(f"Device: {Config.DEVICE}")
    print("=" * 40)

    # ==========================================
    # 2. Training
    # ==========================================
    # Initialize Trainer
    # load_cached_data=True ensures we use the preprocessed .npy files if available
    trainer = Trainer(debug=False, load_cached_data=True)

    # Load data (this populates train/val/test loaders)
    trainer.load_data()

    # Run training
    trainer.fit()

    # ==========================================
    # 3. Validation Assessment
    # ==========================================
    print("\n" + "=" * 40)
    print("VALIDATION ASSESSMENT")
    print("=" * 40)

    # Reload the best model for validation inference
    model = BiGRU_Pool_Net(
        vocab_size=Config.MAX_FEATURES,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.NUM_CLASSES,
        embedding_matrix=None,  # Weights will be loaded from state_dict
        dropout=Config.DROPOUT,
    )

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model file not found.")
        return

    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    val_loader = trainer.val_loader
    all_targets = []
    all_preds = []
    all_lengths = []

    # Inference loop without gradient calculation
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)

            # Calculate sequence lengths (non-padding tokens) for failure analysis
            # Assuming 0 is the PAD token index based on TextTokenizer implementation
            lengths = (inputs != 0).sum(dim=1).cpu().numpy()
            all_lengths.append(lengths)

            # Forward pass
            logits = model(inputs)
            preds = torch.sigmoid(logits)

            all_targets.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    seq_lengths = np.concatenate(all_lengths)

    # Calculate Mean Column-wise ROC AUC
    auc_scores = []
    for i in range(Config.NUM_CLASSES):
        try:
            # Check if class exists in this split
            if len(np.unique(y_true[:, i])) > 1:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
            else:
                score = 0.5
            auc_scores.append(score)
        except ValueError:
            auc_scores.append(0.5)

    final_metric = np.mean(auc_scores)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate Mean Absolute Error per sample (averaged across all 6 labels)
    # MAE is a good proxy for "error magnitude"
    sample_mae = np.mean(np.abs(y_true - y_pred), axis=1)

    # Calculate correlation between Sequence Length and Error
    correlation = np.corrcoef(seq_lengths, sample_mae)[0, 1]

    print(f"Correlation between Input Length and Model Error (MAE): {correlation:.6f}")

    if correlation > 0.1:
        print(
            "Observation: Positive correlation suggests longer comments are harder to classify."
        )
    elif correlation < -0.1:
        print(
            "Observation: Negative correlation suggests shorter comments are harder to classify."
        )
    else:
        print(
            "Observation: No significant linear correlation between length and error."
        )

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n" + "=" * 40)
    print("GENERATING SUBMISSION")
    print("=" * 40)

    # Use the trainer's method which handles test loader and formatting
    trainer.save_submission()


if __name__ == "__main__":
    main()
