import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.data_loader import create_dataloaders
from library.trainer import ModelTrainer


def main():
    # 1. Setup and Configuration
    Config.set_seed(Config.SEED)
    Config.create_directories()
    device = Config.get_device()

    print(f"Using device: {device}")

    # 2. Data Loading
    # Using cached data if available for speed
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Training
    trainer = ModelTrainer(device=device)

    # Train the model (includes early stopping)
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 4. Validation Assessment
    print("Performing final validation assessment...")

    # We need to extract all validation data to compute the metric on the full set
    # and for failure analysis.
    trainer.model.eval()
    all_val_probs = []
    all_val_labels = []
    all_val_features = []

    with torch.no_grad():
        for features, labels, _ in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = trainer.model(features)
            probs = torch.sigmoid(logits)

            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())
            all_val_features.append(features.cpu().numpy())

    val_probs = np.vstack(all_val_probs)
    val_labels = np.vstack(all_val_labels)
    val_features = np.vstack(all_val_features)

    # Calculate robust ROC AUC
    # Using the helper method from trainer class logic locally or accessing it if public.
    # Since _robust_roc_auc is protected/internal in trainer, we re-implement the logic here
    # to ensure we print the exact metric required.

    aucs = []
    num_classes = val_labels.shape[1]
    for i in range(num_classes):
        # Check if both classes (0 and 1) are present
        if len(np.unique(val_labels[:, i])) > 1:
            try:
                auc = roc_auc_score(val_labels[:, i], val_probs[:, i])
                aucs.append(auc)
            except ValueError:
                pass

    final_metric = np.mean(aucs) if aucs else 0.5

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate error magnitude per sample (Mean Absolute Error across classes)
    # shape: (n_samples, )
    sample_errors = np.mean(np.abs(val_probs - val_labels), axis=1)

    # Calculate correlation between error and input features
    # val_features shape: (n_samples, n_features)
    n_features = val_features.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = val_features[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feature_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_errors, feature_vals)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for feat_idx, corr in correlations[:5]:
        print(f"Feature {feat_idx}: Correlation = {corr:.4f}")

    # 6. Submission Generation
    print("\nGenerating submission...")
    predictions, test_ids = trainer.predict(test_loader)

    if len(predictions) > 0:
        trainer.generate_submission(predictions, test_ids, Config.SUBMISSION_FILE_PATH)
    else:
        print("Error: No predictions generated.")


if __name__ == "__main__":
    main()
