import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import components from the provided library
from library.config import Config, set_seed
from library.data_processing import process_data, get_dataloaders
from library.training import train_model, generate_submission
from library.model import SNNModel


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for fast baseline execution
    # Reducing epochs and enabling cache loading
    Config.EPOCHS = 5
    Config.LOAD_CACHED_DATA = True

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("Initializing data processing...")
    train_df, val_df, test_df, meta = process_data(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # Subsample training data to limit runtime (Fast Baseline Requirement)
    # We use 100,000 samples which is sufficient for a quick check
    MAX_TRAIN_SAMPLES = 100000
    if len(train_df) > MAX_TRAIN_SAMPLES:
        print(f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples...")
        train_df = train_df.sample(
            n=MAX_TRAIN_SAMPLES, random_state=Config.SEED
        ).reset_index(drop=True)

    # Create DataLoaders
    # Note: Validation and Test loaders use full datasets
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df,
        val_df,
        test_df,
        meta,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Starting model training...")
    # train_model handles the training loop, validation monitoring, and saving best_model.pth
    train_model(train_loader, val_loader, meta)

    # ==========================================
    # 4. Final Validation & Failure Analysis
    # ==========================================
    print("Performing final validation and failure analysis...")
    device = Config.DEVICE

    # Initialize model architecture
    model = SNNModel(
        num_cont=len(meta["cont_cols"]),
        vocab_sizes=meta["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # Load the best model weights saved during training
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.eval()

    val_preds = []
    val_targets = []
    val_cont_inputs = []

    # Inference loop (No Grad for optimization)
    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_cont, x_cat).squeeze()
            probs = torch.sigmoid(logits)

            val_preds.extend(probs.cpu().numpy())
            val_targets.extend(y.numpy())

            # Store continuous features for failure analysis
            val_cont_inputs.append(x_cont.cpu().numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_cont_inputs = np.concatenate(val_cont_inputs, axis=0)

    # Compute and print the required metric
    final_metric = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between error magnitude and input features
    errors = np.abs(val_targets - val_preds)

    print("Failure Analysis (Feature Correlation with Error):")
    correlations = []
    cont_cols = meta["cont_cols"]

    for i, col_name in enumerate(cont_cols):
        # Calculate correlation coefficient
        feat_values = val_cont_inputs[:, i]
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        correlations.append((col_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 5 correlations
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9971550270448856

    if final_metric > THRESHOLD:
        print(
            f"Validation metric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(test_loader, meta)
    else:
        print(
            f"Validation metric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
