import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import seed_everything, custom_weight_init
from library.data_loader import get_dataloaders
from library.model import BalancedProcessCompressHybrid
from library.train_eval import train_one_epoch, validate


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load cached data for speed
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Extract dimensions for model initialization
    # Get a single batch to inspect shapes
    sample_x_num, sample_x_cat, _ = next(iter(train_loader))
    num_continuous = sample_x_num.shape[1]
    cat_seq_len = sample_x_cat.shape[1]

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = BalancedProcessCompressHybrid(
        num_continuous=num_continuous,
        cat_seq_len=cat_seq_len,
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_heads=Config.TRANSFORMER_HEADS,
        backbone_stages=Config.BACKBONE_STAGES,
        dropout_transformer=Config.DROPOUT_TRANSFORMER,
        dropout_backbone=Config.DROPOUT_BACKBONE,
    ).to(device)

    # Apply context-aware weight initialization
    model.apply(custom_weight_init)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Aggressive Step Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = Config.BEST_MODEL_PATH

    # Use Config.EPOCHS (40) to ensure convergence to the high threshold
    epochs = Config.EPOCHS

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # --------------------------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # --------------------------------------------------------------------------
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    # Collect all validation predictions and inputs for analysis
    val_preds = []
    val_targets = []
    val_inputs_num = []

    with torch.no_grad():
        for x_num, x_cat, y in val_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            logits = model(x_num, x_cat).squeeze()
            preds = torch.sigmoid(logits)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_inputs_num.append(x_num.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_inputs_num = np.concatenate(val_inputs_num)

    # Calculate Final Metric
    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(val_targets - val_preds)

    # Calculate correlation between error magnitude and numerical features
    correlations = []
    n_features = val_inputs_num.shape[1]

    for i in range(n_features):
        feature_col = val_inputs_num[:, i]
        # Check for constant features to avoid NaN correlation
        if np.std(feature_col) > 1e-9:
            corr = np.corrcoef(errors, feature_col)[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 6. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9970005855169476

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for x_num, x_cat in test_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)

                logits = model(x_num, x_cat).squeeze()
                preds = torch.sigmoid(logits)
                test_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds)

        # Retrieve Test IDs
        # Try loading from cache
        cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            test_ids = data["test_ids"]
        else:
            # Fallback
            test_meta = pd.read_csv(
                os.path.join(Config.METADATA_DIR, "test_metadata.csv")
            )
            test_ids = test_meta["id"].values

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nValidation metric {final_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
