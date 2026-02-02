import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from library
from library.config import Config
from library.data_utils import preprocess_pipeline, add_engineered_features
from library.dataset import create_dataloaders
from library.model import ManufacturingTransformer
from library.train_eval import train_one_epoch, evaluate, predict, set_seed


def main():
    # 1. Configuration Override for Fast Baseline
    # Optimizing for A100: Large batch size and reduced epochs for speed
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 4096

    # 2. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 3. Data Loading
    print("Loading and preprocessing data...")
    # Load cached data if available to save time
    data_dict, vocab_size = preprocess_pipeline(load_cached_data=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 4. Model Initialization
    num_numerical_features = data_dict["X_num_train"].shape[1]
    print(
        f"Initializing model with {num_numerical_features} numerical features and vocab size {vocab_size}..."
    )

    model = ManufacturingTransformer(
        num_numerical_features=num_numerical_features,
        vocab_size=vocab_size,
        seq_len=Config.SEQ_LEN,
    ).to(device)

    # 5. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 6. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 7. Final Metrics
    print(f"Final Validation Metric: {best_auc}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Get probabilities on validation set
    val_probs = predict(model, val_loader, device)
    y_val = data_dict["y_val"]

    # Calculate residuals (Error Magnitude)
    residuals = np.abs(y_val - val_probs)

    # Reconstruct feature names to map correlations correctly
    # We load a small sample and apply the same engineering logic
    df_meta = pd.read_csv(Config.TRAIN_METADATA_PATH, nrows=10)
    df_meta = add_engineered_features(df_meta)

    ignore = set(
        Config.IGNORE_COLS
        + [Config.ID_COL, Config.TARGET_COL, "source_path", Config.SEQUENCE_COL]
    )
    numeric_candidates = df_meta.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_candidates if c not in ignore]

    X_num_val = data_dict["X_num_val"]

    correlations = []
    # Ensure we don't go out of bounds if feature lists mismatch (though they shouldn't)
    num_feats_to_check = min(len(feature_cols), X_num_val.shape[1])

    for idx in range(num_feats_to_check):
        col_name = feature_cols[idx]
        feat_values = X_num_val[:, idx]
        # Calculate Pearson correlation
        corr = np.corrcoef(feat_values, residuals)[0, 1]
        correlations.append((col_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 9. Submission
    THRESHOLD = 0.9965336074216435
    if best_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({best_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        test_probs = predict(model, test_loader, device)
        ids_test = data_dict["ids_test"]

        submission = pd.DataFrame({"id": ids_test, "target": test_probs})

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation AUC ({best_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
