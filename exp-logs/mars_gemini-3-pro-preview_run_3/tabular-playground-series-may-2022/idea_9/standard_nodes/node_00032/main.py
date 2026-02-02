import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config, set_seed
from library.model import InputInjectedFunnelMLP
from library.data_utils import get_data_loaders
from library.train_eval import train_epoch, validate, predict


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # We use full data (debug_limit=None) to ensure we can hit the high AUC threshold.
    # load_cached_data=True ensures we use preprocessed tensors from ./working if available.
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab_sizes, cont_dim = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_limit=None
    )

    # 3. Model Initialization
    model = InputInjectedFunnelMLP(
        cont_dim=cont_dim,
        vocab_sizes=vocab_sizes,
        embed_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT,
    ).to(device)

    # 4. Optimizer & Scheduler
    # Using Config defaults but limiting epochs for "fast baseline" requirement
    # while balancing the need for high performance.
    epochs = 20

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 6. Final Evaluation
    print(f"Final Validation Metric: {best_auc}")

    # Load best model for analysis and inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Extract validation data for analysis
    # We access the underlying tensors from the dataset
    val_ds = val_loader.dataset
    x_cont_val = val_ds.x_cont.numpy()
    x_cat_val = val_ds.x_cat.numpy()
    y_val = val_ds.y.numpy().flatten()

    # Get predictions
    val_probs = []
    with torch.no_grad():
        for x_c, x_ca, _ in val_loader:
            x_c, x_ca = x_c.to(device), x_ca.to(device)
            logits = model(x_c, x_ca)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy())
    val_probs = np.concatenate(val_probs).flatten()

    # Calculate Error
    errors = np.abs(y_val - val_probs)

    # Construct DataFrame for correlation
    # Create column names
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]
    cat_cols = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    analysis_df = pd.DataFrame(x_cont_val, columns=cont_cols)
    for i, col in enumerate(cat_cols):
        analysis_df[col] = x_cat_val[:, i]

    analysis_df["error"] = errors

    # Compute correlation
    correlations = (
        analysis_df.corr()["error"].drop("error").abs().sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error:")
    print(correlations.head(5))

    # 8. Submission
    THRESHOLD = 0.9971550270448856

    if best_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({best_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = predict(model, test_loader, device)

        # Read sample submission or test file to get IDs
        test_df = pd.read_csv(Config.TEST_PATH)
        submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation AUC ({best_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
