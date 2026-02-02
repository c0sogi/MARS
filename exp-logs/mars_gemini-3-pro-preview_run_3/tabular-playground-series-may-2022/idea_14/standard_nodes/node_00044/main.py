import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.data import preprocess_data
from library.model import GatedFunnelNetwork
from library.train_eval import train_one_epoch, evaluate, generate_submission
from library.utils import compute_auc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run():
    # 1. Configuration & Setup
    set_seed()

    # Fast baseline settings: Limit epochs to ensure quick execution
    # The A100 is fast, but we limit to 10 epochs to strictly meet "fast baseline" constraints
    Config.EPOCHS = 10

    device = Config.DEVICE

    # 2. Data Loading
    # Load cached data to save time
    train_loader, val_loader, test_loader, vocab_sizes, num_cont = preprocess_data(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    model = GatedFunnelNetwork(vocab_sizes, num_cont, Config).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize OneCycleLR with the modified epoch count
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_PATH

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Early stopping and model checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                break

    # 6. Final Validation & Metric
    # Load best model for final evaluation
    best_model = GatedFunnelNetwork(vocab_sizes, num_cont, Config).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-evaluate to get exact metric on full validation set
    _, final_val_auc = evaluate(best_model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    best_model.eval()

    all_targets = []
    all_probs = []
    all_cont_features = []

    # Collect predictions, targets, and features for analysis
    with torch.no_grad():
        for x_cat, x_cont, y in val_loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device)

            logits = best_model(x_cat, x_cont)
            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_cont_features.append(x_cont.cpu().numpy())

    all_targets = np.concatenate(all_targets).flatten()
    all_probs = np.concatenate(all_probs).flatten()
    all_cont_features = np.concatenate(all_cont_features, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_probs)

    # Reconstruct feature names (logic matches library/data.py)
    cont_features_base = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_char_count"
    ]

    print("Correlation between Error Magnitude and Continuous Features:")
    correlations = []
    for i, feat_name in enumerate(cont_features_base):
        if i < all_cont_features.shape[1]:
            feat_values = all_cont_features[:, i]
            # Calculate Pearson correlation
            if np.std(feat_values) > 0 and np.std(errors) > 0:
                corr = np.corrcoef(errors, feat_values)[0, 1]
            else:
                corr = 0.0
            correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations[:10]:  # Print top 10 correlations
        print(f"{name}: {corr:.4f}")

    # 8. Conditional Submission
    THRESHOLD = 0.9971550270448856
    if final_val_auc > THRESHOLD:
        generate_submission(best_model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    run()
