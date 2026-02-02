import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import get_dataloaders
from library.model import SDPEModel
from library.train_eval import train_one_epoch, evaluate, predict_and_submit, set_seed


def failure_analysis(model, val_loader, device):
    """
    Analyzes the correlation between model error and continuous features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_cont_features = []

    # Collect predictions, targets, and features
    with torch.no_grad():
        for batch in val_loader:
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            target = batch["target"].to(device)

            outputs = model(x_cont, x_cat)

            # Ensemble Mean
            probs = torch.stack([torch.sigmoid(out) for out in outputs])
            avg_prob = torch.mean(probs, dim=0).squeeze(1)

            all_preds.extend(avg_prob.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_cont_features.append(x_cont.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_cont_features = np.vstack(all_cont_features)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Load feature names from metadata cache
    try:
        metadata = np.load(Config.CACHE_METADATA, allow_pickle=True).item()
        cont_cols = metadata["cont_cols"]
    except Exception as e:
        print(
            f"Warning: Could not load metadata for feature names ({e}). Using indices."
        )
        cont_cols = [f"feat_{i}" for i in range(all_cont_features.shape[1])]

    # Calculate correlations
    correlations = []
    for i in range(all_cont_features.shape[1]):
        feat_vals = all_cont_features[:, i]
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        correlations.append(corr)

    # Create DataFrame for analysis
    df_corr = pd.DataFrame(
        {
            "feature": cont_cols,
            "correlation": correlations,
            "abs_correlation": np.abs(correlations),
        }
    )

    print("Top 5 features most correlated with error magnitude:")
    print(
        df_corr.sort_values("abs_correlation", ascending=False)
        .head(5)
        .to_string(index=False)
    )


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for Submission Path
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab_sizes = get_dataloaders(
        load_cached_data=False
    )

    # Infer input dimensions
    sample_batch = next(iter(train_loader))
    num_cont = sample_batch["x_cont"].shape[1]

    # 3. Model Initialization
    print("Initializing SDPE Model...")
    model = SDPEModel(vocab_sizes=vocab_sizes, num_cont=num_cont)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    final_val_auc = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = 0.9975746465492954
    if final_val_auc > THRESHOLD:
        print(
            f"Validation metric {final_val_auc} > {THRESHOLD}. Generating submission..."
        )
        predict_and_submit(model, test_loader, device)
    else:
        print(f"Validation metric {final_val_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
