import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import random

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_dataloaders
from library.modules import HybridResFunnelModel
from library.engine import (
    get_optimizer_params,
    initialize_weights,
    train_one_epoch,
    evaluate,
)


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction errors with input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()
    all_probs = []
    all_targets = []
    all_continuous = []

    # Efficient inference
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            logits = model(continuous, sequence)
            probs = torch.sigmoid(logits).squeeze(1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_continuous.append(continuous.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_continuous = np.concatenate(all_continuous)

    # Calculate residuals (absolute error)
    errors = np.abs(all_targets - all_probs)

    # Calculate correlation between continuous features and error
    correlations = []
    num_features = all_continuous.shape[1]

    for i in range(num_features):
        # Avoid division by zero if feature is constant
        if np.std(all_continuous[:, i]) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(all_continuous[:, i], errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        # Map index back to feature name (skipping f_27)
        # 0-26 -> f_00-f_26; 27-29 -> f_28-f_30
        feat_name = f"f_{idx:02d}" if idx < 27 else f"f_{idx+1:02d}"
        print(f"{feat_name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Configure for Fast Baseline
    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # 3. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 4. Model Initialization
    print("Initializing model...")
    model = HybridResFunnelModel(config=Config).to(device)
    initialize_weights(model)

    # 5. Optimizer & Scheduler
    optimizer_params = get_optimizer_params(
        model,
        weight_decay_encoder=Config.WEIGHT_DECAY_ENCODER,
        weight_decay_bias=Config.WEIGHT_DECAY_BIAS,
    )
    optimizer = optim.AdamW(optimizer_params, lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_auc = 0.0
    best_model_path = Config.MODEL_SAVE_PATH

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 7. Final Evaluation
    print("Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, final_val_auc = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # 8. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 9. Conditional Submission
    threshold = 0.9972336610045187
    if final_val_auc > threshold:
        print(
            f"Validation metric ({final_val_auc}) > threshold ({threshold}). Generating submission..."
        )
        model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                continuous = batch["continuous"].to(device)
                sequence = batch["sequence"].to(device)

                logits = model(continuous, sequence)
                probs = torch.sigmoid(logits).squeeze(1)
                all_preds.extend(probs.cpu().numpy())

        submission_df = pd.DataFrame({"id": test_ids, "target": all_preds})

        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric ({final_val_auc}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
