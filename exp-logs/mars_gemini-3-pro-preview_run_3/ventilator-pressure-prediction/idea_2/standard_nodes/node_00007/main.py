import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_dataloaders
from library.model import HybridCNNLSTM
from library.engine import train_one_epoch, evaluate, predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Initialization
    Config.initialize()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Use full dataset (debug=False) but rely on limited epochs for speed.
    # load_cached_data=True uses pre-computed .npy files if available.
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Model Setup
    model = HybridCNNLSTM().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 4. Training Loop
    # We limit epochs to 15 to ensure the "fast baseline" requirement is met
    # while still allowing convergence for this dataset size on A100.
    num_epochs = 15
    early_stopping_patience = 5
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = evaluate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_loss)

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # 5. Final Evaluation
    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Compute final metric on the full validation set
    final_metric = evaluate(model, val_loader, device)

    # Print required metric output
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Compute correlation between error and features on the validation set
    feature_indices = Config.get_feature_indices()
    u_out_idx = feature_indices["u_out"]

    errors = []
    features_list = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)

            # Extract u_out for masking
            u_out = x[:, :, u_out_idx]

            # Calculate absolute error
            abs_err = torch.abs(preds - y)

            # Mask error (only inspiratory phase matters for the metric)
            mask = 1 - u_out
            masked_err = abs_err * mask

            # Collect data
            errors.append(masked_err.cpu().numpy().flatten())
            features_list.append(x.cpu().numpy().reshape(-1, x.shape[-1]))

    all_errors = np.concatenate(errors)
    all_features = np.concatenate(features_list, axis=0)

    # Filter for inspiratory phase only (u_out == 0)
    # u_out is not scaled, so it is exactly 0 or 1
    u_out_flat = all_features[:, u_out_idx]
    inspiratory_indices = u_out_flat == 0

    valid_errors = all_errors[inspiratory_indices]
    valid_features = all_features[inspiratory_indices]

    print("Failure Analysis (Correlation with Error Magnitude):")
    df_analysis = pd.DataFrame(valid_features, columns=Config.FEATURE_COLS)
    df_analysis["error"] = valid_errors

    correlations = df_analysis.corr()["error"].drop("error")
    for name, val in correlations.items():
        print(f"{name}: {val:.4f}")

    # 7. Conditional Submission
    threshold = 0.8097341656684875
    if final_metric < threshold:
        predict(model, test_loader, device)
    else:
        pass  # Skip submission if metric is not good enough


if __name__ == "__main__":
    main()
