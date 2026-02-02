import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import soundfile as sf
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import HierarchicalCRNN
from library.engine import train_model, evaluate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def pearson_corr(x, y):
    """
    Calculates Pearson correlation coefficient between two numpy arrays.
    Used to avoid dependency on scipy if not strictly guaranteed,
    though scipy is likely available.
    """
    if len(x) != len(y):
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(
        np.sum((y - y_mean) ** 2)
    )

    if denominator == 0:
        return 0.0
    return numerator / denominator


def perform_failure_analysis(model, val_loader, val_csv_path, device):
    """
    Analyzes model failure modes by correlating error with input signal properties.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    # 1. Get Predictions and Targets
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            # Forward pass
            output = model(data)
            preds = torch.sigmoid(output).cpu().numpy().flatten()
            targets = target.numpy().flatten()

            all_preds.extend(preds)
            all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 2. Calculate Absolute Error
    errors = np.abs(all_targets - all_preds)

    # 3. Extract Features for Correlation (RMS Energy)
    print("Extracting audio features (RMS) for failure correlation...")
    df_val = pd.read_csv(val_csv_path)

    if len(df_val) != len(errors):
        print(
            f"Warning: Validation set size mismatch. DF: {len(df_val)}, Preds: {len(errors)}"
        )
        return

    rms_values = []
    for _, row in df_val.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        try:
            # Read audio to compute RMS
            y, sr = sf.read(filepath)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            rms = np.sqrt(np.mean(y**2))
            rms_values.append(rms)
        except Exception:
            rms_values.append(0.0)

    rms_values = np.array(rms_values)

    # 4. Compute Correlations
    # Correlation with Signal Energy
    corr_rms = pearson_corr(errors, rms_values)
    print(f"Correlation between Error and Audio RMS: {corr_rms:.4f}")

    # Correlation with Ground Truth (Bias check)
    corr_label = pearson_corr(errors, all_targets)
    print(f"Correlation between Error and Ground Truth Label: {corr_label:.4f}")

    # Top failures
    print("Top 3 worst predictions:")
    sorted_indices = np.argsort(errors)[::-1]
    for i in range(3):
        idx = sorted_indices[i]
        print(
            f"Clip: {df_val.iloc[idx]['clip']}, True: {all_targets[idx]}, Pred: {all_preds[idx]:.4f}, Error: {errors[idx]:.4f}"
        )


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    # Optimize Batch Size for A100 GPU (40GB VRAM allows larger batches)
    Config.BATCH_SIZE = 128

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load data using cached .npy files if available
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = HierarchicalCRNN().to(device)

    # ==========================================
    # 4. Training
    # ==========================================
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=Config.EPOCHS,
    )

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    # Load the best model checkpoint
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Define criterion for loss calculation during evaluation
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Evaluate
    val_loss, final_val_auc = evaluate(model, val_loader, criterion, device)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {final_val_auc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    perform_failure_analysis(model, val_loader, Config.VAL_CSV, device)

    # ==========================================
    # 7. Submission
    # ==========================================
    threshold = 0.9946524988681537

    if final_val_auc > threshold:
        print(
            f"Validation AUC ({final_val_auc}) meets threshold ({threshold}). Generating submission..."
        )

        model.eval()
        test_preds = []

        # Inference on Test Set
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(output).cpu().numpy().flatten()
                test_preds.extend(probs)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"clip": test_ids, "probability": test_preds})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation AUC ({final_val_auc}) does not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
