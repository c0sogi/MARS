import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import WhaleConvNeXt
from library.train import train_one_epoch, validate, predict_test


def analyze_failures(model, loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_features = []

    print("Performing failure analysis...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            all_targets.append(labels.cpu())
            all_preds.append(probs)

            # Extract features from images (on CPU to save GPU mem)
            # images shape: (B, 1, 224, 224)
            # Flatten spatial dims to compute stats: mean, std, max
            imgs_flat = images.cpu().view(images.size(0), -1)

            batch_means = imgs_flat.mean(dim=1)
            batch_stds = imgs_flat.std(dim=1)
            batch_maxs = imgs_flat.max(dim=1).values

            # Stack features: (B, 3)
            batch_features = torch.stack([batch_means, batch_stds, batch_maxs], dim=1)
            all_features.append(batch_features)

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_features = torch.cat(all_features).numpy()

    # Calculate Error Magnitude: |y_true - y_pred|
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "spec_mean": all_features[:, 0],
            "spec_std": all_features[:, 1],
            "spec_max": all_features[:, 2],
        }
    )

    # Calculate correlation matrix
    corr_matrix = df_analysis.corr()

    # Extract correlations with error
    error_corrs = corr_matrix["error"].drop("error")

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    print(error_corrs)

    return error_corrs


def run():
    # 1. Setup
    # Override Config for fast baseline execution
    Config.EPOCHS = 5

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # Using cached data if available
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    model = WhaleConvNeXt()
    model = model.to(device)

    # 4. Optimizer, Loss, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Adjust T_max to the modified epoch count
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc:.6f}")

    # 6. Final Validation & Metric
    print("\nEvaluating best model on validation set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Warning: Best model not found, using current state.")

    val_loss, final_val_auc = validate(model, val_loader, criterion, device)

    # REQUIRED FORMAT
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.994260809807678

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_val_auc}) > threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test
        predictions = predict_test(model, test_loader, device)
        predictions = predictions.flatten()

        # Load Test Metadata for Clip IDs
        test_df = pd.read_csv(Config.TEST_CSV)

        if len(predictions) != len(test_df):
            print(
                f"Error: Prediction count {len(predictions)} != Test file count {len(test_df)}"
            )
        else:
            submission = pd.DataFrame(
                {"clip": test_df["clip"], "probability": predictions}
            )
            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_val_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
