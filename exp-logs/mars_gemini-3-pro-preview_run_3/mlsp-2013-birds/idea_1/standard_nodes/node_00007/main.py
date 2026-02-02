import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Import provided library modules
from library import config, dataset, model, trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model_net, val_loader, device):
    """
    Analyzes the correlation between model error and input features on the validation set.
    """
    model_net.eval()
    all_features = []
    all_probs = []
    all_labels = []

    # Collect data
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model_net(inputs)
            probs = torch.sigmoid(logits)

            all_features.append(inputs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    X_val = np.vstack(all_features)
    y_pred = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    # Calculate per-sample error (Mean Absolute Error across classes)
    # Shape: (N_samples, )
    per_sample_error = np.mean(np.abs(y_pred - y_true), axis=1)

    print("\n==== Failure Analysis ====")
    print(f"Average Per-Sample MAE: {np.mean(per_sample_error):.6f}")

    # Calculate correlation between each feature and the error
    # X_val shape: (N_samples, 100)
    # per_sample_error shape: (N_samples, )

    if X_val.ndim > 2:
        print(
            "Skipping feature correlation analysis for high-dimensional input (images)."
        )
        return

    correlations = []
    for i in range(X_val.shape[1]):
        feature_col = X_val[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feature_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, per_sample_error)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for feat_idx, corr in correlations[:5]:
        print(f"  Feature {feat_idx}: Correlation = {corr:.4f}")


def main():
    # 1. Setup
    dataset.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Using load_cached_data=True as requested
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # 3. Model Initialization
    net = model.BirdResNet(num_classes=config.NUM_CLASSES, pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # 4. Training Loop with Early Stopping
    best_auc = -1.0
    patience_counter = 0
    best_model_state = None

    # Limit max epochs if needed, but config.NUM_EPOCHS is 50 which is fine for this small dataset

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = trainer.train_epoch(
            net, train_loader, criterion, optimizer, device
        )

        scheduler.step()

        # Validate
        val_loss, val_auc = trainer.evaluate(net, val_loader, criterion, device)

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = net.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            break

    # 5. Final Evaluation and Metric Reporting
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    # Re-evaluate on validation set to confirm metric
    _, final_val_auc = trainer.evaluate(net, val_loader, criterion, device)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    perform_failure_analysis(net, val_loader, device)

    # 7. Submission Generation
    if final_val_auc > 0.8511215587581522:
        print(
            f"Metric improved ({final_val_auc:.6f} > 0.8511). Generating submission..."
        )
        test_predictions = trainer.predict(net, test_loader, device)
        trainer.save_submission(test_predictions, config.SUBMISSION_PATH)
    else:
        print(
            f"Metric did not improve ({final_val_auc:.6f} <= 0.8511). Skipping submission."
        )


if __name__ == "__main__":
    main()
