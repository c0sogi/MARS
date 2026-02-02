import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet
from library.train import train_one_epoch, validate_one_epoch


def main():
    # 1. Setup & Configuration
    # Adjust epochs for fast baseline execution while maintaining performance
    Config.NUM_EPOCHS = 15

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True to utilize preprocessed data if available
    train_loader, val_loader, test_loader = get_dataloaders(
        Config, load_cached_data=True
    )

    # 3. Model Initialization
    model = WhaleEfficientNet(Config)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config
        )
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Validation & Failure Analysis
    print("\nRunning Final Validation and Failure Analysis...")

    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: No best model found. Using current model.")

    model.eval()

    val_probs = []
    val_targets = []
    val_features = []  # Store mean, std, max of spectrograms

    with torch.no_grad():
        for data, target, _ in val_loader:
            data = data.to(device)
            target = target.to(device).view(-1, 1)

            # Forward pass
            output = model(data)
            probs = torch.sigmoid(output)

            # Store predictions and targets
            val_probs.extend(probs.cpu().numpy().flatten())
            val_targets.extend(target.cpu().numpy().flatten())

            # Calculate features for failure analysis
            # Move to CPU for numpy operations
            data_cpu = data.cpu().numpy()
            # Flatten spatial dims: (B, F*T)
            data_flat = data_cpu.reshape(data_cpu.shape[0], -1)

            means = np.mean(data_flat, axis=1)
            stds = np.std(data_flat, axis=1)
            maxs = np.max(data_flat, axis=1)

            batch_features = np.stack([means, stds, maxs], axis=1)
            val_features.append(batch_features)

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    val_features = np.concatenate(val_features, axis=0)

    # Calculate Final Metric
    final_auc = calculate_roc_auc(val_targets, val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    errors = np.abs(val_targets - val_probs)
    feature_names = ["Spec_Mean", "Spec_Std", "Spec_Max"]

    print("\nFailure Analysis (Correlation between Error and Input Features):")
    for i, name in enumerate(feature_names):
        feat_vals = val_features[:, i]
        if np.std(feat_vals) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        print(f"{name}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.994260809807678
    if final_auc > THRESHOLD:
        print(f"\nMetric {final_auc} > {THRESHOLD}. Generating submission...")

        test_clips = []
        test_probs = []

        with torch.no_grad():
            for data, _, clips in test_loader:
                data = data.to(device)

                output = model(data)
                probs = torch.sigmoid(output).cpu().numpy().flatten()

                test_clips.extend(clips)
                test_probs.extend(probs)

        submission_df = pd.DataFrame({"clip": test_clips, "probability": test_probs})

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric {final_auc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
