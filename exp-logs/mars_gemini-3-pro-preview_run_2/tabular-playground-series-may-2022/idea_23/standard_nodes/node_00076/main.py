import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure local library imports work
sys.path.append(os.getcwd())

from library import config, utils, data, model, train


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    net = model.HybridNetwork().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0

    # Ensure model directory exists
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print(f"Starting training for {config.EPOCHS} epochs...")
    for epoch in range(1, config.EPOCHS + 1):
        # Train
        train_loss = train.train_one_epoch(
            net, train_loader, optimizer, criterion, device
        )

        # Validate
        val_auc = train.validate(net, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)

        # Logging
        print(
            f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

    # 6. Final Evaluation
    print(f"Final Validation Metric: {best_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load best model
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    all_errors = []
    all_cont_inputs = []

    # Collect validation inputs and errors
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            outputs = net(continuous, sequence)
            preds = torch.sigmoid(outputs)

            # Calculate absolute error
            error = torch.abs(targets - preds).cpu().numpy()

            all_errors.append(error)
            all_cont_inputs.append(continuous.cpu().numpy())

    # Flatten and concatenate
    all_errors = np.concatenate(all_errors).flatten()
    all_cont_inputs = np.concatenate(all_cont_inputs, axis=0)

    # Create DataFrame for analysis
    # Feature names: f_00 to f_30, excluding f_27
    feature_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    df_analysis = pd.DataFrame(all_cont_inputs, columns=feature_names)
    df_analysis["error"] = all_errors

    # Compute correlation
    correlations = (
        df_analysis.corr()["error"].drop("error").abs().sort_values(ascending=False)
    )

    print("Top 5 Features Correlated with Prediction Error:")
    print(correlations.head(5))

    # 8. Submission
    THRESHOLD = 0.9970005855169476

    if best_auc > THRESHOLD:
        print(
            f"\nMetric {best_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions
        predictions = train.predict(net, test_loader, device)

        # Load Test Metadata for IDs
        test_meta = pd.read_csv(config.TEST_META_PATH)

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_meta["id"], "target": predictions})

        # Save
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {best_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
