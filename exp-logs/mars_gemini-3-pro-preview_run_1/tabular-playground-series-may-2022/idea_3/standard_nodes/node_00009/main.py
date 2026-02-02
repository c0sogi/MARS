import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, save_submission
from library.dataset import get_dataloaders, process_data
from library.model import HybridTransformerModel, train_one_epoch, validate


def main():
    # 1. Setup and Configuration
    # Limit epochs for a fast baseline, but enough to reach convergence on A100
    Config.EPOCHS = 6
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # We load raw arrays first to get feature counts and for later failure analysis
    print("Loading data arrays...")
    (
        (X_train_seq, X_train_num, y_train),
        (X_val_seq, X_val_num, y_val),
        (X_test_seq, X_test_num, ids_test),
    ) = process_data(load_cached_data=True)

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, batch_size=Config.BATCH_SIZE
    )

    num_numerical_features = X_train_num.shape[1]
    print(f"Number of numerical features: {num_numerical_features}")

    # 3. Model Initialization
    model = HybridTransformerModel(num_numerical_features=num_numerical_features)
    model.to(device)

    # 4. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.SCHEDULER_WARMUP_PCT,
        anneal_strategy="cos",
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_runfile.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 6. Evaluation and Failure Analysis
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Re-run validation to get predictions for analysis
    # We use the validate function logic but capture outputs manually to ensure alignment
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            num = batch["numerical"].to(device)
            targets = batch["target"].to(device)

            logits = model(seq, num)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_probs.extend(probs)
            val_targets.extend(targets.cpu().numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    final_auc = roc_auc_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # Create a DataFrame for correlation calculation
    # We use the numerical features from the validation set
    # Note: X_val_num matches the order of the validation loader (no shuffling in val_loader)
    df_analysis = pd.DataFrame(
        X_val_num, columns=[f"f_{i:02d}" for i in range(num_numerical_features)]
    )
    df_analysis["error"] = errors

    # Calculate correlation
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("Top 5 features correlated with error magnitude:")
    print(correlations.head(5))

    # 7. Conditional Submission
    THRESHOLD = 0.9916039887689444

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        test_probs = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(device)
                num = batch["numerical"].to(device)
                ids = batch["id"].numpy()

                logits = model(seq, num)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                test_probs.extend(probs)
                test_ids.extend(ids)

        save_submission(test_ids, test_probs, Config.SUBMISSION_FILE)
    else:
        print(f"\nValidation metric {final_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
