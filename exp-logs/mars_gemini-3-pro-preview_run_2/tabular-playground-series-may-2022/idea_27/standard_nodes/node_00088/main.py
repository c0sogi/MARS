import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import get_model
from library.train import train_one_epoch, validate
from library.utils import seed_everything


def main():
    # 1. Configuration
    config = Config()

    # Ensure reproducible results
    seed_everything(config.SEED)

    # 2. Data Loading
    # We use the full dataset (max_train_samples=None) because the A100 is fast enough
    # and we need high precision to beat the threshold.
    print("Loading data...")
    train_dl, val_dl, test_dl, data_dict = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    model = get_model(config, vocab_size=data_dict["vocab_size"])
    model = model.to(config.DEVICE)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model_runfile.pth")
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Starting training for {config.EPOCHS} epochs on {config.DEVICE}...")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_dl, optimizer, criterion, config.DEVICE
        )
        val_loss, val_auc = validate(model, val_dl, criterion, config.DEVICE)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Validation
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))

    val_loss, final_val_auc = validate(model, val_dl, criterion, config.DEVICE)
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()

    all_targets = []
    all_preds = []
    all_cont_inputs = []

    with torch.no_grad():
        for batch in val_dl:
            x_seq, x_cont, y = batch
            x_seq = x_seq.to(config.DEVICE)
            x_cont_dev = x_cont.to(config.DEVICE)
            y = y.to(config.DEVICE).unsqueeze(1)

            logits = model(x_seq, x_cont_dev)
            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu())
            all_preds.append(probs.cpu())
            all_cont_inputs.append(x_cont.cpu())

    y_true = torch.cat(all_targets).numpy().flatten()
    y_pred = torch.cat(all_preds).numpy().flatten()
    X_cont = torch.cat(all_cont_inputs).numpy()

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Calculate correlation with continuous features
    # Feature names based on data processing (f_00 to f_30, excluding f_27)
    cont_cols = [f"f_{i:02d}" for i in range(31) if f"f_{i:02d}" != "f_27"]

    correlations = []
    for i, col_name in enumerate(cont_cols):
        if i < X_cont.shape[1]:
            # Compute Pearson correlation
            if np.std(X_cont[:, i]) > 0 and np.std(errors) > 0:
                corr = np.corrcoef(X_cont[:, i], errors)[0, 1]
            else:
                corr = 0.0
            correlations.append((col_name, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Input Features:")
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # 8. Submission
    THRESHOLD = 0.9970070375076856

    if final_val_auc > THRESHOLD:
        print(
            f"Validation metric {final_val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        test_preds = []
        model.eval()
        with torch.no_grad():
            for batch in test_dl:
                x_seq, x_cont = batch
                x_seq = x_seq.to(config.DEVICE)
                x_cont = x_cont.to(config.DEVICE)

                logits = model(x_seq, x_cont)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_preds.extend(probs.flatten())

        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        submission_df = pd.DataFrame(
            {"id": data_dict["test_ids"], "target": test_preds}
        )

        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_val_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
