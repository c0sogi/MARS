import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure imports work correctly
sys.path.append(os.getcwd())

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_processing import get_data_loaders
from library.model import MultiGranularityNet
from library.train_eval import train_one_epoch, evaluate, predict


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Override Config for fast baseline execution while maintaining performance
    Config.EPOCHS = 20

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    train_loader, val_loader, test_loader, vocab_sizes = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
    )

    # Determine continuous feature count dynamically from a sample batch
    sample_cont, _, _ = next(iter(train_loader))
    num_cont_features = sample_cont.shape[1]

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = MultiGranularityNet(
        vocab_sizes=vocab_sizes, num_cont_features=num_cont_features
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validate
        val_auc = evaluate(model, val_loader, device)

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                filename=Config.MODEL_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # ==========================================
    # 5. Final Evaluation
    # ==========================================
    # Load best model weights
    load_checkpoint(model, filename=Config.MODEL_PATH, device=Config.DEVICE)

    # Compute final metric on full validation set
    final_val_auc = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    model.eval()
    all_targets = []
    all_preds = []
    all_cont_inputs = []

    # Collect predictions and inputs for analysis
    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            all_cont_inputs.append(x_cont.cpu().numpy())

    all_targets = np.concatenate(all_targets).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    all_cont_inputs = np.concatenate(all_cont_inputs, axis=0)

    # Calculate Absolute Error
    errors = np.abs(all_targets - all_preds)

    # Construct DataFrame for correlation analysis
    # Reconstruct continuous column names based on data_processing.py logic
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]
    cont_cols.append("unique_character_count")

    df_analysis = pd.DataFrame(all_cont_inputs, columns=cont_cols)
    df_analysis["error_magnitude"] = errors

    # Calculate correlations between features and error magnitude
    correlations = (
        df_analysis.corrwith(df_analysis["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("\nFailure Analysis - Feature Correlation with Error Magnitude:")
    print(correlations.head(10))

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.9971550270448856

    if final_val_auc > THRESHOLD:
        test_preds = predict(model, test_loader, device)

        # Load test IDs from metadata
        test_df = pd.read_csv(Config.TEST_CSV)

        submission = pd.DataFrame(
            {
                Config.ID_COL: test_df[Config.ID_COL],
                Config.TARGET_COL: test_preds.flatten(),
            }
        )

        Config.create_dirs()
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
