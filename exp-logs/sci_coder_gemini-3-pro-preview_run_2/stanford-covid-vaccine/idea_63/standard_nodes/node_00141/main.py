import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import from the provided library files
from library.config import (
    seed_everything,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WORKING_DIR,
    VAL_PATH,
    TARGET_COLS,
    SCORED_LENGTH,
    SCORED_COLS,
)
from library.data import get_dataloaders
from library.model import HS_GFDN
from library.loss import MaskedMCRMSE
from library.train import train_one_epoch, validate, inference, generate_submission


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Using cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = HS_GFDN().to(device)
    criterion = MaskedMCRMSE().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 4. Fast Baseline Training
    # We limit epochs and batches to ensure quick execution as per requirements
    FAST_EPOCHS = 15
    MAX_BATCHES_PER_EPOCH = 50  # Limit training samples per epoch

    best_mcrmse = float("inf")
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    for epoch in range(FAST_EPOCHS):
        # Custom training loop to enforce batch limit
        model.train()
        running_loss = 0.0
        dataset_size = 0

        for i, batch in enumerate(train_loader):
            if i >= MAX_BATCHES_PER_EPOCH:
                break

            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            batch_size = inputs.size(0)

            optimizer.zero_grad()

            # Static Path
            x_permuted = inputs.permute(0, 2, 1)
            x_stem = model.stem(x_permuted)
            z = model.backbone(x_stem)

            # Pass 1: Zero Feedback
            N, _, L = z.shape
            e_fb_0 = torch.zeros(
                (N, 32, L), device=device, dtype=z.dtype
            )  # 32 is FEEDBACK_CHANNELS
            y_hat_1 = model.head(z, e_fb_0, partner_indices)

            # Pass 2: Recycled Feedback
            feedback = y_hat_1.detach()
            e_fb_1 = model.feedback_module(feedback)
            y_hat_2 = model.head(z, e_fb_1, partner_indices)

            # Loss
            loss_2 = criterion(y_hat_2, targets)
            loss_1 = criterion(y_hat_1, targets)
            loss = loss_2 + 0.5 * loss_1

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        # Validation
        val_mcrmse = validate(model, val_loader, device)
        scheduler.step(val_mcrmse)

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Compute and print final metric
    final_val_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Get predictions on validation set
    val_preds, val_ids = inference(model, val_loader, device)

    # Load validation metadata to correlate errors
    val_df = pd.read_csv(VAL_PATH)

    # Align dataframe with predictions using IDs
    val_df.set_index("id", inplace=True)
    val_df = val_df.reindex(val_ids).reset_index()

    # Parse ground truth targets for error calculation
    # Target shape: (N, 107, 5)
    targets = np.zeros((len(val_df), 107, 5))
    for idx, row in val_df.iterrows():
        for t_i, col in enumerate(TARGET_COLS):
            try:
                val_list = ast.literal_eval(row[col])
                if len(val_list) > 0:
                    targets[idx, : len(val_list), t_i] = val_list
            except:
                pass

    # Calculate RMSE per sample on scored columns/positions
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    preds_masked = val_preds[:, :SCORED_LENGTH, :]
    targets_masked = targets[:, :SCORED_LENGTH, :]

    # Select scored columns
    preds_selected = preds_masked[:, :, scored_indices]
    targets_selected = targets_masked[:, :, scored_indices]

    # Compute RMSE per sample
    mse_per_sample = np.mean((preds_selected - targets_selected) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    val_df["rmse"] = rmse_per_sample

    # Calculate correlations
    analysis_cols = ["signal_to_noise", "SN_filter", "mean_reactivity"]
    print("Correlation between Error (RMSE) and Metadata Features:")
    for col in analysis_cols:
        if col in val_df.columns:
            corr = val_df["rmse"].corr(val_df[col])
            print(f"  {col}: {corr}")

    # 7. Submission
    THRESHOLD = 0.47142532743789534
    if final_val_metric < THRESHOLD:
        print(f"\nMetric {final_val_metric} < {THRESHOLD}. Generating submission...")
        test_preds, test_ids = inference(model, test_loader, device)

        # Define submission path
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")

        generate_submission(test_preds, test_ids, sub_path)
    else:
        print(f"\nMetric {final_val_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
