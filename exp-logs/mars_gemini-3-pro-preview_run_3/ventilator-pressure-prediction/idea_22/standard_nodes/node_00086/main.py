import pandas as pd
import numpy as np
import torch
import os
import sys

# Ensure the library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library import config, utils, data, model, train


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = utils.get_device()

    # 2. Data Loading
    # Use cached data for speed. Use full dataset (debug=False) to ensure performance.
    train_loader, val_loader, test_loader = data.prepare_data(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    net = model.CWDHNet().to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    criterion = train.MaskedL1Loss()

    # Trainer
    trainer = train.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # 4. Training
    # Extended epochs to allow hybrid architecture to converge (Cite solution_lesson_node_00039)
    # Previous 12 epochs was insufficient for this complexity.
    FAST_EPOCHS = 50
    trainer.fit(epochs=FAST_EPOCHS)

    # 5. Validation & Metric Calculation
    # Load the best model saved during training
    if os.path.exists(config.MODEL_SAVE_PATH):
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    net.eval()

    val_preds_list = []
    val_targets_list = []
    val_u_outs_list = []
    val_inputs_list = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            preds = net(x)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(y.cpu())
            val_u_outs_list.append(u_out.cpu())
            val_inputs_list.append(x.cpu())

    val_preds = torch.cat(val_preds_list)
    val_targets = torch.cat(val_targets_list)
    val_u_outs = torch.cat(val_u_outs_list)
    val_inputs = torch.cat(val_inputs_list)

    # Calculate Metric: MAE on inspiratory phase (u_out == 0)
    mask = val_u_outs == 0
    abs_errors = torch.abs(val_preds - val_targets)
    masked_errors = abs_errors[mask]

    final_metric = masked_errors.mean().item()
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Failure Analysis:")
    # Get indices of inspiratory phase for analysis
    insp_indices = torch.nonzero(val_u_outs.flatten() == 0, as_tuple=True)[0]

    # Flatten errors and select inspiratory
    flat_errors_insp = abs_errors.flatten()[insp_indices].numpy()

    # Flatten inputs and select inspiratory
    # Input shape: (N, 80, Features) -> Flatten to (N*80, Features)
    num_features = val_inputs.shape[2]
    flat_inputs = val_inputs.view(-1, num_features)
    flat_inputs_insp = flat_inputs[insp_indices].numpy()

    correlations = {}
    for i, feature_name in enumerate(config.FEATURE_COLS):
        feat_values = flat_inputs_insp[:, i]
        # Calculate correlation if variance exists
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(feat_values, flat_errors_insp)[0, 1]
            correlations[feature_name] = corr
        else:
            correlations[feature_name] = 0.0

    # Sort by absolute correlation
    sorted_corrs = sorted(
        correlations.items(), key=lambda item: abs(item[1]), reverse=True
    )

    print("Top 5 Feature Correlations with Error (Inspiratory Phase):")
    for name, corr in sorted_corrs[:5]:
        print(f"{name}: {corr:.4f}")

    # 7. Conditional Submission
    THRESHOLD = 0.16391726930343686

    if final_metric < THRESHOLD:
        print("Metric check passed. Generating submission...")

        test_preds_list = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                preds = net(x)
                test_preds_list.append(preds.view(-1).cpu().numpy())

        all_test_preds = np.concatenate(test_preds_list)

        # Load test metadata to map predictions to IDs
        # Data loader sorts by breath_id and id_col, so we must replicate this sort
        test_df = pd.read_csv(config.TEST_PATH)
        test_df = test_df.sort_values(by=[config.BREATH_ID_COL, config.ID_COL])

        if len(test_df) != len(all_test_preds):
            print(
                f"Error: Prediction count {len(all_test_preds)} != Test DF count {len(test_df)}"
            )
        else:
            test_df["pressure"] = all_test_preds

            # Format: id, pressure. Sort by id for final submission.
            submission_df = test_df[[config.ID_COL, "pressure"]].sort_values(
                by=config.ID_COL
            )

            submission_df.to_csv(config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
