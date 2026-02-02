import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import sys

# Import provided library modules
from library import config, utils, data, model, train


def main():
    print("=== Ventilator Pressure Prediction Demo ===")

    # 1. Setup
    # Ensure reproducibility
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    print(f"Device selected: {device}")

    # 2. Data Preparation (Debug Mode)
    print("\n[Data] Preparing data (Debug Mode)...")
    # We set load_cached_data=False to force processing of the debug subset
    train_loader, val_loader, test_loader = data.prepare_data(
        load_cached_data=False, debug=True
    )

    # Verify Data Integrity
    print("[Data] Verifying batch shapes...")
    sample_batch = next(iter(train_loader))
    x, u_out, y = sample_batch["x"], sample_batch["u_out"], sample_batch["y"]

    # Expected shape: (Batch_Size, Seq_Len=80, Features)
    print(f"  Input X: {x.shape}")
    print(f"  Target y: {y.shape}")

    assert x.shape[1] == 80, "Sequence length must be 80"
    assert x.shape[2] == len(config.FEATURE_COLS), "Feature dimension mismatch"
    assert y.shape == (config.BATCH_SIZE, 80), "Target shape mismatch"

    # 3. Model Initialization
    print("\n[Model] Initializing PERDHNet...")
    net = model.PERDHNet().to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_out = net(x.to(device))
    assert dummy_out.shape == (config.BATCH_SIZE, 80), "Model output shape mismatch"
    print("  Forward pass successful.")

    # 4. Training Configuration
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = train.MaskedL1Loss()

    trainer = train.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # 5. Training Loop
    print("\n[Training] Starting 1 epoch of training...")
    trainer.fit(epochs=1)

    # 6. Inference
    print("\n[Inference] Generating predictions...")
    # Load the best model saved during training
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x_in = batch["x"].to(device)
            preds = net(x_in)
            predictions.append(preds.view(-1).cpu().numpy())

    all_predictions = np.concatenate(predictions)
    print(f"  Generated {len(all_predictions)} predictions.")

    # 7. Submission Generation
    # We manually handle the submission creation to account for the debug subset.
    print("\n[Submission] Creating submission file...")

    # Load raw test metadata
    test_df = pd.read_csv(config.TEST_PATH)

    # Replicate the subsetting logic used in data.prepare_data(debug=True)
    # Logic: Select first 50 unique breaths
    test_breaths = test_df[config.BREATH_ID_COL].unique()
    debug_test_breaths = test_breaths[:50]

    # Filter and Sort
    subset_test_df = test_df[
        test_df[config.BREATH_ID_COL].isin(debug_test_breaths)
    ].copy()
    subset_test_df = subset_test_df.sort_values(
        by=[config.BREATH_ID_COL, config.ID_COL]
    )

    # Validate lengths match
    expected_len = len(subset_test_df)
    if len(all_predictions) != expected_len:
        # If the batch size dropped the last incomplete batch in test_loader (unlikely for test),
        # we might have a mismatch. In standard config, test_loader does not drop last.
        raise ValueError(
            f"Prediction count {len(all_predictions)} does not match metadata subset {expected_len}"
        )

    # Assign predictions
    subset_test_df["pressure"] = all_predictions

    # Format for submission: id, pressure
    submission_df = subset_test_df[[config.ID_COL, "pressure"]].sort_values(
        by=config.ID_COL
    )

    # Save
    output_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"  Submission saved to: {output_path}")

    # Final Verification
    saved_df = pd.read_csv(output_path)
    assert saved_df.shape[1] == 2
    assert "id" in saved_df.columns and "pressure" in saved_df.columns
    print("  Verification successful.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
