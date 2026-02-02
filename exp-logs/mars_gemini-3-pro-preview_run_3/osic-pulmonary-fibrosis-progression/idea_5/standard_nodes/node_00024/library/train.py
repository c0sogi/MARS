import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, prepare_submission_df, LungDataset
from library.model import WideAndDeepNet, train_epoch, validate


def run_training(debug=False, epochs=Config.EPOCHS):
    """
    Orchestrates the training, validation, and submission generation process.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.
        epochs (int): Number of training epochs.

    Returns:
        float: The best validation score achieved.
    """
    seed_everything(Config.SEED)

    print(f"Starting Training Run (Debug={debug}, Epochs={epochs})")
    print(f"Device: {Config.DEVICE}")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    if debug:
        print(f"Debug Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. DataLoaders
    train_loader, val_loader = get_dataloaders(train_df, val_df)

    # 3. Model Setup
    model = WideAndDeepNet().to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(epochs):
        # Train and Validate using imported functions
        train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE)
        val_score = validate(model, val_loader, Config.DEVICE)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Score: {val_score}"
        )

        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Score! Model Saved.")

    print(f"Training Complete. Best Validation Score: {best_score}")

    # 5. Submission Generation
    print("Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    test_df = pd.read_csv(Config.TEST_META_PATH)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Expand test set to required Patient_Week rows
    sub_df = prepare_submission_df(test_df, sample_sub)

    # Dataset and Loader for submission
    sub_ds = LungDataset(sub_df, mode="submission")
    sub_loader = DataLoader(
        sub_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
    )

    results_fvc = []
    results_confidence = []

    with torch.no_grad():
        for batch in sub_loader:
            img = batch["image"].to(Config.DEVICE)
            weeks = batch["weeks"].to(Config.DEVICE)
            base_fvc = batch["baseline_fvc"].to(Config.DEVICE)
            age = batch["age"].to(Config.DEVICE)
            sex = batch["sex"].to(Config.DEVICE)
            smoke = batch["smoke"].to(Config.DEVICE)

            # Predict
            mu_z, sigma_z = model(img, weeks, base_fvc, age, sex, smoke)

            # Inverse Transform
            # Target was standardized: z = (x - mean) / std  =>  x = z * std + mean
            mu_ml = mu_z * Config.FVC_STD + Config.FVC_MEAN
            sigma_ml = sigma_z * Config.FVC_STD

            # Clip Sigma for submission (min 70 ml)
            sigma_ml = torch.clamp(sigma_ml, min=Config.MIN_CONFIDENCE)

            # Collect results
            results_fvc.extend(mu_ml.cpu().numpy().flatten())
            results_confidence.extend(sigma_ml.cpu().numpy().flatten())

    # Assign results to dataframe
    sub_df["FVC"] = results_fvc
    sub_df["Confidence"] = results_confidence

    # Format and Save
    submission = sub_df[["Patient_Week", "FVC", "Confidence"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return best_score
