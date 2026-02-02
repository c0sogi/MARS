import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_data, get_dataloaders, get_test_loader
from library.model import DIDPCNN
from library.train import Trainer


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    logger = get_logger("runfile")

    # Optimize for speed as per "Fast Baseline" requirement
    # The dataset is small (~1600 samples), but we limit epochs to ensure < 2h runtime
    Config.NUM_EPOCHS = 25
    Config.PATIENCE = 8

    logger.info("Starting Fast Baseline Training for DIDP-CNN")

    # 2. Data Loading
    # Uses cached data if available
    data = process_data(load_cached_data=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    angle_train = data["angle_train"]

    # 3. Cross-Validation Loop
    oof_preds = np.zeros(len(y_train))
    fold_scores = []

    # Re-create split to map OOF predictions correctly to indices
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(X_train, y_train))

    for fold in range(Config.NUM_FOLDS):
        logger.info(f"--- Fold {fold} ---")

        # Get DataLoaders
        train_loader, val_loader = get_dataloaders(
            data, fold, Config.BATCH_SIZE, Config.NUM_WORKERS
        )

        # Initialize Model & Optimizer
        model = DIDPCNN().to(Config.DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Trainer
        trainer = Trainer(model, Config.DEVICE, criterion, optimizer)

        # Training
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        fold_scores.append(best_val_loss)

        # Inference on Validation Set (OOF)
        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        val_probs = []
        with torch.no_grad():
            for images, angles, _ in val_loader:
                images = images.to(Config.DEVICE)
                angles = angles.to(Config.DEVICE)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                val_probs.extend(probs)

        # Assign to OOF array
        _, val_idx = splits[fold]
        oof_preds[val_idx] = val_probs

        # Cleanup
        del model, optimizer, trainer, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Final Validation Metric
    final_metric = log_loss(y_train, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(y_train - oof_preds)

    # Compute simple features for correlation
    # Mean intensity of Band 1 and Band 2
    # X_train is (N, 3, 75, 75) -> (N, 3) mean
    img_means = X_train.mean(axis=(2, 3))
    b1_means = img_means[:, 0]
    b2_means = img_means[:, 1]

    # Handle NaNs in correlation if any (though data processing handles NaNs in angles)
    valid_mask = ~np.isnan(angle_train)

    corr_angle = pearsonr(errors[valid_mask], angle_train[valid_mask])[0]
    corr_b1 = pearsonr(errors, b1_means)[0]
    corr_b2 = pearsonr(errors, b2_means)[0]

    print("Correlation between Error Magnitude and Features:")
    print(f"  Incidence Angle: {corr_angle:.4f}")
    print(f"  Band 1 Mean: {corr_b1:.4f}")
    print(f"  Band 2 Mean: {corr_b2:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        logger.info("Metric passed threshold. Generating submission...")

        test_loader = get_test_loader(data, Config.BATCH_SIZE, Config.NUM_WORKERS)
        test_preds_accum = np.zeros(len(data["ids_test"]))

        for fold in range(Config.NUM_FOLDS):
            model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")
            model = DIDPCNN().to(Config.DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles, _ in test_loader:
                    images = images.to(Config.DEVICE)
                    angles = angles.to(Config.DEVICE)
                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            test_preds_accum += np.array(fold_preds)

            del model
            torch.cuda.empty_cache()

        avg_preds = test_preds_accum / Config.NUM_FOLDS

        # Create submission DataFrame
        sub_df = pd.DataFrame({"id": data["ids_test"], "is_iceberg": avg_preds})

        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
