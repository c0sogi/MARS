import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import library modules
import library.config as config
import library.utils as utils
import library.data as data_lib
import library.model as model_lib
import library.engine as engine_lib
import library.sam as sam_lib

# Define constants for this run
PHASE1_EPOCHS = 50  # Increased from 25 for better convergence (Cite 00095)
SWA_EPOCHS = 10
BATCH_SIZE = 32
THRESHOLD = 0.16918645240183008


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data (We use the full dataset for CV)
    print("Loading data...")
    data_container = data_lib.load_data(load_cached_data=True)

    # 3. 5-Fold Cross-Validation Ensemble
    # Cite 00023: K-Fold Ensembling > Bagging
    # Cite 00049: SWA Ensembles
    print(f"\n--- Starting {config.N_FOLDS}-Fold Cross-Validation ---")

    oof_preds = []
    oof_targets = []
    oof_angles = []

    # SWA Schedule
    swa_start = PHASE1_EPOCHS - SWA_EPOCHS
    if swa_start < 1:
        swa_start = 1

    for fold in range(config.N_FOLDS):
        print(f"\nFold {fold + 1}/{config.N_FOLDS}")

        # Get Dataloaders
        train_loader, val_loader = data_lib.get_dataloaders(
            fold=fold,
            n_folds=config.N_FOLDS,
            batch_size=BATCH_SIZE,
            mode="train_cv",
            load_cached_data=True,
        )

        # Initialize Model (Cite 00093: SAM + GeM)
        model = model_lib.IcebergResNet(
            backbone_name=config.BACKBONE,
            pretrained=config.PRETRAINED,
            dropout_rate=config.DROPOUT_RATE,
            num_classes=config.NUM_CLASSES,
            gem_p=config.GEM_P_INIT,
            gem_trainable=config.GEM_P_TRAINABLE,
        ).to(device)

        # Optimizer (SAM)
        base_optimizer = torch.optim.AdamW
        optimizer = sam_lib.SAM(
            model.parameters(),
            base_optimizer,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            rho=0.05,
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.base_optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR,
        )

        # Train
        trained_model = engine_lib.fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=PHASE1_EPOCHS,
            patience=config.EARLY_STOPPING_PATIENCE,
            use_swa=True,
            swa_start_epoch=swa_start,
            save_dir=config.CHECKPOINT_DIR,
            fold_idx=fold,
        )

        # Validation Inference (TTA)
        trained_model.eval()
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # TTA Preds
                out1 = torch.sigmoid(trained_model(images, angles))
                out2 = torch.sigmoid(trained_model(torch.flip(images, [3]), angles))
                out3 = torch.sigmoid(trained_model(torch.flip(images, [2]), angles))
                out4 = torch.sigmoid(
                    trained_model(torch.rot90(images, 2, [2, 3]), angles)
                )
                avg_preds = (out1 + out2 + out3 + out4) / 4.0

                oof_preds.extend(avg_preds.cpu().numpy().flatten())
                oof_targets.extend(labels.numpy().flatten())
                oof_angles.extend(angles.cpu().numpy().flatten())

    # 4. Global CV Assessment
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Clip for log loss stability
    eps = 1e-15
    oof_preds_clipped = np.clip(oof_preds, eps, 1 - eps)

    val_loss = log_loss(oof_targets, oof_preds_clipped)

    # REQUIRED OUTPUT FORMAT
    print(f"\nFinal Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(oof_preds - oof_targets)

    analysis_df = pd.DataFrame({"error": errors, "inc_angle": oof_angles})

    print("Correlation between Error and Features:")
    print(analysis_df.corr()["error"].drop("error"))

    # 6. Submission (Ensemble Inference)
    if val_loss < THRESHOLD:
        print("\n--- Generating Submission with 5-Fold Ensemble ---")
        test_loader = data_lib.get_dataloaders(mode="test", load_cached_data=True)

        # Initialize array to store sum of predictions
        test_preds_sum = None
        test_ids = None

        for fold in range(config.N_FOLDS):
            print(f"Predicting with model fold {fold}...")
            # Load model
            model = model_lib.IcebergResNet(
                backbone_name=config.BACKBONE,
                pretrained=config.PRETRAINED,
                dropout_rate=config.DROPOUT_RATE,
                num_classes=config.NUM_CLASSES,
                gem_p=config.GEM_P_INIT,
                gem_trainable=config.GEM_P_TRAINABLE,
            ).to(device)

            # Load checkpoint
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f"swa_model_{fold}.pth"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])

            # Predict
            preds, ids = engine_lib.predict_tta(test_loader, model, device)

            if test_preds_sum is None:
                test_preds_sum = preds
                test_ids = ids
            else:
                test_preds_sum += preds

        # Average
        avg_test_preds = test_preds_sum / config.N_FOLDS

        # Save
        sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_loss} is not better than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
