import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library modules
from library import config, utils, model, data, train


def main():
    # 1. Setup
    utils.seed_everything(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)

    # 2. Load Data
    # Load images (cached or processed)
    images_dict = data.load_data(load_cached_data=True)

    # Load Metadata
    # train.csv: Used for 5-Fold Cross-Validation
    # val.csv: Used for Final Hold-out Validation
    # test.csv: Used for Submission
    df_train_cv = pd.read_csv(config.TRAIN_META_PATH)
    df_val_holdout = pd.read_csv(config.VAL_META_PATH)
    df_test = pd.read_csv(config.TEST_META_PATH)

    # 3. Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    fold_models = []
    fold_scalers = []

    print(
        f"Starting Training on {len(df_train_cv)} samples with {config.N_FOLDS} folds..."
    )

    X = df_train_cv.index.values
    y = df_train_cv["is_iceberg"].values

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")

        # Split Data
        df_fold_train = df_train_cv.iloc[train_idx].copy()
        df_fold_valid = df_train_cv.iloc[valid_idx].copy()

        # Fit Scaler on Training Data of this fold ONLY
        fold_train_ids = df_fold_train["id"].values
        fold_train_images = np.stack([images_dict[uid] for uid in fold_train_ids])

        scaler = utils.FoldScaler()
        scaler.fit(fold_train_images)
        fold_scalers.append(scaler)

        # Create Datasets
        train_dataset = data.IcebergDataset(
            metadata=df_fold_train,
            images_dict=images_dict,
            scaler=scaler,
            transform=data.get_transforms("train"),
        )

        valid_dataset = data.IcebergDataset(
            metadata=df_fold_valid,
            images_dict=images_dict,
            scaler=scaler,
            transform=data.get_transforms("val"),
        )

        # Create Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        net = model.DPCNet().to(device)

        # Loss, Optimizer, Scheduler
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR,
        )

        # Early Stopping
        model_save_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold}.pth")
        early_stopping = train.EarlyStopping(
            patience=config.PATIENCE, verbose=False, path=model_save_path
        )

        # Run Training
        net, _ = train.run_training(
            model=net,
            train_loader=train_loader,
            val_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config.NUM_EPOCHS,
            early_stopping=early_stopping,
        )

        fold_models.append(net)

    # 4. Evaluation on Hold-out Validation Set (Ensemble)
    print("\nEvaluating on Hold-out Validation Set...")

    holdout_preds_accum = np.zeros((len(df_val_holdout), config.N_FOLDS))

    # Inference with each fold model using its specific scaler
    for i, (net, scaler) in enumerate(zip(fold_models, fold_scalers)):
        val_dataset = data.IcebergDataset(
            metadata=df_val_holdout,
            images_dict=images_dict,
            scaler=scaler,
            transform=data.get_transforms("val"),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        net.eval()
        preds = []
        with torch.no_grad():
            for inputs, inc_angles, _ in val_loader:
                inputs = inputs.to(device)
                inc_angles = inc_angles.to(device)
                outputs = net(inputs, inc_angles)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds.extend(probs)

        holdout_preds_accum[:, i] = np.array(preds).flatten()

    # Average Predictions
    avg_val_preds = np.mean(holdout_preds_accum, axis=1)

    # Calculate Metric
    y_true = df_val_holdout["is_iceberg"].values
    # Clip predictions to prevent log(0)
    avg_val_preds_clipped = np.clip(avg_val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_true, avg_val_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(y_true - avg_val_preds)

    # Get Inc Angles (fill NaNs with 0 for correlation calculation)
    inc_angles = (
        pd.to_numeric(df_val_holdout["inc_angle"], errors="coerce").fillna(0).values
    )

    # Get Image Stats (Band 1 Mean, Band 2 Mean) from raw images
    val_ids = df_val_holdout["id"].values
    val_images_stack = np.stack([images_dict[uid] for uid in val_ids])  # (N, 3, 75, 75)

    # Band 1 is index 0, Band 2 is index 1
    b1_means = np.mean(val_images_stack[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(val_images_stack[:, 1, :, :], axis=(1, 2))

    # Correlations
    corr_inc = np.corrcoef(errors, inc_angles)[0, 1]
    corr_b1 = np.corrcoef(errors, b1_means)[0, 1]
    corr_b2 = np.corrcoef(errors, b2_means)[0, 1]

    print(f"Correlation (Error vs Inc Angle): {corr_inc}")
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1}")
    print(f"Correlation (Error vs Band 2 Mean): {corr_b2}")

    # 6. Submission
    threshold = 0.16676861786296204
    if final_metric < threshold:
        print(f"\nMetric passed threshold ({threshold}). Generating submission...")

        test_preds_accum = np.zeros((len(df_test), config.N_FOLDS))

        for i, (net, scaler) in enumerate(zip(fold_models, fold_scalers)):
            test_dataset = data.IcebergDataset(
                metadata=df_test,
                images_dict=images_dict,
                scaler=scaler,
                transform=data.get_transforms("test"),
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )

            net.eval()
            preds = []
            with torch.no_grad():
                for inputs, inc_angles in test_loader:
                    inputs = inputs.to(device)
                    inc_angles = inc_angles.to(device)
                    outputs = net(inputs, inc_angles)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds.extend(probs)

            test_preds_accum[:, i] = np.array(preds).flatten()

        avg_test_preds = np.mean(test_preds_accum, axis=1)

        # Save to ./submission/submission.csv
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"id": df_test["id"], "is_iceberg": avg_test_preds})

        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nMetric failed threshold ({threshold}). Submission skipped.")


if __name__ == "__main__":
    main()
