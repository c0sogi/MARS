import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library modules
import library.config as cfg
import library.utils as utils
import library.data as data
import library.model as model
import library.engine as engine


from sklearn.model_selection import StratifiedKFold


def main():
    # 1. Setup and Reproducibility
    utils.seed_everything(cfg.SEED)
    cfg.EPOCHS = 12  # Slightly increased epochs for CV folds
    device = cfg.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading (Full Train/Val Split)
    print("Loading data...")
    # We load the base dataframes
    full_train_df, holdout_val_df, test_df = data.get_dataframes(load_cached_data=True)

    # We will perform 5-Fold CV on the 'full_train_df'
    # And report final performance on 'holdout_val_df' using the ensemble

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)

    # Placeholders for ensemble predictions
    val_preds_ensemble = np.zeros(len(holdout_val_df))
    test_preds_ensemble = np.zeros(len(test_df))

    # We need a loader for the holdout set and test set that is consistent
    _, val_loader_holdout, test_loader = data.get_loaders(
        train_df=full_train_df,  # Dummy
        val_df=holdout_val_df,
        test_df=test_df,
        load_cached_data=False,
    )

    model_paths = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["diagnosis"])
    ):
        print(f"\n=== Training Fold {fold + 1}/{cfg.NUM_FOLDS} ===")

        # Create Fold DataFrames
        fold_train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

        # Create Loaders
        train_loader, val_loader, _ = data.get_loaders(
            train_df=fold_train_df,
            val_df=fold_val_df,
            test_df=test_df,  # Dummy
            load_cached_data=False,
        )

        # Initialize Model
        net = model.RetinopathyModel(pretrained=True)
        net = net.to(device)

        optimizer = optim.Adam(
            net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR
        )

        # Save path for this fold
        fold_save_path = os.path.join(cfg.WORKING_DIR, f"model_fold_{fold}.pth")
        model_paths.append(fold_save_path)

        # Train
        engine.run_training(
            model=net,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=cfg.EPOCHS,
            patience=cfg.PATIENCE,
            save_path=fold_save_path,
        )

        # Load best model for inference
        net.load_state_dict(torch.load(fold_save_path, map_location=device))
        net.eval()

        # Inference on Holdout Validation Set
        print(f"Predicting on Holdout Set with Fold {fold + 1}...")
        fold_val_preds = []
        with torch.no_grad():
            for images, _ in val_loader_holdout:
                images = images.to(device)
                outputs = net(images)
                fold_val_preds.extend(outputs.detach().cpu().numpy().flatten().tolist())

        val_preds_ensemble += np.array(fold_val_preds) / cfg.NUM_FOLDS

        # Inference on Test Set
        print(f"Predicting on Test Set with Fold {fold + 1}...")
        fold_test_preds = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = net(images)
                fold_test_preds.extend(
                    outputs.detach().cpu().numpy().flatten().tolist()
                )

        test_preds_ensemble += np.array(fold_test_preds) / cfg.NUM_FOLDS

        # Clean up
        del net, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Final Evaluation on Holdout Set
    print("\nRunning final validation analysis on Ensemble...")
    val_targets = holdout_val_df["diagnosis"].values

    # Compute Final Metric
    final_metric = utils.compute_score(val_targets, val_preds_ensemble)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis (using Ensemble predictions)
    # We need image stats. We can compute them once using the val_loader_holdout
    val_img_means = []
    val_img_stds = []

    # Re-iterate val_loader just to get stats (fast)
    for images, _ in val_loader_holdout:
        batch_means = images.mean(dim=(2, 3)).mean(dim=1).numpy()
        batch_stds = images.std(dim=(2, 3)).mean(dim=1).numpy()
        val_img_means.extend(batch_means)
        val_img_stds.extend(batch_stds)

    y_pred_rounded = np.round(np.clip(val_preds_ensemble, 0, 4)).astype(int)
    errors = np.abs(val_targets - y_pred_rounded)

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": val_img_means,
            "std_intensity": val_img_stds,
        }
    )

    corr_std = df_analysis["error"].corr(
        df_analysis["std_intensity"], method="spearman"
    )
    print("\n=== Failure Analysis ===")
    print(f"Correlation (Error vs Input Std Intensity): {corr_std}")

    # 6. Submission
    THRESHOLD = 0.9164260977558991  # Previous best

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold. Generating submission..."
        )

        # Process Test Predictions
        test_ids = test_df["id_code"].values
        final_test_preds = np.round(np.clip(test_preds_ensemble, 0, 4)).astype(int)

        submission_df = pd.DataFrame(
            {"id_code": test_ids, "diagnosis": final_test_preds}
        )
        os.makedirs(os.path.dirname(cfg.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(cfg.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {cfg.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold. Submission skipped."
        )


if __name__ == "__main__":
    main()
