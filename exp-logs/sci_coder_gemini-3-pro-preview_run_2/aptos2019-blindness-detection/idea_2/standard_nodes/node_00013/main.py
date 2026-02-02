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


def main():
    # 1. Setup and Reproducibility
    utils.seed_everything(cfg.SEED)

    # Configuration Overrides
    cfg.EPOCHS = 10
    device = cfg.DEVICE
    print(f"Running on device: {device}")

    # Containers for OOF (Out-Of-Fold) predictions and targets
    oof_preds = []
    oof_targets = []
    oof_img_means = []
    oof_img_stds = []

    # Store model paths for ensemble inference
    model_paths = []

    # 5-Fold Cross Validation Loop
    for fold in range(cfg.NUM_FOLDS):
        print(f"\n=== Starting Fold {fold + 1}/{cfg.NUM_FOLDS} ===")

        # 2. Data Loading for specific fold
        train_loader, val_loader, test_loader = data.get_loaders(
            fold=fold, load_cached_data=True
        )

        # 3. Model Initialization
        net = model.RetinopathyModel(pretrained=True)
        net = net.to(device)

        # 4. Optimizer and Scheduler
        optimizer = optim.Adam(
            net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR
        )

        # 5. Training
        fold_save_path = os.path.join(cfg.WORKING_DIR, f"model_fold_{fold}.pth")
        model_paths.append(fold_save_path)

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

        # 6. Inference on Validation Set (OOF)
        print(f"Validating Fold {fold}...")
        net.load_state_dict(torch.load(fold_save_path, map_location=device))
        net.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                # Forward pass
                outputs = net(images)

                # Collect results
                oof_targets.extend(labels.cpu().numpy().tolist())
                oof_preds.extend(outputs.detach().cpu().numpy().flatten().tolist())

                # Stats for failure analysis
                batch_means = images.mean(dim=(2, 3)).mean(dim=1).cpu().numpy()
                batch_stds = images.std(dim=(2, 3)).mean(dim=1).cpu().numpy()
                oof_img_means.extend(batch_means)
                oof_img_stds.extend(batch_stds)

    # 7. Overall Evaluation
    print("\n=== Overall CV Evaluation ===")
    final_metric = utils.compute_score(oof_targets, oof_preds)
    print(f"Final Validation Metric (OOF QWK): {final_metric}")

    # Failure Analysis on OOF
    y_true = np.array(oof_targets)
    y_pred_raw = np.array(oof_preds)
    y_pred_rounded = np.round(np.clip(y_pred_raw, 0, 4)).astype(int)
    errors = np.abs(y_true - y_pred_rounded)

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": oof_img_means,
            "std_intensity": oof_img_stds,
        }
    )

    corr_mean = df_analysis["error"].corr(
        df_analysis["mean_intensity"], method="spearman"
    )
    corr_std = df_analysis["error"].corr(
        df_analysis["std_intensity"], method="spearman"
    )

    print("\n=== Failure Analysis ===")
    print(f"Correlation (Error vs Input Mean Intensity): {corr_mean}")
    print(f"Correlation (Error vs Input Std Intensity): {corr_std}")

    # 8. Ensemble Submission
    THRESHOLD = 0.9207435978935975

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating ensemble submission..."
        )

        # Load test loader (same for all folds)
        _, _, test_loader = data.get_loaders(fold=0, load_cached_data=True)

        # Aggregate predictions
        ensemble_preds = np.zeros(len(test_loader.dataset))
        test_ids = []

        for i, path in enumerate(model_paths):
            print(f"Predicting with model fold {i}...")
            net = model.RetinopathyModel(pretrained=False)
            net.load_state_dict(torch.load(path, map_location=device))
            net = net.to(device)

            # Use predict_fn from engine
            ids, preds = engine.predict_fn(net, test_loader, device)
            ensemble_preds += preds

            if i == 0:
                test_ids = ids

        # Average predictions (Cite solution_lesson_node_00012)
        ensemble_preds /= cfg.NUM_FOLDS

        # Post-process
        ensemble_preds = np.clip(ensemble_preds, 0, 4)
        ensemble_preds = np.round(ensemble_preds).astype(int)

        # Save
        submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": ensemble_preds})
        os.makedirs(os.path.dirname(cfg.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(cfg.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {cfg.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
