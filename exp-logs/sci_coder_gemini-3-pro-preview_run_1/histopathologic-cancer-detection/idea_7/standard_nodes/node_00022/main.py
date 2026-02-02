import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# Import library modules
from library.config import Config
from library import utils, dataset, engine, network, inference


def run():
    # 1. Setup
    # Set seeds for reproducibility
    utils.set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override Config for Fast Baseline execution
    # We limit data size and epochs to ensure execution within the time limit
    Config.EPOCHS = 2
    Config.DEBUG_SAMPLE_SIZE = 2000

    print(
        f"Running Fast Baseline with EPOCHS={Config.EPOCHS}, SAMPLE_SIZE={Config.DEBUG_SAMPLE_SIZE}"
    )

    # 2. Training Loop (5 Folds)
    # Ensure folds are generated on the full dataset first
    _ = dataset.get_folds(load_cached_data=False)

    for fold in range(Config.NUM_FOLDS):
        # Get loaders (will use DEBUG_SAMPLE_SIZE to subsample)
        train_loader, val_loader, _ = dataset.get_loaders(fold, debug=True)

        # Train the model for this fold
        engine.fit(fold, train_loader, val_loader, device=device)

    # 3. Validation Assessment (Full Hold-out Validation Set)
    # Reset Config to load full data for validation and inference
    Config.DEBUG_SAMPLE_SIZE = None

    print("Starting Validation Assessment on full hold-out set...")
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    folds_df = dataset.get_folds(load_cached_data=True)

    # Map IDs to Folds to determine which model to use for OOF inference
    # We merge to get the 'fold' column for val_df entries
    val_eval_df = val_df.merge(folds_df[["id", "fold"]], on="id", how="left")

    all_preds = []
    all_targets = []

    # Lists for failure analysis
    brightness_list = []
    contrast_list = []
    error_list = []

    # Iterate over folds to perform OOF inference
    for fold in range(Config.NUM_FOLDS):
        # Select validation samples belonging to this fold
        fold_data = val_eval_df[val_eval_df["fold"] == fold].reset_index(drop=True)

        if len(fold_data) == 0:
            continue

        # Create Dataset/Loader for this fold's validation data
        ds = dataset.TumorDataset(
            fold_data, transform=dataset.get_transforms("valid"), phase="valid"
        )
        loader = DataLoader(
            ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Load the Model trained on this fold
        model = network.DenseNet121GeM(pretrained=False)
        model_path = os.path.join(
            Config.WORKING_DIR, f"{Config.MODEL_NAME}_fold{fold}_best.pth"
        )

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found. Skipping evaluation for this fold."
            )
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Inference loop
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, dtype=torch.float)
                targets = batch["target"].to(device, dtype=torch.float)

                # Predict
                logits = model(images).view(-1)
                probs = torch.sigmoid(logits)

                preds_np = probs.cpu().numpy()
                targets_np = targets.cpu().numpy()

                all_preds.extend(preds_np)
                all_targets.extend(targets_np)

                # Failure Analysis Stats: Brightness and Contrast
                # Calculate on normalized tensors (approximate but sufficient for correlation)
                # Mean across spatial dims and channels -> Brightness proxy
                b_batch = images.mean(dim=(1, 2, 3)).cpu().numpy()
                # Std across spatial dims and channels -> Contrast proxy
                c_batch = images.std(dim=(1, 2, 3)).cpu().numpy()

                brightness_list.extend(b_batch)
                contrast_list.extend(c_batch)

                # Absolute Error
                errors = np.abs(preds_np - targets_np)
                error_list.extend(errors)

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # Compute Final Metric
    if len(all_targets) == 0:
        print("Error: No validation predictions generated.")
        final_auc = 0.0
    else:
        final_auc = roc_auc_score(all_targets, all_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    if len(error_list) > 0:
        # Correlation with Brightness
        corr_b, _ = spearmanr(error_list, brightness_list)
        # Correlation with Contrast
        corr_c, _ = spearmanr(error_list, contrast_list)

        print(f"Correlation (Error vs Brightness): {corr_b:.4f}")
        print(f"Correlation (Error vs Contrast): {corr_c:.4f}")

        if abs(corr_b) > abs(corr_c):
            print(
                "Observation: Brightness has a stronger correlation with prediction error."
            )
        else:
            print(
                "Observation: Contrast has a stronger correlation with prediction error."
            )

    # 5. Submission
    # Threshold defined in task
    THRESHOLD = 0.9849192531860572

    if final_auc > THRESHOLD:
        print(f"Validation metric {final_auc} > {THRESHOLD}. Generating submission...")
        # Config.DEBUG_SAMPLE_SIZE is None, so this runs on full test set
        inference.run_inference(device=device)
    else:
        print(
            f"Validation metric {final_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
