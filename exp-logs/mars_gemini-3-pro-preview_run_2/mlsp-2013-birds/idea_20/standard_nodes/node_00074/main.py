import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_roc_auc
from library.dataset import get_data, get_dataloaders, get_test_dataloader
from library.network import BirdModel
from library.engine import train_one_epoch, validate, predict_with_tta


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Data
    # load_cached_data=True allows using pre-processed .npy files if available
    df_dev, images_dev, labels_dev, df_test, images_test = get_data(
        load_cached_data=True
    )

    # Prepare accumulators for Ensemble
    # OOF (Out-Of-Fold) predictions for the development set
    # We sum predictions from each model architecture here.
    # Since it's K-Fold, each sample is predicted exactly once per architecture.
    oof_preds_accumulator = np.zeros_like(labels_dev)

    # Test predictions accumulator
    # We sum predictions from all models and all folds here.
    test_preds_accumulator = np.zeros((len(df_test), Config.NUM_SPECIES))

    # 3. Iterative Training (Heterogeneous Ensemble)
    # Loop over defined architectures
    for model_name in Config.MODELS:
        print(f"Training Model Architecture: {model_name}")

        # Loop over folds
        for fold_idx in range(Config.N_FOLDS):
            print(f"  Fold {fold_idx}/{Config.N_FOLDS - 1}")

            # Get DataLoaders for this fold
            train_loader, val_loader = get_dataloaders(
                fold_idx, df_dev, images_dev, labels_dev, batch_size=Config.BATCH_SIZE
            )

            # Initialize Model
            model = BirdModel(
                model_name=model_name, pretrained=True, num_classes=Config.NUM_SPECIES
            )
            model = model.to(device)

            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Calculate Positive Weights for Loss (Handle Imbalance)
            # We calculate this based on the training subset of the current fold
            train_indices = df_dev[df_dev["fold"] != fold_idx].index
            train_subset_df = df_dev.iloc[train_indices]
            pos_weights = Config.get_pos_weights(train_subset_df).to(device)

            # Training Loop
            best_val_auc = 0.0
            best_model_state = None

            for epoch in range(Config.EPOCHS):
                # Train one epoch
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, device, pos_weights
                )

                # Validate
                val_loss, val_auc = validate(model, val_loader, device, pos_weights)

                # Save best model state
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict()

            # Load best weights for inference
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # --- Inference for Ensembling ---

            # 1. Validation Inference (OOF)
            # We need to predict on the validation set of this fold
            # validate() returns metrics, but we need raw probabilities for ensembling.
            # We'll do a quick pass to get them.
            model.eval()
            val_indices = df_dev[df_dev["fold"] == fold_idx].index

            # Re-use val_loader (which is not shuffled)
            fold_val_preds = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)
                    fold_val_preds.append(probs.cpu().numpy())

            fold_val_preds = np.concatenate(fold_val_preds, axis=0)

            # Accumulate OOF predictions
            # We add the predictions. Later we divide by the number of architectures.
            oof_preds_accumulator[val_indices] += fold_val_preds

            # 2. Test Inference (TTA)
            test_loader = get_test_dataloader(
                df_test, images_test, batch_size=Config.BATCH_SIZE
            )
            fold_test_preds = predict_with_tta(model, test_loader, device)

            # Accumulate Test predictions
            test_preds_accumulator += fold_test_preds

    # 4. Aggregate Results

    # Average OOF predictions across architectures
    # Each sample was predicted once per architecture (due to K-Fold).
    # So we divide by the number of architectures.
    final_oof_preds = oof_preds_accumulator / len(Config.MODELS)

    # Average Test predictions
    # Summed over (Num_Models * Num_Folds)
    total_models_trained = len(Config.MODELS) * Config.N_FOLDS
    final_test_preds = test_preds_accumulator / total_models_trained

    # 5. Final Metrics & Failure Analysis

    # Compute Final Validation Metric
    final_val_auc = compute_roc_auc(labels_dev, final_oof_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(labels_dev - final_oof_preds), axis=1)

    # Extract features for correlation
    # Feature 1: Image Pixel Mean
    pixel_means = np.mean(images_dev, axis=(1, 2, 3))  # Average over H, W, C
    # Feature 2: Image Pixel Std
    pixel_stds = np.std(images_dev, axis=(1, 2, 3))

    # Compute Correlations
    corr_mean, _ = pearsonr(mae_per_sample, pixel_means)
    corr_std, _ = pearsonr(mae_per_sample, pixel_stds)

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"  Pixel Mean: {corr_mean}")
    print(f"  Pixel Std:  {corr_std}")

    # 6. Submission Generation
    threshold = 0.9167709334579945

    if final_val_auc > threshold:
        submission_rows = []

        # Iterate through test recordings
        for idx, row in df_test.iterrows():
            rec_id = int(row["rec_id"])
            probs = final_test_preds[idx]

            # For each species, create a row
            for species_idx in range(Config.NUM_SPECIES):
                # Construct Id: rec_id * 100 + species_id
                submission_id = rec_id * 100 + species_idx
                probability = probs[species_idx]

                submission_rows.append(
                    {"Id": submission_id, "Probability": probability}
                )

        # Create DataFrame
        submission_df = pd.DataFrame(submission_rows)

        # Sort by Id to match sample submission structure (optional but good practice)
        submission_df = submission_df.sort_values(by="Id")

        # Save
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric {final_val_auc} did not meet threshold {threshold}. No submission generated."
        )


if __name__ == "__main__":
    main()
