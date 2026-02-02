import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_auc,
    save_checkpoint,
    compute_pos_weight,
)
from library.data import get_data_with_folds, get_loaders, get_test_loader
from library.models import get_model
from library.engine import train_one_epoch, inference, save_predictions


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing Corrected Tri-Architecture Heterogeneous Ensemble...")

    # 2. Data Loading
    # Load metadata with folds
    df_folds = get_data_with_folds(load_cached_data=True)

    # Storage for OOF (Out-Of-Fold) predictions and targets
    oof_preds_list = []
    oof_targets_list = []
    oof_rec_ids_list = []

    # List to keep track of all trained model paths for final test inference
    trained_model_paths = []

    # 3. Cross-Validation Loop
    for fold in range(Config.N_FOLDS):
        print(f"\n--- Starting Fold {fold}/{Config.N_FOLDS - 1} ---")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_loaders(fold, df_folds)

        # Calculate positive weights for imbalance handling based on training data of this fold
        # Filter training data for this fold
        train_df_fold = df_folds[df_folds["fold"] != fold]
        label_cols = [c for c in df_folds.columns if c.startswith("species_")]
        train_labels = train_df_fold[label_cols].values
        pos_weight = compute_pos_weight(train_labels).to(device)

        # Store models for this fold to do ensemble validation immediately after
        fold_models = []

        # Train each backbone
        for arch in Config.BACKBONES:
            print(f"Training Backbone: {arch}")

            # Initialize Model
            model = get_model(arch, pretrained=Config.PRETRAINED).to(device)

            # Optimizer: AdamW with aggressive LR
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Loss: BCE with Logits and Pos Weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # Calculate Epochs needed to reach MAX_STEPS_PER_FOLD
            # len(train_loader) gives batches per epoch
            steps_per_epoch = len(train_loader)
            if steps_per_epoch == 0:
                # Fallback for very small debug subsets
                steps_per_epoch = 1

            epochs = int(np.ceil(Config.MAX_STEPS_PER_FOLD / steps_per_epoch))

            # Training Loop
            for epoch in range(epochs):
                # We don't print per-epoch logs to keep output clean, engine prints loss
                train_loss = train_one_epoch(
                    model, optimizer, train_loader, device, criterion
                )

            # Save Model
            save_name = f"model_{arch}_fold_{fold}.pth"
            save_path = os.path.join(Config.CACHE_DIR, save_name)
            save_checkpoint(model, save_path)

            trained_model_paths.append((arch, save_path))
            fold_models.append(model)

        # 4. Validation (Ensemble for this Fold)
        print("Validating Fold Ensemble...")

        # Prepare arrays
        fold_probs_accum = []
        fold_targets = []

        # We need to extract rec_ids for this fold to map errors later
        # Since val_loader is shuffle=False, we can take them from the dataframe directly
        val_df_fold = df_folds[df_folds["fold"] == fold].reset_index(drop=True)
        fold_rec_ids = val_df_fold["rec_id"].values

        # Inference
        # We iterate the loader once and pass inputs to all 3 models
        with torch.no_grad():
            # Set all models to eval
            for m in fold_models:
                m.eval()

            for inputs, targets in val_loader:
                inputs = inputs.to(device)

                # Get predictions from each model
                batch_preds = []
                for m in fold_models:
                    logits = m(inputs)
                    probs = torch.sigmoid(logits)
                    batch_preds.append(probs.cpu().numpy())

                # Average predictions (Ensemble)
                # batch_preds shape: [3, batch_size, num_classes]
                avg_preds = np.mean(batch_preds, axis=0)

                fold_probs_accum.append(avg_preds)
                fold_targets.append(targets.numpy())

        # Concatenate batches
        if len(fold_probs_accum) > 0:
            fold_probs_accum = np.concatenate(fold_probs_accum, axis=0)
            fold_targets = np.concatenate(fold_targets, axis=0)

            oof_preds_list.append(fold_probs_accum)
            oof_targets_list.append(fold_targets)
            oof_rec_ids_list.append(fold_rec_ids)
        else:
            print(
                f"Warning: Fold {fold} produced no validation predictions. Skipping OOF accumulation for this fold."
            )

        # Clear memory
        del fold_models, optimizer, criterion, model
        torch.cuda.empty_cache()

    # 5. Global Evaluation
    print("\n--- Global Evaluation ---")
    if len(oof_preds_list) > 0:
        oof_preds = np.concatenate(oof_preds_list, axis=0)
        oof_targets = np.concatenate(oof_targets_list, axis=0)
        oof_rec_ids = np.concatenate(oof_rec_ids_list, axis=0)

        final_auc = calculate_auc(oof_targets, oof_preds)
        print(f"Final Validation Metric: {final_auc}")

        # 6. Failure Analysis
        print("\n--- Failure Analysis ---")
        # Calculate Mean Absolute Error per sample (averaged across classes)
        # Shape: (N_samples,)
        sample_errors = np.mean(np.abs(oof_targets - oof_preds), axis=1)

        # Create DataFrame for analysis
        df_error = pd.DataFrame({"rec_id": oof_rec_ids, "error": sample_errors})

        # Load tabular features for correlation
        hist_path = os.path.join(
            Config.INPUT_ROOT, "supplemental_data", "histogram_of_segments.txt"
        )
        if os.path.exists(hist_path):
            try:
                # Read file, handling potential header issues
                with open(hist_path, "r") as f:
                    first_line = f.readline()

                skip_rows = 0
                if "rec_id" in first_line:
                    skip_rows = 1

                # Read data
                # The file format is rec_id,val1,val2...
                # We assume no header or we skipped it
                data = []
                with open(hist_path, "r") as f:
                    lines = f.readlines()[skip_rows:]
                    for line in lines:
                        parts = line.strip().split(",")
                        if len(parts) > 1:
                            rid = int(parts[0])
                            feats = [float(x) for x in parts[1:]]
                            data.append([rid] + feats)

                num_feats = len(data[0]) - 1
                feat_cols = [f"feat_{i}" for i in range(num_feats)]
                df_feats = pd.DataFrame(data, columns=["rec_id"] + feat_cols)

                # Merge
                df_analysis = df_error.merge(df_feats, on="rec_id", how="inner")

                if not df_analysis.empty:
                    # Compute correlations
                    correlations = df_analysis.drop(columns=["rec_id"]).corrwith(
                        df_analysis["error"]
                    )
                    # Drop self-correlation
                    correlations = correlations.drop("error")

                    # Top positive correlations (features associated with high error)
                    top_corr = correlations.sort_values(ascending=False).head(5)
                    print("Top Feature Correlations with Error:")
                    print(top_corr)
                else:
                    print("No matching records for failure analysis.")

            except Exception as e:
                print(f"Could not perform tabular failure analysis: {e}")
                # Fallback correlation with rec_id
                corr = df_error["rec_id"].corr(df_error["error"])
                print(f"Correlation with rec_id: {corr}")
        else:
            print("Tabular features not found. Skipping detailed feature correlation.")
    else:
        print("No OOF predictions generated. Cannot compute global metrics.")
        final_auc = 0.0

    # 7. Submission
    threshold = 0.9085501956400868
    if final_auc > threshold:
        print(
            f"\nMetric ({final_auc}) > Threshold ({threshold}). Generating Submission..."
        )

        test_loader = get_test_loader()
        all_test_probs = []
        test_rec_ids = None

        # Iterate over all trained models
        for arch, path in trained_model_paths:
            # Load model
            model = get_model(
                arch, pretrained=False
            )  # Pretrained=False because we load state_dict
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)

            # Inference
            rec_ids, probs = inference(model, test_loader, device)

            if test_rec_ids is None:
                test_rec_ids = rec_ids

            all_test_probs.append(probs)

            del model
            torch.cuda.empty_cache()

        # Average predictions
        avg_test_probs = np.mean(all_test_probs, axis=0)

        # Save
        save_predictions(test_rec_ids, avg_test_probs, Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({threshold}). Submission skipped.")


if __name__ == "__main__":
    main()
