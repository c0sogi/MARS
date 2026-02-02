import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_robust_auc, load_checkpoint
from library.data import load_data, SpectrogramDataset, HistogramDataset, get_transforms
from library.models import BirdCNN, BirdMLP
from library.engine import fit_model


def run():
    # 1. Setup
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Data
    # full_train_df contains both original train and val, with folds assigned
    # feature_map contains the Bag-of-Audio-Words features
    full_train_df, test_df, feature_map = load_data(load_cached_data=True)

    # Prepare OOF and Test prediction containers
    # oof_preds: Stores the ensemble prediction for each sample when it is in the validation fold
    oof_preds = np.zeros((len(full_train_df), Config.NUM_CLASSES))
    # test_preds_accumulator: Accumulates predictions from all folds to be averaged later
    test_preds_accumulator = np.zeros((len(test_df), Config.NUM_CLASSES))

    # Define the ensemble components
    # Removed MLP as it underperforms and dilutes the ensemble (Cite {solution_lesson_node_00048})
    model_types = Config.CNN_MODELS

    # 3. Training Loop per Fold
    for fold in range(Config.NUM_FOLDS):
        print(f"\n{'='*20} Processing Fold {fold}/{Config.NUM_FOLDS - 1} {'='*20}")

        # Split data for this fold
        train_sub = full_train_df[full_train_df["fold"] != fold].reset_index(drop=True)
        val_sub = full_train_df[full_train_df["fold"] == fold].reset_index(drop=True)
        val_indices = full_train_df[full_train_df["fold"] == fold].index.values

        # --- Prepare Data Loaders ---

        # CNN Loaders
        train_ds_cnn = SpectrogramDataset(
            train_sub, transforms=get_transforms("train"), mode="train"
        )
        val_ds_cnn = SpectrogramDataset(
            val_sub, transforms=get_transforms("val"), mode="val"
        )
        train_loader_cnn = DataLoader(
            train_ds_cnn,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader_cnn = DataLoader(
            val_ds_cnn,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Test Loaders (for this fold's models)
        test_ds_cnn = SpectrogramDataset(
            test_df, transforms=get_transforms("test"), mode="test"
        )
        test_loader_cnn = DataLoader(
            test_ds_cnn,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Container for this fold's ensemble predictions
        fold_val_preds = np.zeros((len(val_sub), Config.NUM_CLASSES))
        fold_test_preds = np.zeros((len(test_df), Config.NUM_CLASSES))

        models_count = 0

        # --- Train & Predict Each Model ---
        for model_name in model_types:
            print(f"\n--- Model: {model_name} ---")

            # Initialize Model & Select Loaders
            model = BirdCNN(backbone_name=model_name).to(device)
            loader_train = train_loader_cnn
            loader_val = val_loader_cnn
            loader_test = test_loader_cnn

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
            )

            # Fit Model (Handles training, early stopping, and top-k checkpointing)
            best_checkpoints = fit_model(
                model,
                loader_train,
                loader_val,
                optimizer,
                scheduler,
                device,
                fold,
                model_name,
            )

            # Inference (Snapshot Ensemble)
            # Average predictions from top-k checkpoints for this specific model
            model_val_preds = np.zeros((len(val_sub), Config.NUM_CLASSES))
            model_test_preds = np.zeros((len(test_df), Config.NUM_CLASSES))

            for score, ckpt_path in best_checkpoints:
                # Load checkpoint
                load_checkpoint(model, ckpt_path, device=device)
                model.eval()

                # Predict Val
                temp_preds = []
                with torch.no_grad():
                    for inputs, _, _ in loader_val:
                        inputs = inputs.to(device)
                        outputs = torch.sigmoid(model(inputs))
                        temp_preds.append(outputs.cpu().numpy())
                if temp_preds:
                    model_val_preds += np.concatenate(temp_preds)

                # Predict Test
                temp_preds = []
                with torch.no_grad():
                    for inputs, _, _ in loader_test:
                        inputs = inputs.to(device)
                        outputs = torch.sigmoid(model(inputs))
                        temp_preds.append(outputs.cpu().numpy())
                if temp_preds:
                    model_test_preds += np.concatenate(temp_preds)

            # Normalize by number of snapshots
            if len(best_checkpoints) > 0:
                model_val_preds /= len(best_checkpoints)
                model_test_preds /= len(best_checkpoints)

            # Add to fold ensemble
            fold_val_preds += model_val_preds
            fold_test_preds += model_test_preds
            models_count += 1

            # Cleanup to save memory
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        # Average fold ensemble across all model types
        fold_val_preds /= models_count
        fold_test_preds /= models_count

        # Store OOF predictions
        oof_preds[val_indices] = fold_val_preds

        # Accumulate Test predictions (will divide by NUM_FOLDS later)
        test_preds_accumulator += fold_test_preds

    # Average Test Predictions across folds
    final_test_preds = test_preds_accumulator / Config.NUM_FOLDS

    # 4. Validation Analysis (Hold-out)
    print(f"\n{'='*20} Validation Analysis {'='*20}")

    # Load original hold-out validation set metadata
    val_meta_df = pd.read_csv(Config.VAL_METADATA)

    # Parse ground truth labels for val_meta_df
    y_true_val = np.zeros((len(val_meta_df), Config.NUM_CLASSES))
    for idx, row in val_meta_df.iterrows():
        lbl_str = str(row["labels"])
        if lbl_str and lbl_str != "nan" and lbl_str != "?":
            try:
                indices = [int(x) for x in lbl_str.split()]
                y_true_val[idx, indices] = 1
            except ValueError:
                pass

    # Map OOF predictions to the validation set using rec_id
    rec_id_to_pred = {
        rec_id: pred for rec_id, pred in zip(full_train_df["rec_id"], oof_preds)
    }

    y_pred_val = np.zeros_like(y_true_val)

    for idx, row in val_meta_df.iterrows():
        rec_id = row["rec_id"]
        if rec_id in rec_id_to_pred:
            y_pred_val[idx] = rec_id_to_pred[rec_id]

    # Compute Metric
    final_auc = compute_robust_auc(y_true_val, y_pred_val)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print(f"\n{'='*20} Failure Analysis {'='*20}")

    # Calculate Binary Cross Entropy per sample
    eps = 1e-7
    y_pred_clipped = np.clip(y_pred_val, eps, 1 - eps)

    # Mean BCE per sample (averaged over classes)
    bce_per_sample = -(
        y_true_val * np.log(y_pred_clipped)
        + (1 - y_true_val) * np.log(1 - y_pred_clipped)
    )
    mean_bce = np.mean(bce_per_sample, axis=1)

    # Correlate with Number of Labels
    num_labels = np.sum(y_true_val, axis=1)

    if np.std(num_labels) > 0:
        correlation = np.corrcoef(mean_bce, num_labels)[0, 1]
        print(f"Correlation (Error vs Num Labels): {correlation}")
    else:
        print("Correlation (Error vs Num Labels): N/A (No variance in labels)")

    # 6. Submission
    threshold = 0.9479806884980326
    if final_auc > threshold:
        print(
            f"\nMetric ({final_auc}) > Threshold ({threshold}). Generating submission..."
        )

        submission_rows = []
        for idx, row in test_df.iterrows():
            rec_id = int(row["rec_id"])
            probs = final_test_preds[idx]

            for species_idx in range(Config.NUM_CLASSES):
                # Format: rec_id * 100 + species_number
                submission_id = rec_id * 100 + species_idx
                prob = probs[species_idx]
                submission_rows.append({"Id": submission_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({threshold}). Submission skipped.")


if __name__ == "__main__":
    run()
