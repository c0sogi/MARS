import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import warnings
from copy import deepcopy

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_robust_auc
from library.data import get_data, BirdDataset, get_transforms
from library.models import BirdModel
from library.optimization import get_optimizer, get_scheduler
from library.engine import train_one_epoch, validate, inference_with_tta

# Try importing IterativeStratification for multi-label stratification
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False
    from sklearn.model_selection import KFold


def run():
    # 1. Setup and Initialization
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    warnings.filterwarnings("ignore")

    print("Initializing Data...")
    # Load datasets.
    # train_ds_full corresponds to metadata/train.csv (used for CV)
    # val_ds_holdout corresponds to metadata/val.csv (used for final hold-out evaluation)
    # test_ds corresponds to metadata/test.csv (used for submission)
    train_ds_full, val_ds_holdout, test_ds = get_data(load_cached_data=True)

    # Extract data arrays for manual CV splitting
    X_train_full = train_ds_full.images
    y_train_full = train_ds_full.labels

    # 2. Prepare CV Folds
    # We split the training data into N_FOLDS
    n_folds = Config.N_FOLDS
    folds_indices = []

    if HAS_SKMULTILEARN:
        print(f"Using IterativeStratification with {n_folds} folds.")
        # IterativeStratification requires 2D X, providing dummy as we only need indices based on y
        dummy_X = np.zeros((len(y_train_full), 1))
        k_fold = IterativeStratification(n_splits=n_folds, order=1)
        for train_idx, val_idx in k_fold.split(dummy_X, y_train_full):
            folds_indices.append((train_idx, val_idx))
    else:
        print(f"skmultilearn not found. Fallback to KFold with {n_folds} folds.")
        k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)
        for train_idx, val_idx in k_fold.split(X_train_full):
            folds_indices.append((train_idx, val_idx))

    # 3. Training Loop
    # Strategy: Heterogeneous Ensemble (ResNet18, EfficientNetB0, DenseNet121)
    backbones = Config.MODEL_BACKBONES
    saved_checkpoints = []

    print(f"Starting Training: {len(backbones)} Models x {n_folds} Folds")

    for backbone in backbones:
        print(f"\n=== Training Backbone: {backbone} ===")

        for fold_idx, (train_idx, val_idx) in enumerate(folds_indices):
            print(f"  Fold {fold_idx+1}/{n_folds}")

            # Create Fold Datasets
            X_fold_train = X_train_full[train_idx]
            y_fold_train = y_train_full[train_idx]
            X_fold_val = X_train_full[val_idx]
            y_fold_val = y_train_full[val_idx]

            train_dataset = BirdDataset(
                X_fold_train, y_fold_train, transforms=get_transforms("train")
            )
            val_dataset = BirdDataset(
                X_fold_val, y_fold_val, transforms=get_transforms("val")
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model, Optimizer (LLRD+Lookahead), Scheduler
            model = BirdModel(backbone, Config.NUM_CLASSES, pretrained=True)
            model.to(device)

            optimizer = get_optimizer(model)
            scheduler = get_scheduler(optimizer, Config.EPOCHS)

            # Tracking Top K Checkpoints
            # List of tuples: (auc, epoch, state_dict)
            top_k_checkpoints = []

            best_auc = 0.0
            patience_counter = 0

            for epoch in range(1, Config.EPOCHS + 1):
                train_loss = train_one_epoch(
                    model, optimizer, train_loader, device, epoch
                )
                val_loss, val_auc = validate(model, val_loader, device)

                scheduler.step()

                # Manage Top K Checkpoints
                # Save state_dict in memory temporarily
                current_state = deepcopy(model.state_dict())

                if len(top_k_checkpoints) < Config.TOP_K_CHECKPOINTS:
                    top_k_checkpoints.append((val_auc, epoch, current_state))
                    top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
                else:
                    if val_auc > top_k_checkpoints[-1][0]:
                        top_k_checkpoints.pop()  # Remove worst
                        top_k_checkpoints.append((val_auc, epoch, current_state))
                        top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)

                # Early Stopping
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= Config.PATIENCE:
                    break

            # Save Top K Checkpoints to disk
            for rank, (auc, ep, state) in enumerate(top_k_checkpoints):
                ckpt_name = f"{backbone}_fold{fold_idx}_rank{rank}.pth"
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)
                torch.save(state, ckpt_path)
                saved_checkpoints.append(ckpt_path)

            # Clean up to free memory
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

    # 4. Evaluation on Hold-out Validation Set
    print("\nEvaluating Ensemble on Hold-out Validation Set...")

    val_holdout_loader = torch.utils.data.DataLoader(
        val_ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Ensemble Inference: Average predictions from all saved checkpoints
    ensemble_preds = []

    for ckpt_path in saved_checkpoints:
        # Determine backbone from filename
        fname = os.path.basename(ckpt_path)
        if "resnet18" in fname:
            bb = "resnet18"
        elif "efficientnet_b0" in fname:
            bb = "efficientnet_b0"
        elif "densenet121" in fname:
            bb = "densenet121"
        else:
            continue

        model = BirdModel(bb, Config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        # Predict with TTA (Original + Left Shift + Right Shift)
        preds = inference_with_tta(model, val_holdout_loader, device)
        ensemble_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    if not ensemble_preds:
        print("No checkpoints found! Training failed.")
        return

    # Average predictions
    avg_preds = np.mean(ensemble_preds, axis=0)

    # Compute Metric
    y_val_true = val_ds_holdout.labels
    final_metric = compute_robust_auc(y_val_true, avg_preds)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate per-sample error: Mean Absolute Error across classes
    errors = np.abs(y_val_true - avg_preds)
    mean_errors = np.mean(errors, axis=1)

    # Extract Audio Features (Energy) for Validation Set
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    energies = []

    for _, row in val_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            data, _ = sf.read(full_path)
            # Energy = mean(amplitude^2)
            energy = np.mean(data**2)
            energies.append(energy)
        except:
            energies.append(0.0)

    energies = np.array(energies)

    # Correlation
    if len(energies) == len(mean_errors):
        corr = np.corrcoef(mean_errors, energies)[0, 1]
        print(f"Correlation between Error Magnitude and Signal Energy: {corr:.4f}")
    else:
        print("Mismatch in validation samples for analysis.")

    # 6. Submission
    threshold = 0.9479806884980326
    if final_metric > threshold:
        print("\nGenerating Submission...")

        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_ensemble_preds = []

        for ckpt_path in saved_checkpoints:
            fname = os.path.basename(ckpt_path)
            if "resnet18" in fname:
                bb = "resnet18"
            elif "efficientnet_b0" in fname:
                bb = "efficientnet_b0"
            elif "densenet121" in fname:
                bb = "densenet121"
            else:
                continue

            model = BirdModel(bb, Config.NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)

            preds = inference_with_tta(model, test_loader, device)
            test_ensemble_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        avg_test_preds = np.mean(test_ensemble_preds, axis=0)

        # Load test metadata to get rec_ids
        test_df_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        rec_ids = test_df_meta["rec_id"].values

        # Format Submission: Id = rec_id * 100 + species_id
        submission_rows = []
        for i, rec_id in enumerate(rec_ids):
            probs = avg_test_preds[i]  # shape (19,)
            for species_id, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation Metric {final_metric} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
