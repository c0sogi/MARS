import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_roc_auc
from library.data import get_dataloaders, get_test_loader, get_folds
from library.models import get_model
from library.engine import train_loop, validate, predict


def run():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Training & OOF Generation
    # We need to store OOF predictions for every sample in the dev set (train+val)
    # to later extract the specific subset for 'metadata/val.csv'

    # Load folds to get all rec_ids
    folds_df = get_folds(load_cached_data=True)
    all_rec_ids = folds_df["rec_id"].values

    # Map rec_id to index for OOF storage
    id_to_idx = {rid: i for i, rid in enumerate(all_rec_ids)}
    num_samples = len(all_rec_ids)
    num_classes = Config.NUM_CLASSES

    # Accumulator for ensemble probabilities
    # We will sum probabilities from all valid models for each sample
    oof_probs_sum = np.zeros((num_samples, num_classes))
    oof_counts = np.zeros((num_samples, 1))

    # Extract Ground Truth for Metric Computation
    y_true = np.zeros((num_samples, num_classes))
    for idx, row in folds_df.iterrows():
        rid = row["rec_id"]
        idx_map = id_to_idx[rid]
        lbl_str = str(row["labels"])
        if lbl_str and lbl_str != "?" and lbl_str.lower() != "nan":
            try:
                indices = [int(x) for x in lbl_str.split()]
                y_true[idx_map, indices] = 1.0
            except:
                pass

    # Define Strategy Components
    sources = [s["name"] for s in Config.DATA_SOURCES]
    archs = Config.ARCHITECTURES

    # Store model info for final test inference
    test_models = []

    print("Starting Dual-Stream Heterogeneous Ensemble Training...")

    for source in sources:
        for arch in archs:
            print(f"\n=== Training Stream: {source} | Architecture: {arch} ===")

            # Iterate through 5 Folds
            for fold in range(Config.N_FOLDS):
                print(f"  Fold {fold}/{Config.N_FOLDS - 1}")

                # Get DataLoaders
                train_loader, val_loader = get_dataloaders(
                    fold_idx=fold, data_source=source, batch_size=Config.BATCH_SIZE
                )

                # Setup Model
                model = get_model(
                    arch, pretrained=True, num_classes=num_classes, device=device
                )

                # Setup Optimizer & Scheduler
                optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=Config.EPOCHS, eta_min=1e-6
                )

                # Checkpoint Path
                ckpt_name = f"{arch}_{source}_fold_{fold}.pth"
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)

                # Train (handles EMA and Early Stopping)
                train_loop(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_epochs=Config.EPOCHS,
                    save_path=ckpt_path,
                )

                # Load Best Model (EMA weights) for OOF Prediction
                best_model = get_model(
                    arch, pretrained=False, num_classes=num_classes, device=device
                )
                best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
                best_model.eval()

                # OOF Inference on Validation Fold
                # Note: val_loader contains the hold-out fold for this iteration
                preds, ids = predict(best_model, val_loader, device)

                # Accumulate OOF Predictions
                # Each sample in the dev set appears in exactly one fold's val_loader per (Source, Arch)
                for i, rid in enumerate(ids):
                    if rid in id_to_idx:
                        idx_map = id_to_idx[rid]
                        oof_probs_sum[idx_map] += preds[i]
                        oof_counts[idx_map] += 1

                # Store model info for Test Inference
                test_models.append({"source": source, "arch": arch, "path": ckpt_path})

                # Cleanup
                del model, best_model, optimizer, scheduler, train_loader, val_loader
                torch.cuda.empty_cache()

    # 3. Validation Analysis
    # Compute averaged predictions
    # oof_counts should be 6 (2 sources * 3 archs) for every sample
    oof_counts[oof_counts == 0] = 1  # Safety
    oof_preds = oof_probs_sum / oof_counts

    # Identify the specific hold-out validation set from metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    val_meta_df = pd.read_csv(val_meta_path)
    val_ids = val_meta_df["rec_id"].values

    # Extract predictions and targets for the validation subset
    val_indices = [id_to_idx[rid] for rid in val_ids if rid in id_to_idx]

    y_val_true = y_true[val_indices]
    y_val_pred = oof_preds[val_indices]

    # Compute Final Metric
    val_auc = compute_roc_auc(y_val_true, y_val_pred)
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample Mean Absolute Error
    abs_err = np.abs(y_val_true - y_val_pred)
    mean_abs_err = np.mean(abs_err, axis=1)

    # Correlate error with Label Count (proxy for complexity)
    num_labels = []
    for _, row in val_meta_df.iterrows():
        lbl_str = str(row["labels"])
        if lbl_str and lbl_str != "?" and lbl_str.lower() != "nan":
            cnt = len(lbl_str.split())
        else:
            cnt = 0
        num_labels.append(cnt)

    num_labels = np.array(num_labels)

    if len(mean_abs_err) > 1 and np.std(num_labels) > 0:
        correlation = np.corrcoef(mean_abs_err, num_labels)[0, 1]
        print(f"Correlation between Error and Label Count: {correlation}")
    else:
        print("Correlation between Error and Label Count: N/A")

    # 5. Submission
    threshold = 0.9479806884980326
    if val_auc > threshold:
        print("\nValidation metric meets threshold. Generating submission...")

        # Load Test Metadata
        test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        test_df = pd.read_csv(test_meta_path)
        test_ids = test_df["rec_id"].values
        n_test = len(test_df)

        # Accumulator for Test Predictions
        test_probs_sum = np.zeros((n_test, num_classes))
        test_counts = 0
        test_id_to_idx = {rid: i for i, rid in enumerate(test_ids)}

        # Group models by source to optimize data loading
        models_by_source = {}
        for m in test_models:
            s = m["source"]
            if s not in models_by_source:
                models_by_source[s] = []
            models_by_source[s].append(m)

        # Run Inference
        for source, models in models_by_source.items():
            test_loader = get_test_loader(
                data_source=source, batch_size=Config.BATCH_SIZE
            )

            for m_info in models:
                arch = m_info["arch"]
                ckpt_path = m_info["path"]

                # Load Model
                model = get_model(
                    arch, pretrained=False, num_classes=num_classes, device=device
                )
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()

                # Predict
                preds, ids = predict(model, test_loader, device)

                # Accumulate
                for i, rid in enumerate(ids):
                    if rid in test_id_to_idx:
                        idx = test_id_to_idx[rid]
                        test_probs_sum[idx] += preds[i]

                test_counts += 1

                del model
                torch.cuda.empty_cache()

        # Average Predictions
        if test_counts > 0:
            avg_test_preds = test_probs_sum / test_counts
        else:
            avg_test_preds = test_probs_sum  # Should not happen

        # Format Submission
        submission_rows = []
        for i, rid in enumerate(test_ids):
            probs = avg_test_preds[i]
            for species_id in range(num_classes):
                # Id format: rec_id * 100 + species_id
                row_id = int(rid * 100 + species_id)
                prob = probs[species_id]
                submission_rows.append({"Id": row_id, "Probability": prob})

        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
