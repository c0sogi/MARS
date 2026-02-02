import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import prepare_folds, get_loaders, get_test_loader
from library.model import get_model
from library.trainer import run_training


def main():
    # 1. Setup
    # Increase epochs to 50 to allow Mixup to converge (Cite solution_lesson_node_00040)
    cfg = Config(epochs=50)
    seed_everything(cfg.seed)

    # Ensure working directory exists
    os.makedirs(cfg.working_dir, exist_ok=True)

    # 2. Data Preparation
    # Prepare folds with Iterative Stratification
    df_folds = prepare_folds(cfg)

    # Prepare arrays for Out-Of-Fold (OOF) predictions
    # We will accumulate predictions from both models here
    # Shape: (N_samples, N_classes)
    oof_preds_accum = np.zeros((len(df_folds), cfg.num_classes))

    # Create a mapping from rec_id to index in df_folds to store OOF preds correctly
    rec_id_to_idx = {row["rec_id"]: i for i, row in df_folds.iterrows()}

    # Extract ground truth labels for evaluation
    y_true = np.zeros((len(df_folds), cfg.num_classes))
    for idx, row in df_folds.iterrows():
        label_str = str(row["labels"])
        if label_str != "?" and label_str.lower() != "nan" and label_str.strip():
            try:
                indices = [int(x) for x in label_str.split()]
                valid_indices = [i for i in indices if 0 <= i < cfg.num_classes]
                y_true[idx, valid_indices] = 1
            except ValueError:
                pass

    # 3. Training Loop (Dual-Backbone Ensemble)
    model_names = cfg.models  # ['resnet18', 'efficientnet_b0']

    for model_name in model_names:
        print(f"\n=== Processing Architecture: {model_name} ===")

        for fold in range(cfg.n_folds):
            print(f"--- Fold {fold} ---")

            # Get DataLoaders
            train_loader, val_loader = get_loaders(fold, df_folds, cfg)

            # Initialize Model
            model = get_model(cfg, model_name)

            # Train
            # run_training handles the loop, saving best checkpoint, and returns best score
            run_training(cfg, model, train_loader, val_loader, fold, model_name)

            # Reload Best Model for OOF Inference
            checkpoint_path = os.path.join(
                cfg.working_dir, f"{model_name}_fold_{fold}_best.pth"
            )
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(cfg.device)
            model.eval()

            # OOF Inference
            fold_preds = []
            fold_rec_ids = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(cfg.device)
                    # Use autocast for consistency with training, though not strictly necessary for inference
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        probs = torch.sigmoid(outputs)

                    fold_preds.append(probs.cpu().numpy())
                    # We need rec_ids to map back to the global OOF array.
                    # The val_loader returns (images, labels).
                    # We can retrieve rec_ids from the dataset using the indices,
                    # but the loader shuffles=False, so order is preserved relative to the subset.
                    # However, to be robust, let's rely on the fact that val_loader iterates sequentially
                    # over the validation subset of the dataframe.

            fold_preds = np.concatenate(fold_preds)

            # Get the rec_ids for this fold from the dataframe
            val_df_fold = df_folds[df_folds["fold"] == fold].reset_index(drop=True)
            fold_rec_ids = val_df_fold["rec_id"].values

            # Store predictions
            for i, rec_id in enumerate(fold_rec_ids):
                global_idx = rec_id_to_idx[rec_id]
                oof_preds_accum[global_idx] += fold_preds[i]

    # 4. Validation Analysis
    # Average predictions across the ensemble (2 models)
    oof_preds_final = oof_preds_accum / len(model_names)

    # Calculate Final Metric
    final_metric = calculate_roc_auc(y_true, oof_preds_final)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error per sample (Mean Absolute Error across classes)
    # Shape: (N_samples,)
    sample_errors = np.mean(np.abs(y_true - oof_preds_final), axis=1)

    # Extract feature: Number of labels (complexity)
    # We calculate this from y_true
    num_labels = np.sum(y_true, axis=1)

    # Correlation
    if np.std(sample_errors) > 0 and np.std(num_labels) > 0:
        corr, _ = pearsonr(sample_errors, num_labels)
        print(f"Correlation between Error Magnitude and Number of Labels: {corr:.4f}")
    else:
        print("Correlation could not be computed (zero variance).")

    # 5. Submission
    threshold = 0.9072993371210134
    if final_metric > threshold:
        print(f"\nMetric {final_metric} > {threshold}. Generating submission...")

        test_loader = get_test_loader(cfg)
        test_preds_accum = None

        # Iterate over all trained models
        total_models = 0

        for model_name in model_names:
            for fold in range(cfg.n_folds):
                checkpoint_path = os.path.join(
                    cfg.working_dir, f"{model_name}_fold_{fold}_best.pth"
                )
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: Checkpoint {checkpoint_path} not found.")
                    continue

                # Load Model
                model = get_model(cfg, model_name)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(cfg.device)
                model.eval()

                # Inference
                model_preds = []
                rec_ids_list = []

                with torch.no_grad():
                    for images, rec_ids in test_loader:
                        images = images.to(cfg.device)
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            probs = torch.sigmoid(outputs)

                        model_preds.append(probs.cpu().numpy())
                        rec_ids_list.append(rec_ids.numpy())

                model_preds = np.concatenate(model_preds)
                rec_ids_all = np.concatenate(rec_ids_list)

                if test_preds_accum is None:
                    test_preds_accum = np.zeros_like(model_preds)
                    final_rec_ids = rec_ids_all  # Store rec_ids from first pass

                test_preds_accum += model_preds
                total_models += 1

        if total_models > 0:
            avg_preds = test_preds_accum / total_models

            # Format Submission
            # Id = rec_id * 100 + species_id
            submission_rows = []

            for i in range(len(final_rec_ids)):
                r_id = final_rec_ids[i]
                probs = avg_preds[i]

                for species_id, prob in enumerate(probs):
                    submission_id = int(r_id * 100 + species_id)
                    submission_rows.append({"Id": submission_id, "Probability": prob})

            submission_df = pd.DataFrame(submission_rows)
            submission_df.to_csv(cfg.submission_path, index=False)
            print(f"Submission saved to {cfg.submission_path}")
        else:
            print("Error: No models were loaded for inference.")

    else:
        print(f"\nMetric {final_metric} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
