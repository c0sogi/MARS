import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import library modules
from library import config, dataset, model, trainer, utils


def simple_pearson(x, y):
    """
    Calculates Pearson correlation coefficient using NumPy.
    """
    if len(x) != len(y):
        return 0
    if len(x) < 2:
        return 0
    mx = np.mean(x)
    my = np.mean(y)
    xm = x - mx
    ym = y - my
    num = np.sum(xm * ym)
    den = np.sqrt(np.sum(xm**2)) * np.sqrt(np.sum(ym**2))
    if den == 0:
        return 0
    return num / den


def main():
    # 1. Configuration & Setup
    # Override config for fast baseline execution
    config.EPOCHS = 25
    config.DEBUG = False

    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    # Generate/Load folds
    df_folds = dataset.get_iterative_folds(load_cached_data=True)

    # Storage for OOF predictions
    oof_preds = []
    oof_targets = []
    oof_rec_ids = []

    # 3. Training Loop (5 Folds)
    for fold_id in range(config.NUM_FOLDS):
        print(f"\n=== Starting Fold {fold_id} ===")

        # Train
        # trainer.run_fold handles training, validation, and saving the best model
        best_auc = trainer.run_fold(fold_id, df_folds)
        print(f"Fold {fold_id} Best AUC: {best_auc}")

        # Load Best Model for Inference
        net = model.BirdResNet(pretrained=config.PRETRAINED)
        net.to(device)
        model_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold_id}.pth")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()

        # Get Validation Data for this fold
        _, val_loader = dataset.get_dataloaders(
            fold_id, df_folds, batch_size=config.BATCH_SIZE, debug=config.DEBUG
        )

        # Inference on Validation Set
        fold_preds = []
        fold_targets = []
        fold_rec_ids = []

        # We need rec_ids to map back for failure analysis
        val_rec_ids_source = val_loader.dataset.df["rec_id"].values

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)

                # Forward
                outputs = net(inputs)  # Logits
                probs = torch.sigmoid(outputs)

                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(targets.numpy())

                # Batch indices to get rec_ids
                start_idx = i * val_loader.batch_size
                end_idx = start_idx + inputs.size(0)
                fold_rec_ids.extend(val_rec_ids_source[start_idx:end_idx])

        fold_preds = np.concatenate(fold_preds, axis=0)
        fold_targets = np.concatenate(fold_targets, axis=0)

        oof_preds.append(fold_preds)
        oof_targets.append(fold_targets)
        oof_rec_ids.extend(fold_rec_ids)

    # 4. Global Validation Metric
    oof_preds = np.concatenate(oof_preds, axis=0)
    oof_targets = np.concatenate(oof_targets, axis=0)

    # Compute Macro AUC
    final_metric = utils.calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    errors = np.abs(oof_targets - oof_preds).mean(axis=1)

    # Create DataFrame for analysis
    df_error = pd.DataFrame({"rec_id": oof_rec_ids, "error": errors})

    # Load tabular features
    hist_path = os.path.join(
        config.INPUT_DIR, "supplemental_data", "histogram_of_segments.txt"
    )
    if os.path.exists(hist_path):
        try:
            # Read manually to handle potential header issues
            with open(hist_path, "r") as f:
                lines = f.readlines()

            data = []
            start_line = 0
            if "rec_id" in lines[0]:
                start_line = 1

            for line in lines[start_line:]:
                parts = line.strip().split(",")
                if len(parts) > 1:
                    rid = int(parts[0])
                    feats = [float(x) for x in parts[1:]]
                    data.append([rid] + feats)

            cols = ["rec_id"] + [f"feat_{i}" for i in range(len(data[0]) - 1)]
            df_feats = pd.DataFrame(data, columns=cols)

            # Merge
            df_analysis = df_error.merge(df_feats, on="rec_id", how="inner")

            if len(df_analysis) > 0:
                # Compute correlations
                correlations = {}
                for c in cols[1:]:  # Skip rec_id
                    if df_analysis[c].std() > 0:  # Avoid constant columns
                        corr = simple_pearson(
                            df_analysis["error"].values, df_analysis[c].values
                        )
                        correlations[c] = corr

                # Sort by absolute correlation
                sorted_corr = sorted(
                    correlations.items(), key=lambda x: abs(x[1]), reverse=True
                )

                print("Top 5 Features correlated with Error:")
                for name, val in sorted_corr[:5]:
                    print(f"{name}: {val}")
            else:
                print("No overlapping rec_ids found for failure analysis.")
        except Exception as e:
            print(f"Failed to process failure analysis features: {e}")
    else:
        print("Supplemental feature file not found.")

    # 6. Submission
    THRESHOLD = 0.8739452549958209

    if final_metric > THRESHOLD:
        print("\n=== Generating Submission ===")
        test_loader = dataset.get_test_dataloader(batch_size=config.BATCH_SIZE)

        # Ensemble Prediction
        test_preds_accum = None

        for fold_id in range(config.NUM_FOLDS):
            # Load model
            net = model.BirdResNet(pretrained=config.PRETRAINED)
            net.to(device)
            model_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold_id}.pth")
            net.load_state_dict(torch.load(model_path, map_location=device))
            net.eval()

            fold_test_preds = []

            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    outputs = net(inputs)
                    probs = torch.sigmoid(outputs)
                    fold_test_preds.append(probs.cpu().numpy())

            fold_test_preds = np.concatenate(fold_test_preds, axis=0)

            if test_preds_accum is None:
                test_preds_accum = fold_test_preds
            else:
                test_preds_accum += fold_test_preds

        # Average
        avg_preds = test_preds_accum / config.NUM_FOLDS

        # Create Submission DataFrame
        df_test = test_loader.dataset.df
        rec_ids = df_test["rec_id"].values

        # Format: Id,Probability
        # Id = rec_id * 100 + species_id
        submission_rows = []
        for i, rec_id in enumerate(rec_ids):
            probs = avg_preds[i]
            for species_id, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append([row_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])
        df_sub.to_csv(config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE}")

    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
