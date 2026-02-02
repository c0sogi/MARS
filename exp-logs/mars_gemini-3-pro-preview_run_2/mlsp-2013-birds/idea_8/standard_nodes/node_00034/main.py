import os
import sys
import torch
import pandas as pd
import numpy as np
import glob
from scipy.stats import pearsonr
from torch.optim import AdamW

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data_loader import get_dataloaders, get_test_dataloader, make_folds
from library.modeling import get_model
from library.trainer import train_model


def main():
    # --- 1. Setup ---
    Config.setup()
    seed_everything(Config.SEED)

    # Ensure strict adherence to the strategy's hyperparameters
    Config.NUM_EPOCHS = 80
    Config.BATCH_SIZE = 16
    Config.LEARNING_RATE = 1e-3
    Config.WEIGHT_DECAY = 1e-4

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Prepare storage for OOF predictions
    # We need to store predictions for every rec_id in the development set
    # Structure: {rec_id: {'resnet18': [probs], 'densenet121': [probs], 'target': [labels]}}
    oof_data = {}

    # --- 2. Training Loop ---
    architectures = Config.MODEL_ARCHS  # ["resnet18", "densenet121"]

    for arch in architectures:
        print(f"\n{'='*20} Training Architecture: {arch} {'='*20}")

        for fold_idx in range(Config.NUM_FOLDS):
            print(f"\n--- Fold {fold_idx} / {Config.NUM_FOLDS - 1} ---")

            # Get DataLoaders
            train_loader, val_loader = get_dataloaders(fold_idx, load_cached_data=True)

            # Initialize Model
            model = get_model(arch, pretrained=True)
            model.to(device)

            # Optimizer
            optimizer = AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Train
            # Patience is set high because we want to rely on the full schedule
            # or a significant plateau, but given 80 epochs, 15 is reasonable.
            history = train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                device,
                num_epochs=Config.NUM_EPOCHS,
                patience=20,
            )

            # Save Model
            save_path = os.path.join(
                Config.WORKING_DIR, f"model_{arch}_fold_{fold_idx}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

            # Generate OOF Predictions for this fold
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    # Forward pass
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    targets_np = targets.cpu().numpy()

                    # We need rec_ids to map back.
                    # The val_loader dataset is a Subset or BirdDataset.
                    # We can iterate the dataset indices if needed, but the loader shuffles=False
                    # However, getting exact rec_ids from the loader batch is tricky without modifying the loader
                    # to return rec_ids.
                    # Fortunately, get_dataloaders creates a fresh BirdDataset for val_df.
                    # We can access the rec_ids from val_loader.dataset.df directly
                    # assuming the loader iterates sequentially (shuffle=False).

                    # Since we are iterating the loader, we need to track the global index in the dataset
                    # But batching makes this slightly complex.
                    # EASIER APPROACH: Run a separate inference loop using the dataset directly or
                    # rely on the fact that val_loader is not shuffled.
                    pass

            # Re-run inference on validation set to strictly map rec_ids
            # We construct a map of rec_id -> prediction
            val_df = val_loader.dataset.df
            # Create a temporary loader that returns (image, rec_id) like test loader?
            # No, let's just use the index. val_loader is shuffle=False.

            current_idx = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    targets_np = targets.cpu().numpy()

                    batch_size = inputs.size(0)

                    for b in range(batch_size):
                        # Get rec_id from the dataframe
                        rec_id = val_df.iloc[current_idx + b]["rec_id"]

                        if rec_id not in oof_data:
                            oof_data[rec_id] = {"targets": targets_np[b]}

                        if arch not in oof_data[rec_id]:
                            oof_data[rec_id][arch] = probs[b]
                        else:
                            # Should not happen in K-Fold, each rec_id appears once in val
                            oof_data[rec_id][arch] = probs[b]

                    current_idx += batch_size

    # --- 3. Evaluation & Failure Analysis ---
    print(f"\n{'='*20} Evaluation {'='*20}")

    # Load the specific hold-out validation set metadata
    df_val_meta = pd.read_csv(Config.VAL_CSV)
    val_rec_ids = df_val_meta["rec_id"].values

    y_true = []
    y_pred = []
    rec_ids_eval = []

    # Aggregate OOF predictions for the validation set
    for rid in val_rec_ids:
        if rid in oof_data:
            data = oof_data[rid]

            # Check if we have predictions from both architectures
            if "resnet18" in data and "densenet121" in data:
                # Heterogeneous Ensemble: Average the two
                pred_avg = (data["resnet18"] + data["densenet121"]) / 2.0

                y_true.append(data["targets"])
                y_pred.append(pred_avg)
                rec_ids_eval.append(rid)
            else:
                print(f"Warning: Missing predictions for rec_id {rid}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Metric
    final_metric = calculate_roc_auc(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate MAE per sample
    # y_true and y_pred are (N, 19)
    # MAE per sample = mean over classes of |true - pred|
    sample_mae = np.mean(np.abs(y_true - y_pred), axis=1)

    # Load segment features
    seg_feat_path = os.path.join(
        Config.INPUT_ROOT, "supplemental_data", "histogram_of_segments.txt"
    )
    if os.path.exists(seg_feat_path):
        # Read manually to handle potential header issues
        with open(seg_feat_path, "r") as f:
            lines = f.readlines()

        # Parse
        feat_data = []
        for line in lines:
            if "rec_id" in line:
                continue
            parts = line.strip().split(",")
            if len(parts) > 1:
                rid = int(parts[0])
                feats = [float(x) for x in parts[1:]]
                feat_data.append([rid] + feats)

        # Create DataFrame
        cols = ["rec_id"] + [f"feat_{i}" for i in range(len(feat_data[0]) - 1)]
        df_feats = pd.DataFrame(feat_data, columns=cols)

        # Create Error DataFrame
        df_error = pd.DataFrame({"rec_id": rec_ids_eval, "mae": sample_mae})

        # Merge
        df_analysis = df_error.merge(df_feats, on="rec_id", how="inner")

        if len(df_analysis) > 5:
            correlations = []
            for c in cols[1:]:  # Skip rec_id
                if df_analysis[c].std() > 0:  # Avoid constant columns
                    corr, _ = pearsonr(df_analysis["mae"], df_analysis[c])
                    correlations.append((c, corr))
                else:
                    correlations.append((c, 0.0))

            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print("Top 5 Features Correlated with Error (MAE):")
            for name, corr in correlations[:5]:
                print(f"  {name}: {corr:.4f}")
        else:
            print("Not enough samples for correlation analysis.")
    else:
        print("Segment features file not found.")

    # --- 4. Submission ---
    THRESHOLD = 0.8739452549958209

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_loader = get_test_dataloader()

        # We need to average predictions from all 10 models
        # (5 folds * 2 architectures)

        # Initialize accumulator for probabilities
        # We'll use a dictionary to map rec_id to accumulated probs
        test_preds_accum = {}
        test_rec_ids = []

        # To ensure order, we'll populate rec_ids from the first run
        first_run = True

        model_files = glob.glob(os.path.join(Config.WORKING_DIR, "model_*.pth"))
        print(f"Found {len(model_files)} models for ensemble.")

        for model_path in model_files:
            # Determine architecture from filename
            if "resnet18" in model_path:
                arch = "resnet18"
            elif "densenet121" in model_path:
                arch = "densenet121"
            else:
                continue

            # Load Model
            model = get_model(arch, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            print(f"Inference with {os.path.basename(model_path)}...")

            with torch.no_grad():
                for inputs, rec_ids in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu().numpy()

                    batch_rec_ids = rec_ids.numpy()

                    for i, rid in enumerate(batch_rec_ids):
                        if rid not in test_preds_accum:
                            test_preds_accum[rid] = np.zeros(Config.NUM_CLASSES)

                        test_preds_accum[rid] += probs[i]

                        if first_run:
                            test_rec_ids.append(rid)

            first_run = False

        # Create Submission Data
        submission_rows = []
        num_models = len(model_files)

        # The submission format requires flattening: Id = rec_id * 100 + species_id
        for rid in sorted(test_preds_accum.keys()):
            # Average probabilities
            avg_probs = test_preds_accum[rid] / num_models

            for species_id, prob in enumerate(avg_probs):
                row_id = int(rid * 100 + species_id)
                submission_rows.append([row_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

        # Save
        os.makedirs("submission", exist_ok=True)
        sub_path = "submission/submission.csv"
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
