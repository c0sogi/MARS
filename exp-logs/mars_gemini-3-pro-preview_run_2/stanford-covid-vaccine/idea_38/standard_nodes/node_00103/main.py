import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import provided library modules
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import process_data, RNADataset
from library.model import DF_DCN
from library.train import run_training


def get_predictions(model, loader, device):
    """
    Runs inference on a dataloader using the DF-DCN iterative refinement strategy.
    Returns:
        preds (np.ndarray): Shape (N, L, 5)
        ids (list): List of sample IDs
    """
    model.eval()
    all_preds = []
    all_ids = []

    # For test set, loader returns (inputs, partner_indices, ids)
    # For val set, loader returns (inputs, partner_indices, targets)
    # We handle both cases.

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, partner_indices, third_item = batch
                # Check if third_item is targets (Tensor) or ids (tuple/list)
                if isinstance(third_item, torch.Tensor):
                    # It's targets, ignore for prediction
                    pass
                else:
                    all_ids.extend(third_item)
            else:
                raise ValueError("Unexpected batch structure")

            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # 1. Backbone
            z = model.forward_backbone(inputs)

            # 2. Pass 1 (Zero Feedback)
            preds_1 = model.forward_head(z, partner_indices, prev_preds=None)

            # 3. Pass 2 (With Feedback)
            preds_2 = model.forward_head(z, partner_indices, prev_preds=preds_1)

            all_preds.append(preds_2.cpu().numpy())

    return np.concatenate(all_preds, axis=0), all_ids


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure submission directory exists (though config usually handles working dir)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 2. Train Model
    # run_training handles the entire training loop and saves the best model to Config.MODEL_PATH
    print("=== Starting Training Phase ===")
    run_training()
    print("=== Training Phase Complete ===")

    # 3. Load Best Model
    print("Loading best model for validation and inference...")
    model = DF_DCN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 4. Validation & Metrics
    print("Processing validation data...")
    val_data = process_data("val", load_cached_data=True)
    val_dataset = RNADataset(val_data, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print("Running validation inference...")
    # Get predictions
    val_preds_np, _ = get_predictions(model, val_loader, device)

    # Get targets (N, 107, 5)
    val_targets_np = val_data["targets"]

    # Calculate MCRMSE on scored columns
    # Scored indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [Config.ALL_TARGETS.index(t) for t in Config.SCORED_TARGETS]

    # Slice to scored sequence length to avoid calculating metric on padded regions
    # Cite debug_lesson_8: Mask or Slice Padded Regions Before Computing Sequence Metrics
    val_preds_scored = val_preds_np[:, : Config.SEQ_SCORED, scored_indices]
    val_targets_scored = val_targets_np[:, : Config.SEQ_SCORED, scored_indices]

    # Compute MSE per element
    squared_error = (val_preds_scored - val_targets_scored) ** 2
    # Mean over batch and sequence
    mse_per_col = np.mean(squared_error, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)
    mcrmse = np.mean(rmse_per_col)

    print(f"Final Validation Metric: {mcrmse}")

    # 5. Failure Analysis
    print("=== Failure Analysis ===")
    # Calculate Mean Squared Error per sample (averaging over seq and scored cols)
    # Shape: (N,)
    sample_mse = np.mean(squared_error, axis=(1, 2))

    # Load metadata to get signal_to_noise
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Ensure alignment (dataset loader preserves order)
    if len(val_df) != len(sample_mse):
        print("Warning: Metadata length mismatch. Skipping detailed failure analysis.")
    else:
        val_df["model_mse"] = sample_mse

        # Correlation with Signal to Noise
        if "signal_to_noise" in val_df.columns:
            corr = val_df["signal_to_noise"].corr(val_df["model_mse"])
            print(f"Correlation between Error (MSE) and Signal_to_Noise: {corr:.4f}")

            # Check high error samples
            worst_samples = val_df.nlargest(5, "model_mse")
            print("\nTop 5 Worst Samples (by MSE):")
            print(worst_samples[["id", "signal_to_noise", "model_mse"]])
        else:
            print("signal_to_noise column not found in metadata.")

    # 6. Submission
    THRESHOLD = 0.47142532743789534

    if mcrmse < THRESHOLD:
        print(f"\nValidation metric {mcrmse} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_data = process_data("test", load_cached_data=True)
        test_dataset = RNADataset(test_data, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Inference
        test_preds, test_ids = get_predictions(model, test_loader, device)

        # Format Submission
        # Output format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # test_preds shape: (N_samples, 107, 5)
        # Config.ALL_TARGETS order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        submission_rows = []
        target_cols = Config.ALL_TARGETS

        print("Formatting submission file...")
        for i, sample_id in enumerate(test_ids):
            # Get preds for this sample: (107, 5)
            sample_pred = test_preds[i]

            for seqpos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos].tolist()

                row_dict = {"id_seqpos": row_id}
                for col_name, val in zip(target_cols, row_values):
                    row_dict[col_name] = val

                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)

        # Save
        save_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nValidation metric {mcrmse} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
