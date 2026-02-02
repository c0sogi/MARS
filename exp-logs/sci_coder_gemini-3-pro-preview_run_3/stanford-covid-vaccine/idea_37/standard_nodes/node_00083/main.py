import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.train import train_model
from library.model import DeepDecoupledModel
from library.utils import set_seed, MCRMSE
from library.data import RNADataset, load_or_process_data

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_inference(model, loader, device):
    """
    Runs inference on a dataloader and returns predictions, targets, and IDs.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for inputs, pair_indices, pair_mask, targets, ids in loader:
            inputs = inputs.to(device)
            pair_indices = pair_indices.to(device)
            pair_mask = pair_mask.to(device)

            # Forward pass
            preds = model(inputs, pair_indices, pair_mask)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0), all_ids


def main():
    # 1. Configuration and Setup
    config = Config()

    # Fast baseline settings
    config.num_epochs = 5  # Limit epochs for speed

    # Ensure reproducibility
    set_seed(config.seed)
    device = torch.device(config.device)

    # 2. Train the Model
    # train_model handles the training loop, validation, and saving the best model.
    print("Starting training...")
    _ = train_model(config)
    print("Training finished.")

    # 3. Load the Best Model
    model = DeepDecoupledModel().to(device)
    model_path = config.model_save_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Validation and Metric Calculation
    print("Running validation inference...")
    val_inputs, val_pair_indices, val_pair_mask, val_targets, val_ids = (
        load_or_process_data("val", config.val_file, config, load_cached_data=True)
    )

    val_dataset = RNADataset(
        val_inputs, val_pair_indices, val_pair_mask, val_targets, val_ids
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Get predictions
    val_preds, val_trues, val_ids_out = run_inference(model, val_loader, device)

    # Calculate Final Validation Metric (MCRMSE on scored columns)
    criterion = MCRMSE()
    # Move to device for calculation to match criterion expectations if any
    val_metric = criterion(
        val_preds.to(device), val_trues.to(device), mode="val"
    ).item()

    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")

    # Load metadata for feature extraction
    val_df = pd.read_parquet(config.val_file)
    val_df.set_index("id", inplace=True)

    # Calculate error per sample
    # Slice predictions to scored length (68)
    val_preds_sliced = val_preds[:, : config.pred_len, :]
    val_trues_sliced = val_trues  # val_targets is already (N, 68, 5)

    # Filter for scored columns [0, 1, 3]
    scored_indices = config.scored_indices
    val_preds_scored = val_preds_sliced[:, :, scored_indices]
    val_trues_scored = val_trues_sliced[:, :, scored_indices]

    # MSE per sample (averaged over seq_len and scored_cols)
    mse_per_sample = torch.mean((val_preds_scored - val_trues_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"id": val_ids_out, "error": rmse_per_sample})

    # Merge with metadata features
    # We need signal_to_noise and structure info
    analysis_df = analysis_df.merge(
        val_df[["signal_to_noise", "sequence", "structure", "SN_filter"]],
        on="id",
        how="left",
    )

    # Extract additional features
    analysis_df["pct_paired"] = analysis_df["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )
    analysis_df["pct_A"] = analysis_df["sequence"].apply(
        lambda s: s.count("A") / len(s)
    )
    analysis_df["pct_G"] = analysis_df["sequence"].apply(
        lambda s: s.count("G") / len(s)
    )
    analysis_df["pct_U"] = analysis_df["sequence"].apply(
        lambda s: s.count("U") / len(s)
    )
    analysis_df["pct_C"] = analysis_df["sequence"].apply(
        lambda s: s.count("C") / len(s)
    )

    # Compute correlations
    features_to_corr = [
        "signal_to_noise",
        "SN_filter",
        "pct_paired",
        "pct_A",
        "pct_G",
        "pct_U",
        "pct_C",
    ]
    correlations = analysis_df[features_to_corr].corrwith(analysis_df["error"])

    print("Correlation between Error and Features:")
    print(correlations)

    # 6. Submission Generation
    THRESHOLD = 0.5978901386

    if val_metric < THRESHOLD:
        print(
            f"Validation metric ({val_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test data
        test_inputs, test_pair_indices, test_pair_mask, test_targets, test_ids = (
            load_or_process_data(
                "test", config.test_file, config, load_cached_data=True
            )
        )

        test_dataset = RNADataset(
            test_inputs, test_pair_indices, test_pair_mask, test_targets, test_ids
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        # Inference
        test_preds, _, test_ids_out = run_inference(model, test_loader, device)
        test_preds = test_preds.numpy()  # (N, 107, 5)

        # Format submission
        # We need to flatten: 240 samples * 107 positions
        # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        submission_data = []
        target_cols = (
            config.target_cols
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids_out):
            sample_preds = test_preds[i]  # (107, 5)
            for seqpos in range(config.seq_len):
                row_id = f"{sample_id}_{seqpos}"
                preds_values = sample_preds[seqpos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = preds_values[col_idx]

                submission_data.append(row_dict)

        submission_df = pd.DataFrame(submission_data)

        # Save submission
        # Ensure directory exists (though working/idea_37 should exist from config init)
        submission_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation metric ({val_metric}) is not below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
