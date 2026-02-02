import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from the provided library
from library.config import Hyperparameters
from library.utils import seed_everything, MCRMSELoss, metric_mcrmse_scored
from library.data import get_dataloaders
from library.model import RNAModel
from library.engine import fit


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Initialization
    # --------------------------------------------------------------------------
    seed_everything(Hyperparameters.SEED)
    device = Hyperparameters.DEVICE
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading datasets...")
    # load_cached_data_flag=True ensures we use preprocessed .npz files if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data_flag=True)

    # --------------------------------------------------------------------------
    # 3. Model Configuration
    # --------------------------------------------------------------------------
    print("Initializing High-Capacity Full-Rank GLU-Decoupled BiGRU model...")
    model = RNAModel().to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=Hyperparameters.LEARNING_RATE,
        weight_decay=Hyperparameters.WEIGHT_DECAY,
    )

    # Loss Function
    criterion = MCRMSELoss()

    # Scheduler: Cosine Annealing
    # We set T_max to the number of epochs we intend to run
    EPOCHS = 20  # Reduced from 50 to ensure fast baseline execution within time limits
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    model_save_path = os.path.join(Hyperparameters.MODELS_DIR, "best_model.pth")
    print(f"Starting training for {EPOCHS} epochs...")

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=EPOCHS,
        patience=Hyperparameters.EARLY_STOPPING_PATIENCE,
        model_save_path=model_save_path,
    )

    # --------------------------------------------------------------------------
    # 5. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            adjacency = batch["adjacency"].to(device)
            targets = batch["targets"]  # Keep on CPU for aggregation
            ids = batch["id"]

            outputs = model(inputs, adjacency).cpu()

            val_preds_list.append(outputs)
            val_targets_list.append(targets)
            val_ids_list.extend(ids)

    # Concatenate
    val_preds = torch.cat(val_preds_list, dim=0)  # (N, 107, 5)
    val_targets = torch.cat(val_targets_list, dim=0)  # (N, 68, 5)

    # Compute Metric
    final_metric = metric_mcrmse_scored(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample (averaged over scored columns and positions)
    # Scored columns indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = [0, 1, 3]
    seq_scored = Hyperparameters.SEQ_SCORED

    # Slice predictions to scored length
    preds_sliced = val_preds[:, :seq_scored, :]

    # Select scored columns
    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = val_targets[:, :, scored_indices]

    # Compute Squared Error
    mse_per_sample = torch.mean((preds_scored - targets_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load validation metadata for feature correlation
    val_df = pd.read_parquet(Hyperparameters.VAL_DATA_PATH)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({"id": val_ids_list, "error": rmse_per_sample})

    # Merge with metadata features
    # Note: val_ids_list comes from the loader, which should match val_df if not shuffled.
    # To be safe, we merge on 'id'.
    full_analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # Feature Engineering for Analysis
    full_analysis_df["pct_A"] = full_analysis_df["sequence"].apply(
        lambda x: x.count("A") / len(x)
    )
    full_analysis_df["pct_G"] = full_analysis_df["sequence"].apply(
        lambda x: x.count("G") / len(x)
    )
    full_analysis_df["pct_U"] = full_analysis_df["sequence"].apply(
        lambda x: x.count("U") / len(x)
    )
    full_analysis_df["pct_C"] = full_analysis_df["sequence"].apply(
        lambda x: x.count("C") / len(x)
    )

    # Compute Correlations
    corr_cols = [
        "error",
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_U",
        "pct_C",
    ]
    correlations = (
        full_analysis_df[corr_cols].corr()["error"].sort_values(ascending=False)
    )

    print("Correlations with Error Magnitude:")
    print(correlations.drop("error"))

    # --------------------------------------------------------------------------
    # 7. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_preds_list = []
        test_ids_list = []

        # Inference on Test Set
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                adjacency = batch["adjacency"].to(device)
                ids = batch["id"]

                outputs = model(inputs, adjacency).cpu()

                test_preds_list.append(outputs)
                test_ids_list.extend(ids)

        test_preds = torch.cat(test_preds_list, dim=0).numpy()  # (240, 107, 5)

        # Format submission
        # We need to flatten: 240 samples * 107 positions = 25680 rows
        submission_data = []
        target_col_names = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]

        for i, sample_id in enumerate(test_ids_list):
            pred_matrix = test_preds[i]  # Shape (107, 5)
            for seqpos in range(Hyperparameters.SEQ_LENGTH):
                # ID format: id_sequence_position
                row_id = f"{sample_id}_{seqpos}"
                row_values = pred_matrix[seqpos].tolist()
                submission_data.append([row_id] + row_values)

        submission_df = pd.DataFrame(
            submission_data, columns=["id_seqpos"] + target_col_names
        )

        # Save
        submission_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
