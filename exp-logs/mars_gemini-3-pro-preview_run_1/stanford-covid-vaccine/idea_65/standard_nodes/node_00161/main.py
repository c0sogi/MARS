import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.dataset import get_data, RNADataset
from library.train import Trainer


def run():
    # 1. Setup Environment
    # Restore full epochs for proper convergence
    Config.EPOCHS = 20
    device = Config.setup_environment()
    seed_everything(Config.SEED)

    print(f"Running on device: {device}")

    # 2. Load Data
    # Using load_cached_data=True as requested
    train_data = get_data("train", load_cached_data=True)
    val_data = get_data("val", load_cached_data=True)

    # 3. Create DataLoaders
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Train Model
    trainer = Trainer(device=device)
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, patience=5)

    # 5. Load Best Model for Evaluation
    print(f"Loading best model from {Config.MODEL_PATH}...")
    trainer.model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    trainer.model.eval()

    # 6. Validation Inference & Metric Calculation
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for seq, loop, dist, targets in val_loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass
            main_pred = trainer.model(seq, loop, dist)

            # Move to CPU and slice to scored length
            main_pred = main_pred.cpu().numpy()[:, : Config.PRED_LEN, :]
            targets = targets.cpu().numpy()[:, : Config.PRED_LEN, :]

            val_preds.append(main_pred)
            val_targets.append(targets)

    y_pred = np.concatenate(val_preds, axis=0)
    y_true = np.concatenate(val_targets, axis=0)

    # Calculate Metric
    score = metric_mcrmse(y_true, y_pred)
    # Print full precision as required
    print(f"Final Validation Metric: {score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate RMSE per sample (scalar value representing error magnitude)
    # y_true/pred shape: (N, 68, 3)
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load validation metadata to get features
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_val["error_magnitude"] = rmse_per_sample

    # Derived features
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_C",
        "len_U",
    ]
    print("Correlation between Error Magnitude and Features:")

    for feat in features_to_check:
        if feat in df_val.columns:
            corr = df_val[feat].corr(df_val["error_magnitude"])
            print(f"  {feat}: {corr:.6f}")

    # 8. Submission Generation
    THRESHOLD = 0.6176461577
    if score < THRESHOLD:
        print(
            f"\nMetric ({score}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_data = get_data("test", load_cached_data=True)
        test_dataset = RNADataset(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_preds = []
        with torch.no_grad():
            for seq, loop, dist, _ in test_loader:
                seq = seq.to(device)
                loop = loop.to(device)
                dist = dist.to(device)

                # Forward pass
                # Output shape: (Batch, 107, 3)
                pred = trainer.model(seq, loop, dist)
                test_preds.append(pred.cpu().numpy())

        all_test_preds = np.concatenate(test_preds, axis=0)  # (N_test, 107, 3)
        ids = test_data["ids"]

        # Construct Submission Rows
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Model predicts: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (2)
        submission_rows = []

        for i, sample_id in enumerate(ids):
            pred_matrix = all_test_preds[i]  # (107, 3)

            for j in range(Config.SEQ_LEN):  # 0 to 106
                row_id = f"{sample_id}_{j}"

                # Initialize all to 0.0
                val_reactivity = 0.0
                val_deg_Mg_pH10 = 0.0
                val_deg_pH10 = 0.0
                val_deg_Mg_50C = 0.0
                val_deg_50C = 0.0

                # Only fill from model if within scored length (68)
                # Positions > 67 are not scored and should be 0 (or whatever baseline),
                # but we must provide a value.
                if j < Config.PRED_LEN:
                    val_reactivity = float(pred_matrix[j, 0])
                    val_deg_Mg_pH10 = float(pred_matrix[j, 1])
                    val_deg_Mg_50C = float(pred_matrix[j, 2])
                    # deg_pH10 and deg_50C remain 0.0 as model doesn't predict them

                submission_rows.append(
                    [
                        row_id,
                        val_reactivity,
                        val_deg_Mg_pH10,
                        val_deg_pH10,
                        val_deg_Mg_50C,
                        val_deg_50C,
                    ]
                )

        # Create DataFrame and Save
        sub_df = pd.DataFrame(
            submission_rows,
            columns=[
                "id_seqpos",
                "reactivity",
                "deg_Mg_pH10",
                "deg_pH10",
                "deg_Mg_50C",
                "deg_50C",
            ],
        )

        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(sub_df)} rows.")

    else:
        print(
            f"\nMetric ({score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
