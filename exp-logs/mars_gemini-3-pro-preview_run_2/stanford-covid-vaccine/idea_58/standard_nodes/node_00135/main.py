import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, MCRMSELoss, GlobalMetrics
from library.model import GC_SSN
from library.data import get_dataloaders


def analyze_failures(model, val_loader, val_csv_path):
    """
    Performs failure analysis by correlating model error with sample metadata.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    # 1. Calculate per-sample error
    sample_errors = []

    # Indices for scored columns in the model output
    # Model output order: [reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C]
    # Scored cols: reactivity, deg_Mg_pH10, deg_Mg_50C -> Indices 0, 1, 2
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    with torch.no_grad():
        for x, bpp, y in val_loader:
            x = x.to(Config.DEVICE)
            bpp = bpp.to(Config.DEVICE)
            y = y.to(Config.DEVICE)

            # Forward pass (Final prediction is y_final)
            y_final, _ = model(x, bpp)

            # Slice to scored length and columns
            y_pred_s = y_final[:, : Config.SCORED_LEN, scored_indices]
            y_true_s = y[:, : Config.SCORED_LEN, scored_indices]

            # Compute RMSE per sample (averaged over positions and columns)
            # Shape: (Batch, Seq, Cols) -> Mean over (1, 2) -> (Batch,)
            mse_per_sample = torch.mean((y_pred_s - y_true_s) ** 2, dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample)

            sample_errors.extend(rmse_per_sample.cpu().numpy())

    # 2. Load Metadata
    if not os.path.exists(val_csv_path):
        print(f"Validation metadata not found at {val_csv_path}. Skipping analysis.")
        return

    val_df = pd.read_csv(val_csv_path)

    # Ensure lengths match (loader might drop last if configured, though val usually doesn't)
    # library.data val_loader does not drop last.
    if len(sample_errors) != len(val_df):
        print(
            f"Warning: Number of errors ({len(sample_errors)}) does not match metadata rows ({len(val_df)}). Truncating to match."
        )
        min_len = min(len(sample_errors), len(val_df))
        sample_errors = sample_errors[:min_len]
        val_df = val_df.iloc[:min_len]

    val_df["model_error"] = sample_errors

    # 3. Compute Correlations
    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "mean_reactivity",
        "seq_length",
    ]

    print("-" * 50)
    print(f"{'Feature':<20} | {'Correlation with Error':<20}")
    print("-" * 50)

    for feat in features_to_check:
        if feat in val_df.columns:
            corr = val_df["model_error"].corr(val_df[feat])
            print(f"{feat:<20} | {corr:.4f}")
    print("-" * 50)


def generate_submission(model, test_loader, output_path):
    """
    Generates submission file for the test set.
    """
    print("\nGenerating Submission...")
    model.eval()

    all_preds = []
    ids = []

    # 1. Inference
    with torch.no_grad():
        # Retrieve IDs from dataset inside loader
        # The loader's dataset has an 'ids' attribute based on library.data implementation logic
        # We can also just iterate and assume order matches test.csv as per standard Torch Dataset behavior
        dataset_ids = (
            test_loader.dataset.features
        )  # This is the features array, not IDs.
        # Accessing IDs from the underlying data dictionary is safer if accessible,
        # but library.data.RNADataset doesn't expose IDs in __getitem__.
        # We will load test.csv to get IDs in order, assuming DataLoader preserves order (shuffle=False).

        for x, bpp in test_loader:
            x = x.to(Config.DEVICE)
            bpp = bpp.to(Config.DEVICE)

            y_final, _ = model(x, bpp)
            all_preds.append(y_final.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, Seq_Len, 5)

    # 2. Load Test Metadata for IDs
    test_df = pd.read_csv(Config.TEST_CSV)
    sample_ids = test_df["id"].values

    # 3. Format Submission
    # Model Output Order: reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C
    # Indices: 0, 1, 2, 3, 4
    # Submission Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Mapping: 0->0, 1->1, 3->2, 2->3, 4->4

    submission_rows = []

    for i, sample_id in enumerate(sample_ids):
        preds = all_preds[i]  # (107, 5)
        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"
            p = preds[pos]

            # Apply mapping
            row_vals = [
                p[0],  # reactivity
                p[1],  # deg_Mg_pH10
                p[3],  # deg_pH10
                p[2],  # deg_Mg_50C
                p[4],  # deg_50C
            ]

            submission_rows.append([row_id] + row_vals)

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    Config.EPOCHS = 15
    print(f"Running Fast Baseline with {Config.EPOCHS} epochs.")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    model = GC_SSN().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = MCRMSELoss()

    # 4. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for x, bpp, y in train_loader:
            x = x.to(Config.DEVICE)
            bpp = bpp.to(Config.DEVICE)
            y = y.to(Config.DEVICE)

            optimizer.zero_grad()

            # Two-Pass Forward
            y_final, y_aux = model(x, bpp)

            # Loss Calculation
            loss = criterion(y_final, y) + 0.5 * criterion(y_aux, y)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        global_metrics = GlobalMetrics()

        with torch.no_grad():
            for x, bpp, y in val_loader:
                x = x.to(Config.DEVICE)
                bpp = bpp.to(Config.DEVICE)
                y = y.to(Config.DEVICE)

                y_final, _ = model(x, bpp)
                global_metrics.update(y_final, y)

        val_score = global_metrics.compute()
        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print(f"Final Validation Metric: {best_score:.16f}")

    # 5. Failure Analysis
    # Reload best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    analyze_failures(model, val_loader, Config.VAL_CSV)

    # 6. Submission Logic
    THRESHOLD = 0.47142532743789534
    if best_score < THRESHOLD:
        generate_submission(model, test_loader, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation score ({best_score:.6f}) did not meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
