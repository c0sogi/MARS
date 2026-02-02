import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_checkpoint, StandardScaler, compute_metrics
from library.data import get_dataloaders
from library.model import SRACGN
from library.train import run_training, generate_submission


def main():
    # 1. Setup Configuration and Seeds
    Config.setup()
    set_seed(Config.SEED)

    print("=" * 40)
    print("SRA-CGN Pipeline Started")
    print("=" * 40)

    # 2. Train Model
    # We limit epochs to 20 to ensure fast execution within the time limit while allowing convergence.
    # The scheduler and early stopping in library.train will handle optimization dynamics.
    print("\n[Step 1] Training Model...")
    run_training(num_epochs=20, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 3. Load Best Model for Inference
    print("\n[Step 2] Loading Best Model...")
    device = Config.DEVICE
    model = SRACGN(config=Config).to(device)
    scaler = StandardScaler()
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Load checkpoint containing model weights and scaler stats
    checkpoint = load_checkpoint(model, None, best_model_path, scaler)
    if checkpoint is None:
        print("Critical Error: Checkpoint not found. Exiting.")
        return

    model.eval()

    # 4. Validation Assessment
    print("\n[Step 3] Performing Validation Assessment...")
    _, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached=True)

    val_ids = []
    val_preds = []
    val_targets = []

    # Inference loop without gradient computation for speed
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            preds_norm = model(batch)

            # Denormalize predictions
            preds_raw = scaler.inverse_transform(preds_norm)

            # Apply physical constraint (energy cannot be negative, though formation energy can be,
            # bandgap cannot. However, for RMSLE stability, we clamp to non-negative or handle in metric.
            # The metric function handles log1p safely. Here we clamp to 0 for logic consistency with bandgap).
            # Note: Formation energy can be negative, but log metric implies positive domain or shifted.
            # Given the metric is RMSLE, targets are likely positive or shifted.
            # Looking at data analysis, min formation energy is 0.0.
            preds_raw = torch.clamp(preds_raw, min=0.0)

            # Collect IDs, predictions, and targets
            # batch.material_id is [batch_size] or [batch_size, 1]
            val_ids.extend(batch.material_id.cpu().numpy().flatten())
            val_preds.append(preds_raw.cpu().numpy())
            val_targets.append(batch.y.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Final Metric
    metrics = compute_metrics(val_preds, val_targets)
    final_metric = metrics["mean_rmsle"]

    # PRINT REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n[Step 4] Performing Failure Analysis...")
    # Calculate Mean Squared Log Error per sample as error magnitude
    log_preds = np.log1p(val_preds)
    log_targets = np.log1p(val_targets)
    # Average error across the two targets for each sample
    sample_errors = np.mean((log_preds - log_targets) ** 2, axis=1)

    # Create dataframe for analysis
    error_df = pd.DataFrame({"id": val_ids, "error_magnitude": sample_errors})

    # Load validation metadata to get features
    val_meta_df = pd.read_csv(Config.VAL_METADATA)

    # Merge errors with features
    analysis_df = pd.merge(error_df, val_meta_df, on="id", how="inner")

    # Define features to analyze
    feature_cols = [
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    print("Correlation between Error Magnitude (MSLE) and Input Features:")
    correlations = []
    for col in feature_cols:
        if col in analysis_df.columns:
            corr = analysis_df["error_magnitude"].corr(analysis_df[col])
            correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, corr in correlations:
        print(f"  {feat:<30}: {corr:.4f}")

    # 6. Submission Generation
    print("\n[Step 5] Checking Submission Criteria...")
    threshold = 0.049412816762924194

    if final_metric < threshold:
        print(f"Validation metric {final_metric} is better than threshold {threshold}.")
        print("Generating submission file...")
        generate_submission(load_cached_data=True)
    else:
        print(f"Validation metric {final_metric} did not meet threshold {threshold}.")
        print("Skipping submission generation.")

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
