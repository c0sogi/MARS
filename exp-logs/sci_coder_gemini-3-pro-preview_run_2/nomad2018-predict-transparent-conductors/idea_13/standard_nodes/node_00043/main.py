import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error

# Import from provided library
from library.config import Config
from library.utils import set_seed, TargetScaler
from library.data import get_dataloaders
from library.model import CrystalGraphResNet
from library.train import train_one_epoch, evaluate, predict


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    """
    # Clip predictions to be non-negative for log calculation
    y_pred = np.maximum(y_pred, 0)
    # Ensure y_true is non-negative (physically expected for these properties)
    y_true = np.maximum(y_true, 0)

    # Calculate RMSLE for each column (formation energy and bandgap)
    # MSLE is mean((log(p+1) - log(t+1))^2)
    msle_form = mean_squared_log_error(y_true[:, 0], y_pred[:, 0])
    msle_gap = mean_squared_log_error(y_true[:, 1], y_pred[:, 1])

    rmsle_form = np.sqrt(msle_form)
    rmsle_gap = np.sqrt(msle_gap)

    # Metric is the mean of the column-wise RMSLEs
    return (rmsle_form + rmsle_gap) / 2


def get_val_predictions(model, loader, device, scaler):
    """
    Runs inference on validation set and returns ids, predictions, and ground truth targets.
    """
    model.eval()
    ids_list = []
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Inverse transform to get original units
            outputs_orig = scaler.inverse_transform(outputs)
            targets_orig = scaler.inverse_transform(batch.y)

            ids_list.append(batch.id.cpu())
            preds_list.append(outputs_orig.cpu())
            targets_list.append(targets_orig.cpu())

    return (
        torch.cat(ids_list, dim=0).numpy().flatten(),
        torch.cat(preds_list, dim=0).numpy(),
        torch.cat(targets_list, dim=0).numpy(),
    )


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.prepare_directories()
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using 30 epochs for a fast baseline execution as requested
    NUM_EPOCHS = 30

    print("Loading data...")
    # Load cached data if available to speed up processing
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = CrystalGraphResNet(config=Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, scaler)
        val_loss = val_metrics["loss"]

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:02d}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}"
        )

    # 6. Evaluation & Metric Calculation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get predictions on validation set
    val_ids, val_preds, val_targets = get_val_predictions(
        model, val_loader, device, scaler
    )

    # Calculate Final Metric
    final_metric = calculate_rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create analysis dataframe
    analysis_df = pd.DataFrame(
        {
            "id": val_ids,
            "pred_form": val_preds[:, 0],
            "pred_gap": val_preds[:, 1],
            "target_form": val_targets[:, 0],
            "target_gap": val_targets[:, 1],
        }
    )

    # Calculate absolute errors
    analysis_df["error_form"] = np.abs(
        analysis_df["pred_form"] - analysis_df["target_form"]
    )
    analysis_df["error_gap"] = np.abs(
        analysis_df["pred_gap"] - analysis_df["target_gap"]
    )
    # Combined mean error for correlation analysis
    analysis_df["mean_error"] = (
        analysis_df["error_form"] + analysis_df["error_gap"]
    ) / 2

    # Merge with metadata features
    full_df = pd.merge(analysis_df, val_meta, on="id")

    # Numerical features to check correlation against
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

    correlations = {}
    for col in feature_cols:
        if col in full_df.columns:
            corr = full_df["mean_error"].corr(full_df[col])
            correlations[col] = corr

    # Print top correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top correlations between Mean Absolute Error and Features:")
    for name, val in sorted_corrs[:5]:
        print(f"  {name}: {val:.4f}")

    # 8. Submission Generation
    THRESHOLD = 0.05085437756413089
    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        test_ids, test_preds = predict(model, test_loader, device, scaler)

        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID to ensure correct order
        submission_df = submission_df.sort_values("id")

        # Save submission
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
