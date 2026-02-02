import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import IS_RA_CGN
from library.utils import set_seed, TargetScaler


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    RMSLE = sqrt(mean((log(p + 1) - log(a + 1))^2))
    """
    # Ensure non-negative for log (clip at 0)
    y_true = torch.clamp(y_true, min=0)
    y_pred = torch.clamp(y_pred, min=0)

    log_true = torch.log1p(y_true)
    log_pred = torch.log1p(y_pred)

    # Squared differences
    squared_error = (log_pred - log_true) ** 2

    # Mean over samples for each column
    mean_squared_error = torch.mean(squared_error, dim=0)

    # Root
    rmsle_per_column = torch.sqrt(mean_squared_error)

    # Mean over columns (formation energy and bandgap)
    final_metric = torch.mean(rmsle_per_column)

    return final_metric.item()


def main():
    # 1. Setup
    start_time = time.time()
    config = Config()

    # Fast baseline adjustments
    config.num_epochs = 80  # Increased to allow convergence
    config.patience = 15

    set_seed(config.seed)
    print(f"Running on device: {config.device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=config.batch_size
    )

    # 3. Model & Optimizer
    model = IS_RA_CGN(config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    # 4. Target Scaling
    print("Fitting target scaler...")
    scaler = TargetScaler()
    all_targets = []
    for data in train_loader:
        all_targets.append(data.y)

    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)
        scaler.fit(all_targets)
    else:
        raise ValueError("No training data found.")

    # 5. Training Loop
    print(f"Starting training for {config.num_epochs} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        total_train = 0

        for data in train_loader:
            data = data.to(config.device)
            optimizer.zero_grad()

            outputs = model(data)
            targets_scaled = scaler.transform(data.y)

            loss = criterion(outputs, targets_scaled)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.num_graphs
            total_train += data.num_graphs

        avg_train_loss = train_loss / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        total_val = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(config.device)
                outputs = model(data)
                targets_scaled = scaler.transform(data.y)
                loss = criterion(outputs, targets_scaled)
                val_loss += loss.item() * data.num_graphs
                total_val += data.num_graphs

        avg_val_loss = val_loss / total_val

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    # 6. Final Validation & Metric
    print("\nPerforming Final Validation Assessment...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()

    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(config.device)
            outputs = model(data)
            preds_orig = scaler.inverse_transform(outputs)

            val_preds_list.append(preds_orig)
            val_targets_list.append(data.y)

    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    final_metric = calculate_rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate MAE per sample (average over the two targets)
    # Shape: (N, 2)
    errors = torch.abs(val_preds - val_targets).cpu().numpy()
    # Mean error per sample
    mean_errors = np.mean(errors, axis=1)

    # Load validation metadata to get features
    val_meta_df = pd.read_csv(config.val_metadata_path)

    # Ensure alignment (DataLoader with shuffle=False preserves order)
    if len(val_meta_df) != len(mean_errors):
        print("Warning: Metadata length mismatch with validation predictions.")
    else:
        # Select numeric columns for correlation
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

        # Add error to dataframe
        val_meta_df["model_error"] = mean_errors

        print("Correlation between Model Error and Features:")
        correlations = val_meta_df[feature_cols].corrwith(val_meta_df["model_error"])
        print(correlations.sort_values(key=abs, ascending=False))

    # 8. Submission
    threshold = 0.049412816762924194
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        test_ids = []
        test_preds_form = []
        test_preds_band = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(config.device)
                outputs = model(data)
                preds_orig = scaler.inverse_transform(outputs)

                # Handle IDs
                if hasattr(data, "id"):
                    # PyG batches lists if they are not tensors
                    if isinstance(data.id, list):
                        test_ids.extend(data.id)
                    else:
                        test_ids.extend(data.id.tolist())

                preds_np = preds_orig.cpu().numpy()
                test_preds_form.extend(preds_np[:, 0])
                test_preds_band.extend(preds_np[:, 1])

        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds_form,
                "bandgap_energy_ev": test_preds_band,
            }
        )

        sub_path = os.path.join(config.submission_dir, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nMetric {final_metric} >= {threshold}. Skipping submission generation."
        )

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
