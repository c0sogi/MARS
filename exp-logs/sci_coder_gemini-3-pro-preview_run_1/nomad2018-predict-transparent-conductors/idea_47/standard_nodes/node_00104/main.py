import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import from provided libraries
from library.data import get_train_val_loaders, get_test_loader
from library.model import CEADSModel, train_one_epoch, validate

# Set seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def calculate_mcrmse(preds, targets):
    """
    Calculates Mean Column-wise Root Mean Squared Error.
    Since inputs are already log-transformed (log1p), this is equivalent to MCRMSE on original data.
    """
    # preds and targets are (N, 2) arrays
    rmse_col1 = np.sqrt(mean_squared_error(targets[:, 0], preds[:, 0]))
    rmse_col2 = np.sqrt(mean_squared_error(targets[:, 1], preds[:, 1]))
    return (rmse_col1 + rmse_col2) / 2.0


def perform_failure_analysis(model, val_loader, device):
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_global_feats = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats, batch_indices, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(device)
            batch_indices = batch_indices.to(device)
            global_feats_cuda = global_feats.to(device)

            outputs = model(atomic_feats, batch_indices, global_feats_cuda)

            all_global_feats.append(global_feats.numpy())
            all_targets.append(targets.numpy())
            all_preds.append(outputs.cpu().numpy())

    X_global = np.vstack(all_global_feats)
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    # Calculate error magnitude (Euclidean distance in target space per sample)
    # Or just mean absolute error per sample
    errors = np.mean(np.abs(y_true - y_pred), axis=1)

    # Feature names (based on library.features.extract_global_features)
    # [a, b, c, alpha, beta, gamma] (0-5)
    # [vol] (6)
    # [ar1, ar2, ar3] (7-9)
    # [ang_dist] (10)
    # [density] (11)
    # [n_atoms] (12)
    # [stoich_al, stoich_ga, stoich_in] (13-15)
    # [mass_mean, mass_std, rad_mean, rad_std, en_mean, en_std] (16-21)

    feature_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "alpha",
        "beta",
        "gamma",
        "volume",
        "ar_ab",
        "ar_bc",
        "ar_ca",
        "angular_distortion",
        "atomic_density",
        "num_atoms",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
        "mass_mean",
        "mass_std",
        "radius_mean",
        "radius_std",
        "en_mean",
        "en_std",
    ]

    # Calculate correlations
    correlations = []
    for i in range(X_global.shape[1]):
        if i < len(feature_names):
            feat_col = X_global[:, i]
            # Handle potential constant columns (std=0)
            if np.std(feat_col) > 1e-9:
                corr = np.corrcoef(feat_col, errors)[0, 1]
                correlations.append((feature_names[i], corr))
            else:
                correlations.append((feature_names[i], 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Input Features and Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name:<20}: {corr:.4f}")


def main():
    # Config
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 150  # Fast baseline
    PATIENCE = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    THRESHOLD_METRIC = 0.04819517582654953

    print(f"Running on {DEVICE}")

    # 1. Load Data
    # Use debug_size=None for full training to get best performance
    train_loader, val_loader = get_train_val_loaders(
        batch_size=BATCH_SIZE, num_workers=2, debug_size=None
    )

    # 2. Initialize Model
    model = CEADSModel(
        atomic_input_dim=21,
        global_input_dim=22,
        atomic_hidden=512,
        global_hidden=256,
        fusion_hidden=256,
        output_dim=2,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # 3. Train
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 4. Final Validation Assessment
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats, batch_indices, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(DEVICE)
            batch_indices = batch_indices.to(DEVICE)
            global_feats = global_feats.to(DEVICE)

            outputs = model(atomic_feats, batch_indices, global_feats)

            all_targets.append(targets.numpy())
            all_preds.append(outputs.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    # Calculate Metric (MCRMSE on log-transformed data)
    final_metric = calculate_mcrmse(y_pred, y_true)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(model, val_loader, DEVICE)

    # 6. Conditional Submission
    if final_metric < THRESHOLD_METRIC:
        print(f"Metric {final_metric} < {THRESHOLD_METRIC}. Generating submission...")

        test_loader = get_test_loader(batch_size=BATCH_SIZE, num_workers=2)
        results = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_feats, batch_indices, global_feats, _, ids = batch

                atomic_feats = atomic_feats.to(DEVICE)
                batch_indices = batch_indices.to(DEVICE)
                global_feats = global_feats.to(DEVICE)

                outputs = model(atomic_feats, batch_indices, global_feats)

                # Inverse transform: exp(x) - 1
                preds = torch.expm1(outputs).cpu().numpy()
                ids_np = ids.numpy()

                for i in range(len(ids_np)):
                    results.append(
                        {
                            "id": ids_np[i],
                            "formation_energy_ev_natom": preds[i, 0],
                            "bandgap_energy_ev": preds[i, 1],
                        }
                    )

        df_sub = pd.DataFrame(results)
        df_sub = df_sub[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]

        os.makedirs("./submission", exist_ok=True)
        df_sub.to_csv("./submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")
    else:
        print(f"Metric {final_metric} >= {THRESHOLD_METRIC}. Skipping submission.")


if __name__ == "__main__":
    main()
