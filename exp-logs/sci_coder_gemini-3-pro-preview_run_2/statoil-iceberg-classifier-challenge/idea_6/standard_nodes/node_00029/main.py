import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import provided library functions and classes
from library.utils import set_seed, IcebergDataset, A2SHN
from library.data_loader import (
    load_and_process_data,
    get_kfold_loaders,
    get_test_loader,
)
from library.train_eval import train_fold


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30
    N_SPLITS = 5
    THRESHOLD = 0.18594860991006174
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # ==========================================
    # 2. Data Loading & Splitting
    # ==========================================
    print("Loading and processing data...")
    # Load all data processed from train.json and test.json
    X_full, y_full, inc_full, X_test, inc_test, test_ids = load_and_process_data(
        load_cached_data=True
    )

    # Load raw train.json to map IDs to array indices
    # This is necessary because process_data returns arrays indexed by the original file order
    with open("./input/train.json", "r") as f:
        train_json = json.load(f)
    df_raw = pd.DataFrame(train_json)
    id_to_idx = {row["id"]: i for i, row in df_raw.iterrows()}

    # Load Metadata for splitting
    df_meta_train = pd.read_csv("./metadata/train.csv")
    df_meta_val = pd.read_csv("./metadata/val.csv")

    # Get indices for the splits
    train_indices = [id_to_idx[uid] for uid in df_meta_train["id"]]
    val_indices = [id_to_idx[uid] for uid in df_meta_val["id"]]

    # Create the Training Set (for CV) and Hold-out Validation Set
    X_cv = X_full[train_indices]
    y_cv = y_full[train_indices]
    inc_cv = inc_full[train_indices]

    X_holdout = X_full[val_indices]
    y_holdout = y_full[val_indices]
    inc_holdout = inc_full[val_indices]

    print(f"Training Set (CV) shape: {X_cv.shape}")
    print(f"Hold-out Validation Set shape: {X_holdout.shape}")

    # ==========================================
    # 3. Ensemble Training (Stratified K-Fold)
    # ==========================================
    print(f"\nStarting {N_SPLITS}-Fold Ensemble Training...")
    trained_models = []

    # Get generators for K-Fold
    fold_gen = get_kfold_loaders(
        X_cv, y_cv, inc_cv, n_splits=N_SPLITS, batch_size=BATCH_SIZE, seed=SEED
    )

    for fold, (train_loader, val_loader) in enumerate(fold_gen):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        # Train model for this fold
        model = train_fold(
            train_loader, val_loader, epochs=EPOCHS, lr=2e-4, patience=8, device=DEVICE
        )
        trained_models.append(model)

    # ==========================================
    # 4. Validation Assessment
    # ==========================================
    print("\nEvaluating on Hold-out Validation Set...")
    holdout_ds = IcebergDataset(X_holdout, inc_holdout, y=None, transform=False)
    holdout_loader = DataLoader(
        holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Generate predictions using the ensemble
    all_preds = []
    for model in trained_models:
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, incs in holdout_loader:
                imgs, incs = imgs.to(DEVICE), incs.to(DEVICE)
                outputs = model(imgs, incs)
                preds.extend(outputs.cpu().numpy().flatten())
        all_preds.append(preds)

    # Average predictions across all models
    avg_preds_val = np.mean(all_preds, axis=0)

    # Compute Metric
    final_metric = log_loss(y_holdout, avg_preds_val)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    # Calculate Log Loss contribution per sample
    epsilon = 1e-15
    preds_clipped = np.clip(avg_preds_val, epsilon, 1 - epsilon)
    sample_losses = -(
        y_holdout * np.log(preds_clipped) + (1 - y_holdout) * np.log(1 - preds_clipped)
    )

    # Calculate Mean Intensity per sample (averaged over channels and spatial dims)
    # X_holdout is (N, 3, 75, 75)
    mean_intensity = np.mean(X_holdout, axis=(1, 2, 3))

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "loss": sample_losses,
            "inc_angle": inc_holdout,
            "mean_intensity": mean_intensity,
        }
    )

    # Compute Correlations
    corr_inc = df_analysis["loss"].corr(df_analysis["inc_angle"])
    corr_int = df_analysis["loss"].corr(df_analysis["mean_intensity"])

    print("Correlation between Error (Log Loss) and Features:")
    print(f"  Incidence Angle: {corr_inc:.4f}")
    print(f"  Mean Intensity:  {corr_int:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        test_loader = get_test_loader(X_test, inc_test, batch_size=64)

        test_preds_all = []
        for model in trained_models:
            model.eval()
            preds = []
            with torch.no_grad():
                for imgs, incs in test_loader:
                    imgs, incs = imgs.to(DEVICE), incs.to(DEVICE)
                    outputs = model(imgs, incs)
                    preds.extend(outputs.cpu().numpy().flatten())
            test_preds_all.append(preds)

        avg_preds_test = np.mean(test_preds_all, axis=0)

        sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds_test})
        output_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sub_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print(
            f"\nValidation metric {final_metric} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
