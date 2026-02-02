import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.utils import load_data, seed_everything
from library.data import IcebergDataset
from library.model import DPCNet
from library import engine


def main():
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    THRESHOLD = 0.1703794832369958
    BATCH_SIZE = 32
    EPOCHS = 60  # Increased to allow convergence with higher dropout
    N_FOLDS = 5

    # 2. Load Data
    # load_data returns a dictionary with keys:
    # 'X_train', 'y_train', 'meta_train' (from metadata/train.csv)
    # 'X_val', 'y_val', 'meta_val' (from metadata/val.csv - HOLD OUT)
    # 'X_test', 'meta_test', 'test_ids'
    print("Loading data...")
    data_dict = load_data(load_cached_data=True)

    X_train_full = data_dict["X_train"]
    y_train_full = data_dict["y_train"]
    meta_train_full = data_dict["meta_train"]

    X_holdout = data_dict["X_val"]
    y_holdout = data_dict["y_val"]
    meta_holdout = data_dict["meta_val"]

    # 3. Train with 5-Fold CV on X_train only (preserving X_val as hold-out)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    model_paths = []

    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation on Training Set...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        # Split internal train/val for CV
        X_tr, X_v = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_v = y_train_full[train_idx], y_train_full[val_idx]
        m_tr, m_v = meta_train_full[train_idx], meta_train_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, m_tr, y_tr, transform=True)
        val_ds = IcebergDataset(X_v, m_v, y_v, transform=False)

        # Create Loaders
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Train Fold
        # engine.train_fold returns (save_path, best_val_loss)
        save_path, _ = engine.train_fold(
            fold_idx=fold,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=EPOCHS,
            output_dir="./working/models",
        )
        model_paths.append(save_path)

    # 4. Validation on Hold-out Set
    print("\nPerforming validation on Hold-out Set...")
    holdout_ds = IcebergDataset(X_holdout, meta_holdout, y_holdout, transform=False)
    holdout_loader = DataLoader(
        holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Ensemble Inference
    holdout_preds_accum = np.zeros((len(y_holdout), 1))

    for path in model_paths:
        model = DPCNet().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for inputs, _ in holdout_loader:
                img, angle = inputs
                img = img.to(device)
                angle = angle.to(device)

                outputs = model((img, angle))
                probs = torch.sigmoid(outputs)
                fold_preds.extend(probs.cpu().numpy())

        holdout_preds_accum += np.array(fold_preds)

    avg_holdout_preds = holdout_preds_accum / N_FOLDS

    # 5. Metric
    # Flatten arrays for log_loss
    y_true = y_holdout
    y_pred = avg_holdout_preds.flatten()

    # Clip predictions to avoid log(0) errors
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    val_metric = log_loss(y_true, y_pred)
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(y_true - y_pred)

    # Extract simple features from X_holdout for correlation
    # X_holdout is (N, 75, 75, 3). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_means = np.mean(X_holdout[:, :, :, 0], axis=(1, 2))
    b2_means = np.mean(X_holdout[:, :, :, 1], axis=(1, 2))

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": meta_holdout,
            "band_1_mean": b1_means,
            "band_2_mean": b2_means,
        }
    )

    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 7. Submission
    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} < {THRESHOLD}. Generating submission..."
        )

        X_test = data_dict["X_test"]
        meta_test = data_dict["meta_test"]
        test_ids = data_dict["test_ids"]

        test_ds = IcebergDataset(X_test, meta_test, y=None, transform=False)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        test_preds_accum = np.zeros((len(test_ids), 1))

        for path in model_paths:
            model = DPCNet().to(device)
            model.load_state_dict(torch.load(path))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for inputs in test_loader:
                    # Test loader returns (img, angle) tuple (no label)
                    img, angle = inputs
                    img = img.to(device)
                    angle = angle.to(device)

                    outputs = model((img, angle))
                    probs = torch.sigmoid(outputs)
                    fold_preds.extend(probs.cpu().numpy())

            test_preds_accum += np.array(fold_preds)

        avg_test_preds = test_preds_accum / N_FOLDS

        # Save
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds.flatten()})

        os.makedirs("./submission", exist_ok=True)
        sub_path = "./submission/submission.csv"
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(f"\nValidation metric {val_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
