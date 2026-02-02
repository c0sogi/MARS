import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import from provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_processed_data, get_test_loader
from library.train import run_fold
from library.model import DWB_DPN


def run():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Fast baseline settings
    Config.EPOCHS = 20

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load processed data (cached)
    # X_train contains both train.csv and val.csv data concatenated
    X_train, y_train, inc_train, X_test, inc_test, test_ids = get_processed_data(
        load_cached_data=True
    )

    # Load metadata to identify the split between original train and hold-out val
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    train_meta = pd.read_csv(train_meta_path)

    # The processed data concatenates train then val.
    # So indices [0 : len(train_meta)] are train.csv
    # Indices [len(train_meta) : ] are val.csv
    split_idx = len(train_meta)

    # ==========================================
    # 3. Training (Stratified 5-Fold CV)
    # ==========================================
    # We run full CV to get OOF predictions for the validation subset.
    # Since StratifiedKFold shuffles, the 'val.csv' samples are distributed across folds.
    oof_preds = np.zeros(len(X_train))

    for fold in range(Config.N_FOLDS):
        # run_fold trains the model, saves it, and returns validation metrics/preds for that fold
        _, val_preds, val_indices = run_fold(fold, X_train, y_train, inc_train, device)

        # Store OOF predictions
        oof_preds[val_indices] = val_preds

    # ==========================================
    # 4. Validation Assessment
    # ==========================================
    # Extract predictions and targets for the hold-out validation set (val.csv)
    val_subset_preds = oof_preds[split_idx:]
    val_subset_targets = y_train[split_idx:]
    val_subset_inc = inc_train[split_idx:]
    val_subset_X = X_train[split_idx:]

    # Calculate Metric
    final_metric = log_loss(val_subset_targets, val_subset_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    # Calculate per-sample log loss
    epsilon = 1e-15
    preds_clipped = np.clip(val_subset_preds, epsilon, 1 - epsilon)
    sample_losses = -(
        val_subset_targets * np.log(preds_clipped)
        + (1 - val_subset_targets) * np.log(1 - preds_clipped)
    )

    # Calculate simple image statistics for correlation
    # X is (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    mean_b1 = np.mean(val_subset_X[:, 0, :, :], axis=(1, 2))
    mean_b2 = np.mean(val_subset_X[:, 1, :, :], axis=(1, 2))

    df_analysis = pd.DataFrame(
        {
            "loss": sample_losses,
            "inc_angle": val_subset_inc,
            "mean_b1": mean_b1,
            "mean_b2": mean_b2,
        }
    )

    # Compute correlation
    corr = df_analysis.corr()["loss"].drop("loss")
    print("Failure Analysis (Correlation with Error):")
    print(corr)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.16676861786296204

    if final_metric < THRESHOLD:
        print("Metric passed threshold. Generating submission...")

        # Initialize accumulator for ensemble predictions
        test_preds_accum = np.zeros(len(X_test))

        # Get test loader
        test_loader = get_test_loader(X_test, inc_test)

        # Iterate over all saved fold models
        for fold in range(Config.N_FOLDS):
            model = DWB_DPN().to(device)
            model_path = os.path.join(Config.WORKING_DIR, f"dwb_dpn_fold_{fold}.pth")

            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for imgs, incs in test_loader:
                    imgs = imgs.to(device)
                    incs = incs.to(device)

                    # Inference
                    outputs = model(imgs, incs)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    fold_preds.extend(probs)

            test_preds_accum += np.array(fold_preds).flatten()

        # Average predictions
        avg_test_preds = test_preds_accum / Config.N_FOLDS

        # Save submission
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
