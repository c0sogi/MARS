import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_data, get_fold_loaders
from library.train import run_fold
from library.inference import predict_test
from library.model import ACICNN


def main():
    # 1. Configuration for Fast Baseline
    Config.EPOCHS = 20
    Config.setup()
    seed_everything(Config.SEED)

    # 2. Load Data
    # Utilizing cached data for speed
    data = get_data(load_cached_data=True)
    X = data["X_train"]
    y = data["y_train"]
    ids = data["ids_train"]
    angles = data["angles_train"]

    # 3. Cross-Validation Loop
    oof_preds = np.zeros(len(y))
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Iterate through folds to train and collect OOF predictions
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Train the fold (returns best loss, scaler, imputation value, and checkpoint path)
        best_loss, scaler, imp_val, ckpt_path = run_fold(fold_idx, data)

        # Generate OOF predictions for this fold
        # Retrieve the validation loader for the current fold (applies correct preprocessing)
        _, val_loader, _, _ = get_fold_loaders(
            fold_idx, data, batch_size=Config.BATCH_SIZE
        )

        # Load the best model for this fold
        model = ACICNN().to(Config.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for imgs, raw_angs, norm_angs, _ in val_loader:
                imgs = imgs.to(Config.DEVICE)
                raw_angs = raw_angs.to(Config.DEVICE)
                norm_angs = norm_angs.to(Config.DEVICE)

                outputs = model(imgs, raw_angs, norm_angs).squeeze(1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.append(probs)

        # Store predictions in the OOF array
        oof_preds[val_idx] = np.concatenate(fold_preds)

    # 4. Validation Assessment on Hold-out Set (Metadata)
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Metadata file not found: {val_meta_path}")

    val_meta = pd.read_csv(val_meta_path)

    # Map IDs to indices in the loaded dataset
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    # Identify indices corresponding to the metadata validation set
    val_indices = []
    for id_ in val_meta["id"]:
        if id_ in id_to_idx:
            val_indices.append(id_to_idx[id_])

    if not val_indices:
        raise ValueError("No matching validation IDs found in training data.")

    y_true_val = y[val_indices]
    y_pred_val = oof_preds[val_indices]

    # Compute and Print Metric
    final_metric = log_loss(y_true_val, y_pred_val)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Calculate absolute error
    errors = np.abs(y_true_val - y_pred_val)
    val_angles = angles[val_indices]

    # Calculate correlation with incidence angle (ignoring NaNs)
    mask = ~np.isnan(val_angles)
    if np.sum(mask) > 1:
        corr, _ = pearsonr(errors[mask], val_angles[mask])
        print(f"Correlation between Error and Incidence Angle: {corr}")
    else:
        print("Not enough valid incidence angles for correlation analysis.")

    # 6. Submission
    threshold = 0.17174082291273365
    if final_metric < threshold:
        predict_test()
    else:
        print(f"Validation metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
