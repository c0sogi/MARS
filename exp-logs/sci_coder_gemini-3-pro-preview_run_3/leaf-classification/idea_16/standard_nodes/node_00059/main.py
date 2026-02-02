import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import spearmanr

from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.trainer import KFoldTrainer
from library.data_manager import DensifiedDataLoader


def main():
    # Set random seeds for reproducibility
    seed_everything()

    print("Initializing workflow...")

    # ---------------------------------------------------------
    # 1. Train Ensemble
    # ---------------------------------------------------------
    # The trainer handles feature extraction (with caching) and K-Fold training.
    # Models are saved to Config.MODEL_DIR.
    trainer = KFoldTrainer()
    trainer.train_kfold_ensemble(load_cached_data=True)

    # ---------------------------------------------------------
    # 2. Validation on Hold-Out Set
    # ---------------------------------------------------------
    print("\nPerforming validation on hold-out dataset...")

    # Load validation data (Canonical representation: 1 centroid per image)
    loader = DensifiedDataLoader()
    val_data = loader.generate_inference_data(
        dataset_name="val", csv_path=Config.VAL_CSV, load_cached_data=True
    )

    X_dino_val = val_data["dino"]
    X_conv_val = val_data["convnext"]
    X_tab_val = val_data["tabular"]
    y_val = val_data["y"]
    ids_val = val_data["ids"]

    # Retrieve class names from the first trained model
    # We assume all models in the ensemble have the same class ordering
    model_path_0 = os.path.join(Config.MODEL_DIR, "model_fold_0.pkl")
    if not os.path.exists(model_path_0):
        raise RuntimeError(f"Model file not found: {model_path_0}")

    with open(model_path_0, "rb") as f:
        model_0 = pickle.load(f)
    class_names = model_0.classes_

    # Perform Ensemble Inference
    # Average probabilities across all K folds
    avg_probs_val = np.zeros((len(y_val), len(class_names)))
    models_found = 0

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold}.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Predict probabilities
            probs = model.predict_proba(X_dino_val, X_conv_val, X_tab_val)
            avg_probs_val += probs
            models_found += 1

    if models_found == 0:
        raise RuntimeError("No models found for validation inference.")

    avg_probs_val /= models_found

    # Clip probabilities to ensure numerical stability and metric consistency
    avg_probs_val = clip_probabilities(avg_probs_val)

    # Compute and print the Final Validation Metric (Log Loss)
    val_metric = log_loss(y_val, avg_probs_val, labels=class_names)
    print(f"Final Validation Metric: {val_metric}")

    # ---------------------------------------------------------
    # 3. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample log loss
    # Map class strings to integer indices
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    y_indices = np.array([class_to_idx[label] for label in y_val])

    # Extract predicted probability for the true class
    # p_true[i] is the probability assigned to the correct label for sample i
    p_true = avg_probs_val[np.arange(len(y_val)), y_indices]

    # Calculate loss: -log(p)
    sample_losses = -np.log(p_true)

    # Calculate Spearman correlation between error (loss) and each tabular feature
    feature_cols = loader._get_feature_columns()
    correlations = []

    for i, col_name in enumerate(feature_cols):
        feat_values = X_tab_val[:, i]

        # Skip constant features to avoid warnings
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr, _ = spearmanr(sample_losses, feat_values)
            if np.isnan(corr):
                corr = 0.0

        correlations.append((col_name, corr))

    # Sort by magnitude of correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # ---------------------------------------------------------
    # 4. Generate Submission
    # ---------------------------------------------------------
    # Generate predictions for the test set and save to CSV
    trainer.generate_submission(load_cached_data=True)

    print("\nExecution complete.")


if __name__ == "__main__":
    main()
