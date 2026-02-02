import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import data_loader
from library import model as lib_model

# Suppress warnings
warnings.filterwarnings("ignore")


# Set Seeds for Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_proba_gpu(model_instance, X_numpy):
    """
    Performs inference using the learned parameters of ParsimoniousOASDiscriminant
    on the GPU (if available) using PyTorch.

    Args:
        model_instance: Trained ParsimoniousOASDiscriminant instance.
        X_numpy: Input features as numpy array.

    Returns:
        np.ndarray: Probability matrix.
    """
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract parameters from the linear model
    # W_ shape: (n_classes, n_features)
    # b_ shape: (n_classes,)
    W = torch.tensor(model_instance.W_, dtype=torch.float64, device=device)
    b = torch.tensor(model_instance.b_, dtype=torch.float64, device=device)

    # Prepare input
    # Process in batches to avoid OOM if X is massive, though dataset is small here.
    # Given dataset size (~100-1000 rows), full batch is fine.
    X = torch.tensor(X_numpy, dtype=torch.float64, device=device)

    # Inference: Z = X @ W.T + b
    with torch.no_grad():
        logits = torch.matmul(X, W.T) + b
        probs = torch.nn.functional.softmax(logits, dim=1)

    return probs.cpu().numpy()


def run_pipeline():
    set_seed(config.SEED)

    # 1. Load Data
    print("Loading dataset...")
    (train_data, val_data, test_data) = data_loader.load_dataset(load_cached_data=True)
    X_train_raw, y_train, ids_train = train_data
    X_val_raw, y_val, ids_val = val_data
    X_test_raw, ids_test = test_data

    # 2. Preprocess Data (High-Precision Pipeline)
    print("Preprocessing data...")
    X_train, X_val, X_test = data_loader.preprocess_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 3. Train Model
    print("Training ParsimoniousOASDiscriminant...")
    model = lib_model.ParsimoniousOASDiscriminant()
    model.fit(X_train, y_train)

    # 4. Validation Inference (GPU Optimized)
    print("Validating model...")
    y_pred_probs_val = predict_proba_gpu(model, X_val)

    # Calculate Metric
    val_loss = log_loss(y_val, y_pred_probs_val, labels=model.classes_)
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Encode y_val to get indices
    le = model.le_
    y_val_indices = le.transform(y_val)

    # Calculate per-sample error (Negative Log Likelihood of the true class)
    # Clip probabilities to avoid log(0)
    eps = 1e-15
    probs_clipped = np.clip(y_pred_probs_val, eps, 1 - eps)

    # Extract prob of true class
    true_class_probs = probs_clipped[np.arange(len(y_val)), y_val_indices]
    errors = -np.log(true_class_probs)

    # Correlate errors with features
    correlations = []
    n_features = X_val.shape[1]

    # Since features are unnamed in the numpy array, we use indices
    # However, we know they are sorted alphanumerically.
    # We'll just report top indices.
    for i in range(n_features):
        feat_col = X_val[:, i]
        # Handle constant features (though VarianceThreshold should have removed them)
        if np.std(feat_col) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(errors, feat_col)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.4f}")

    # 6. Submission Generation
    threshold = 3.058881515561734e-14

    if val_loss < threshold:
        print(
            f"\nValidation metric ({val_loss}) meets threshold ({threshold}). Generating submission..."
        )

        # Inference on Test Set (GPU Optimized)
        y_pred_probs_test = predict_proba_gpu(model, X_test)

        # Construct Submission DataFrame
        submission_df = pd.DataFrame(y_pred_probs_test, columns=model.classes_)
        submission_df.insert(0, config.ID_COL, ids_test)

        # Save
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric ({val_loss}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
