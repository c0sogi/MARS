import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch

# Import provided library modules
from library import config
from library import data_loader
from library import feature_engine
from library import hybrid_model


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_failure_analysis(model, X_val, y_val):
    """
    Analyzes model performance on the validation set.
    Calculates per-sample loss and correlates it with feature statistics.
    """
    print("\nPerforming Failure Analysis...")

    # Predict probabilities
    probs = model.predict_proba(X_val)

    # Ensure y_val is integer type for indexing
    y_val = y_val.astype(int)

    # Extract probabilities for the true classes
    # probs shape: (n_samples, n_classes)
    true_class_probs = probs[np.arange(len(y_val)), y_val]

    # Clip to avoid log(0) and calculate Log Loss per sample
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_losses = -np.log(true_class_probs)

    print(f"Mean Validation Loss: {np.mean(sample_losses):.6f}")
    print(f"Max Validation Loss: {np.max(sample_losses):.6f}")

    # Calculate feature statistics to correlate with error
    # X_val shape: (n_samples, 4608)
    feature_stats = {
        "Feature_Mean": np.mean(X_val, axis=1),
        "Feature_Std": np.std(X_val, axis=1),
        "Feature_L2_Norm": np.linalg.norm(X_val, axis=1),
        "Feature_Max": np.max(X_val, axis=1),
        "Feature_Min": np.min(X_val, axis=1),
    }

    # Calculate view-specific statistics (1536 dims per view)
    dim = 1536
    feature_stats["Global_View_Mean"] = np.mean(X_val[:, :dim], axis=1)
    feature_stats["Standard_View_Mean"] = np.mean(X_val[:, dim : 2 * dim], axis=1)
    feature_stats["Local_View_Mean"] = np.mean(X_val[:, 2 * dim :], axis=1)

    print("\nCorrelation between Error Magnitude (Log Loss) and Input Features:")
    for stat_name, stat_values in feature_stats.items():
        corr, p_val = pearsonr(sample_losses, stat_values)
        print(f"  {stat_name}: Correlation={corr:.4f}, P-value={p_val:.4f}")


def main():
    # 1. Setup
    set_seed(config.SEED)

    # 2. Data Loading
    # debug=False ensures we use the full dataset for maximum performance
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, class_to_idx = data_loader.get_dataloaders(
        debug=False
    )

    # 3. Feature Extraction
    # Extracts features for Global, Standard, and Local views and fuses them.
    # Uses caching to speed up subsequent runs.
    print("Extracting Features...")
    X_train, y_train, train_ids = feature_engine.extract_features(
        train_loader, "train", load_cached_data=True
    )
    X_val, y_val, val_ids = feature_engine.extract_features(
        val_loader, "val", load_cached_data=True
    )
    X_test, _, test_ids = feature_engine.extract_features(
        test_loader, "test", load_cached_data=True
    )

    # 4. Model Training
    print("Initializing and Training Hybrid Ensemble...")
    model = hybrid_model.HybridEnsemble()
    model.fit(X_train, y_train)

    # 5. Optimization & Validation
    # Optimizes the weight between Linear and Non-Linear heads on the validation set
    final_metric = model.optimize_weights(X_val, y_val)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(model, X_val, y_val)

    # 7. Submission Generation
    # Only submit if the metric is better than the specified threshold
    THRESHOLD = 0.11640673500383826

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        hybrid_model.generate_submission(
            model=model,
            X_test=X_test,
            test_ids=test_ids,
            output_path=config.SUBMISSION_FILE,
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
