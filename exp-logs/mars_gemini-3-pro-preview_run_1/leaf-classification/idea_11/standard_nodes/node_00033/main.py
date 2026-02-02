import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config
from library import data_loader
from library import preprocessing
from library import model

# Set fixed random seed for reproducibility
np.random.seed(config.SEED)


def perform_failure_analysis(lda_model, X_val, y_val):
    """
    Analyzes the correlation between feature values and the log loss error
    on the validation set to identify systematic failure modes.
    """
    print("\n--- Failure Analysis ---")

    # Predict probabilities
    probas = lda_model.predict_proba(X_val)
    probas_clipped = model.clip_probabilities(probas)

    # Map class names to integer indices based on the model's classes_
    class_to_idx = {cls: i for i, cls in enumerate(lda_model.classes_)}

    # Get the index of the true class for each sample
    y_indices = np.array([class_to_idx[cls] for cls in y_val])

    # Extract the probability assigned to the true class
    # Using advanced indexing: [row_indices, col_indices]
    true_class_probs = probas_clipped[np.arange(len(y_val)), y_indices]

    # Calculate Log Loss per sample: -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between each feature and the error magnitude
    # X_val is (n_samples, n_features)
    feature_names = config.get_ordered_feature_list()
    correlations = []

    for i in range(X_val.shape[1]):
        feat_values = X_val[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            # Pearson correlation
            corr = np.corrcoef(feat_values, sample_losses)[0, 1]

        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation strength (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    return sample_losses


def main():
    # 1. Train and Evaluate
    # This function handles pipeline fitting, data transformation, model training,
    # and validation scoring using the provided libraries.
    # load_cached_data=True ensures we use the pre-processed parquet/npy files if available.
    lda_model, pipeline, val_loss = model.train_and_evaluate(load_cached_data=True)

    # 2. Print Final Metric
    # Must be printed exactly in this format with full precision.
    print(f"Final Validation Metric: {val_loss}")

    # 3. Failure Analysis
    # We need to retrieve the validation data again to perform per-sample analysis.
    # The pipeline is already fitted, so we pass it to transform the data.
    X_val, y_val, _ = preprocessing.get_transformed_data(
        "val", pipeline=pipeline, load_cached_data=True
    )
    perform_failure_analysis(lda_model, X_val, y_val)

    # 4. Conditional Submission Generation
    # The task requires generating a submission ONLY if the metric is below a specific threshold.
    threshold = 1.470544781593644e-08

    if val_loss < threshold:
        print(f"\nValidation metric ({val_loss}) meets the threshold ({threshold}).")
        model.generate_submission(lda_model, pipeline, load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({val_loss}) is not lower than threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
