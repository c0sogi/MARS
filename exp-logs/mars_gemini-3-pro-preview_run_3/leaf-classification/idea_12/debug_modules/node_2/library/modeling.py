import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss

from library.config import (
    SUBMISSION_PATH,
    N_SPLITS,
    PCA_VARIANCE,
    EPSILON,
    WORKING_DIR,
    SEED,
)
from library.utils import save_pickle, seed_everything


def create_pipeline(tabular_dim=192):
    """
    Constructs the Scikit-Learn pipeline for the Leaf Classification task.

    Structure:
    1. ColumnTransformer:
       - Image Features (first N-192 cols): Passthrough
       - Tabular Features (last 192 cols): QuantileTransformer (Normal)
    2. PCA: Retain 99% variance
    3. LDA: LSQR solver with Ledoit-Wolf shrinkage

    Args:
        tabular_dim (int): Number of tabular features at the end of the input vector.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # Define the preprocessor
    # We assume the input array is [ImageFeatures, TabularFeatures]
    # We use negative slicing to target the last 'tabular_dim' columns safely
    preprocessor = ColumnTransformer(
        transformers=[
            # Pass image features through untouched
            ("img_pass", "passthrough", slice(0, -tabular_dim)),
            # Gaussianize tabular features
            (
                "tab_quant",
                QuantileTransformer(output_distribution="normal", random_state=SEED),
                slice(-tabular_dim, None),
            ),
        ]
    )

    # Construct the full pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "pca",
                PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED),
            ),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    return pipeline


def clip_probabilities(probs):
    """
    Clips probabilities to the range [EPSILON, 1-EPSILON] to avoid log loss extremes.
    """
    return np.maximum(np.minimum(probs, 1 - EPSILON), EPSILON)


def run_training(data_manager):
    """
    Executes the Cross-Validated Manifold-Expanded Linear Discriminant Ensemble strategy.

    1. Loads Test Data (Centroid Topology).
    2. Iterates through N_SPLITS folds.
    3. For each fold:
       - Loads Train Data (Expanded Topology) and Val Data (Centroid Topology).
       - Trains the Pipeline.
       - Evaluates on Validation set.
       - Generates predictions on Test set.
    4. Aggregates Test predictions (Ensemble Averaging).
    5. Saves submission file.

    Args:
        data_manager (LeafDataManager): The data manager instance to retrieve data.
    """
    seed_everything(SEED)

    print("Retrieving Test Data (Centroid)...")
    X_test, test_ids = data_manager.get_test_data(load_cached_data=True)

    # Initialize storage for ensemble predictions
    # Shape: (N_test, N_classes)
    num_classes = len(data_manager.classes)
    test_probs_sum = np.zeros((X_test.shape[0], num_classes))

    fold_scores = []

    print(f"Starting {N_SPLITS}-Fold Cross-Validation...")

    for fold_idx in range(N_SPLITS):
        print(f"\n--- Fold {fold_idx} ---")

        # 1. Get Data
        # X_train is Expanded (4 views per ID)
        # X_val is Centroid (1 mean view per ID)
        X_train, y_train, X_val, y_val = data_manager.get_fold_data(
            fold_idx, load_cached_data=True
        )

        # 2. Create and Fit Pipeline
        pipeline = create_pipeline(tabular_dim=192)
        pipeline.fit(X_train, y_train)

        # 3. Validation
        val_probs = pipeline.predict_proba(X_val)
        val_probs_clipped = clip_probabilities(val_probs)
        score = log_loss(y_val, val_probs_clipped)
        fold_scores.append(score)

        print(f"Fold {fold_idx} Log Loss: {score}")

        # 4. Test Inference
        test_probs = pipeline.predict_proba(X_test)
        test_probs_sum += test_probs

        # 5. Save Model Artifact
        model_path = os.path.join(WORKING_DIR, f"pipeline_fold_{fold_idx}.pkl")
        save_pickle(pipeline, model_path)

    # --- Aggregation and Reporting ---
    mean_val_score = np.mean(fold_scores)
    std_val_score = np.std(fold_scores)
    print("\n=== Cross-Validation Results ===")
    print(f"Mean Log Loss: {mean_val_score}")
    print(f"Std Log Loss:  {std_val_score}")

    # Average test probabilities
    avg_test_probs = test_probs_sum / N_SPLITS

    # Clip final probabilities
    final_probs = clip_probabilities(avg_test_probs)

    # --- Submission Generation ---
    print(f"\nGenerating submission file at {SUBMISSION_PATH}...")

    # Create DataFrame
    # Columns: id, Class_1, Class_2, ...
    df_sub = pd.DataFrame(final_probs, columns=data_manager.classes)
    df_sub.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
