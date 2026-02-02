import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset
from library.feature_engine import EmbeddingGenerator
from library.processors import FoldProcessor
from library.model_definitions import get_bagged_lr_pipeline, get_hyperparameter_grid


def train_pipeline(X, y, pipeline_name, logger):
    """
    Trains a Bagged Logistic Regression pipeline using GridSearchCV.
    """
    logger.info(f"Starting Grid Search for {pipeline_name}...")

    # Get hyperparameter grid
    raw_grid = get_hyperparameter_grid()
    # Prefix for BaggingClassifier's base estimator
    param_grid = {f"estimator__{k}": v for k, v in raw_grid.items()}

    # Initialize model
    # n_jobs=1 for the estimator to allow GridSearchCV to manage parallelism
    base_model = get_bagged_lr_pipeline(n_jobs=1)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X, y)

    logger.info(f"{pipeline_name} Best CV Score: {grid_search.best_score_:.4f}")
    logger.info(f"{pipeline_name} Best Params: {grid_search.best_params_}")

    return grid_search.best_estimator_


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    logger = setup_logger("runfile.log")
    logger.info("Starting runfile execution...")

    # 2. Load Data
    # load_cached_data=True allows using pre-processed parquet files if they exist
    df_train, df_val, df_test = load_dataset(load_cached_data=True)

    y_train = df_train[Config.TARGET_COL].values.astype(int)
    y_val = df_val[Config.TARGET_COL].values.astype(int)

    logger.info(
        f"Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape: {df_test.shape}"
    )

    # 3. Generate Embeddings
    # This handles caching internally
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_dataset_embeddings(
        df_train, df_val, df_test, load_cached_data=True
    )

    # Extract embeddings for the fixed split
    anchor_train = embeddings["anchor"]["train"]
    aux_train = embeddings["aux"]["train"]

    anchor_val = embeddings["anchor"]["val"]
    aux_val = embeddings["aux"]["val"]

    anchor_test = embeddings["anchor"]["test"]
    aux_test = embeddings["aux"]["test"]

    # 4. Feature Processing
    # We fit the processor on the Training set and transform Validation and Test
    logger.info("Processing features...")
    processor = FoldProcessor()

    # Fit & Transform Train
    views_train = processor.fit_transform(df_train, anchor_train, aux_train)
    # Transform Val
    views_val = processor.transform(df_val, anchor_val, aux_val)
    # Transform Test (prepared for later)
    views_test = processor.transform(df_test, anchor_test, aux_test)

    # 5. Train Models on Fixed Split
    # Pipeline A: Parsimonious (Anchor + Meta)
    model_a = train_pipeline(views_train["view_A"], y_train, "Pipeline A", logger)

    # Pipeline B: Augmented (Anchor + Aux + Meta)
    model_b = train_pipeline(views_train["view_B"], y_train, "Pipeline B", logger)

    # 6. Validation Inference & Consensus
    logger.info("Evaluating on Validation Set...")

    # Get probabilities for class 1
    prob_a_val = model_a.predict_proba(views_val["view_A"])[:, 1]
    prob_b_val = model_b.predict_proba(views_val["view_B"])[:, 1]

    # Soft Voting Consensus
    prob_val_consensus = 0.5 * prob_a_val + 0.5 * prob_b_val

    # Calculate Metric
    val_auc = roc_auc_score(y_val, prob_val_consensus)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")
    df_val_analysis = df_val.copy()
    df_val_analysis["pred"] = prob_val_consensus
    df_val_analysis["target"] = y_val
    df_val_analysis["error"] = np.abs(
        df_val_analysis["target"] - df_val_analysis["pred"]
    )

    print("\n--- Failure Analysis: Correlation with Error ---")
    correlations = []
    # Analyze numerical columns
    for col in Config.NUMERICAL_COLS:
        if col in df_val_analysis.columns:
            # Fill NaNs with median for correlation calculation
            col_data = df_val_analysis[col].fillna(df_val_analysis[col].median())
            # Calculate Pearson correlation
            if col_data.std() > 0:  # Avoid division by zero
                corr = np.corrcoef(col_data, df_val_analysis["error"])[0, 1]
                correlations.append((col, corr))
            else:
                correlations.append((col, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for col, corr in correlations:
        print(f"{col}: {corr:.4f}")
    print("------------------------------------------------\n")

    # 8. Submission Logic
    THRESHOLD = 0.7190361601447052

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation metric {val_auc} > {THRESHOLD}. Proceeding to submission..."
        )

        # Strategy: Retrain on Full Data (Train + Val) for best performance
        logger.info("Retraining on combined Train + Validation set...")

        # Merge DataFrames
        df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        y_full = df_full[Config.TARGET_COL].values.astype(int)

        # Merge Embeddings
        anchor_full = np.vstack([anchor_train, anchor_val])
        aux_full = np.vstack([aux_train, aux_val])

        # Re-initialize Processor and Fit on Full Data
        full_processor = FoldProcessor()
        views_full = full_processor.fit_transform(df_full, anchor_full, aux_full)

        # Transform Test Data using the Full Processor
        views_test_full = full_processor.transform(df_test, anchor_test, aux_test)

        # Retrain Models
        model_a_full = train_pipeline(
            views_full["view_A"], y_full, "Full Pipeline A", logger
        )
        model_b_full = train_pipeline(
            views_full["view_B"], y_full, "Full Pipeline B", logger
        )

        # Inference on Test
        prob_a_test = model_a_full.predict_proba(views_test_full["view_A"])[:, 1]
        prob_b_test = model_b_full.predict_proba(views_test_full["view_B"])[:, 1]

        prob_test_consensus = 0.5 * prob_a_test + 0.5 * prob_b_test

        # Save Submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": prob_test_consensus,
            }
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation metric {val_auc} is below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
