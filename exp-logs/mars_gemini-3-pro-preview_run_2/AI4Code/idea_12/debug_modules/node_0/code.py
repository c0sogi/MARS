import os
import sys
import warnings
import numpy as np
import pandas as pd

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, kendall_tau_metric, format_submission
from library.data_manager import get_partition_data
from library.vectorization import TextPipeline
from library.stage1_ridge import RidgeStacker
from library.feature_engineering import NeighborhoodFeatureExtractor
from library.stage2_lgbm import LGBMRanker

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration Pipeline...")

    # --------------------------------------------------------------------------
    # 0. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for speed in this demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for rapid execution
    Config.LGBM_PARAMS["n_estimators"] = 10  # Reduce boosting rounds
    Config.LGBM_PARAMS["verbose"] = -1

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration set to DEBUG mode for fast execution.")

    # --------------------------------------------------------------------------
    # 1. Data Loading
    # --------------------------------------------------------------------------
    print("\n--- Step 1: Data Loading ---")
    # Load Train, Val, and Test data using the Data Manager
    # We force load_cached_data=False to demonstrate processing logic
    df_train = get_partition_data("train", load_cached_data=False, debug=True)
    df_val = get_partition_data("val", load_cached_data=False, debug=True)
    df_test = get_partition_data("test", load_cached_data=False, debug=True)

    # Validations
    assert not df_train.empty, "Training dataframe should not be empty."
    assert (
        "pct_rank" in df_train.columns
    ), "Training data must contain target 'pct_rank'."
    assert not df_val.empty, "Validation dataframe should not be empty."
    assert not df_test.empty, "Test dataframe should not be empty."

    print(
        f"Loaded {len(df_train)} train rows, {len(df_val)} val rows, {len(df_test)} test rows."
    )

    # --------------------------------------------------------------------------
    # 2. Text Vectorization
    # --------------------------------------------------------------------------
    print("\n--- Step 2: Vectorization ---")
    text_pipeline = TextPipeline()

    # Fit on training sources
    train_corpus = df_train["source"].astype(str).tolist()
    text_pipeline.fit(train_corpus, load_cached_models=False)

    # Validations
    assert text_pipeline.is_fitted, "TextPipeline should be fitted."
    assert (
        len(text_pipeline.tfidf.vocabulary_) > 0
    ), "TF-IDF vocabulary should not be empty."

    print("TextPipeline fitted successfully.")

    # --------------------------------------------------------------------------
    # 3. Stage 1: Ridge Regression (The "Signpost" Model)
    # --------------------------------------------------------------------------
    print("\n--- Step 3: Stage 1 Ridge Regression ---")
    ridge_stacker = RidgeStacker()

    # 3a. Generate Out-Of-Fold (OOF) predictions for the training set
    # These are needed to train the Stage 2 model without leakage
    print("Generating OOF predictions for training set...")
    oof_preds_train = ridge_stacker.train_and_predict_oof(df_train, text_pipeline)

    # Validations
    assert (
        "ridge_pred" in oof_preds_train.columns
    ), "OOF predictions must contain 'ridge_pred'."
    # Note: oof_preds only contains markdown cells, so length <= len(df_train)

    # 3b. Fit the Ridge model on the full training set
    print("Fitting Ridge model on full training set...")
    ridge_stacker.fit(df_train, text_pipeline)

    # 3c. Predict on Validation set
    print("Predicting Stage 1 for validation set...")
    preds_val_ridge = ridge_stacker.predict(df_val, text_pipeline)

    assert not preds_val_ridge.empty, "Validation predictions should not be empty."

    # --------------------------------------------------------------------------
    # 4. Feature Engineering (Multi-View Neighborhood)
    # --------------------------------------------------------------------------
    print("\n--- Step 4: Feature Engineering ---")
    extractor = NeighborhoodFeatureExtractor()

    # 4a. Extract features for Training set (using OOF Ridge preds)
    print("Extracting features for Training set...")
    train_features = extractor.extract_features(
        df_train,
        text_pipeline,
        ridge_preds=oof_preds_train,
        partition="train",
        load_cached_data=False,
    )

    # 4b. Extract features for Validation set (using standard Ridge preds)
    print("Extracting features for Validation set...")
    val_features = extractor.extract_features(
        df_val,
        text_pipeline,
        ridge_preds=preds_val_ridge,
        partition="val",
        load_cached_data=False,
    )

    # Validations
    expected_cols = ["lex_mean", "lat_mean", "ridge_pred", "pct_rank"]
    for col in expected_cols:
        assert col in train_features.columns, f"Train features missing column: {col}"
        assert col in val_features.columns, f"Val features missing column: {col}"

    print(f"Generated features: Train {train_features.shape}, Val {val_features.shape}")

    # --------------------------------------------------------------------------
    # 5. Stage 2: LightGBM Ranking
    # --------------------------------------------------------------------------
    print("\n--- Step 5: Stage 2 LightGBM Training ---")
    lgbm_ranker = LGBMRanker()

    # Train the ranker
    lgbm_ranker.train_model(train_features, val_features)

    # Validate model file creation
    assert os.path.exists(
        Config.CACHE_STAGE2_LGBM
    ), "LightGBM model file was not saved."

    # Validate prediction logic on validation set
    val_final_preds = lgbm_ranker.predict_rank(val_features)
    assert len(val_final_preds) == len(val_features), "Prediction length mismatch."

    print("Stage 2 model trained and validated.")

    # --------------------------------------------------------------------------
    # 6. Inference on Test Set
    # --------------------------------------------------------------------------
    print("\n--- Step 6: Test Set Inference ---")

    # 6a. Stage 1 Predictions for Test
    test_ridge_preds = ridge_stacker.predict(df_test, text_pipeline)

    # 6b. Feature Extraction for Test
    test_features = extractor.extract_features(
        df_test,
        text_pipeline,
        ridge_preds=test_ridge_preds,
        partition="test",
        load_cached_data=False,
    )

    if test_features.empty:
        print(
            "Warning: No markdown cells found in test sample (likely due to small debug sample)."
        )
        # Creating dummy submission for logic flow if empty
        submission_dict = {nb_id: [] for nb_id in df_test["id"].unique()}
    else:
        # 6c. Stage 2 Predictions for Test
        test_ranks = lgbm_ranker.predict_rank(test_features)

        # Add predictions back to dataframe
        test_features["pred_rank"] = test_ranks

        # 6d. Reconstruct Order
        # We need to merge predictions back with code cells to form the full sequence
        # Code cells are anchors; we assume their relative order is fixed (0..N)
        # Markdown cells are inserted based on predicted rank (0.0..1.0)

        # Get code cells from original df_test
        code_cells = df_test[df_test["cell_type"] == "code"].copy()
        # In test, we don't know absolute rank, but we know relative code_rank
        # Normalize code rank to 0..1 to be comparable with predicted markdown rank

        # Helper to process per notebook
        submission_dict = {}

        for nb_id, group in df_test.groupby("id"):
            # Get code cells
            nb_code = group[group["cell_type"] == "code"].copy()

            # Get markdown predictions for this notebook
            nb_md_preds = test_features[test_features["id"] == nb_id].copy()

            cells_to_sort = []

            # Add code cells with their implicit relative rank
            n_code = len(nb_code)
            if n_code > 0:
                # We distribute code cells evenly between 0 and 1
                # e.g., if 2 code cells: ranks 0.0, 1.0? Or 0.33, 0.66?
                # A simple heuristic is linspace(0, 1, n_code)
                if n_code == 1:
                    code_pos = [0.5]  # Center if single
                else:
                    code_pos = np.linspace(0.0, 1.0, n_code)

                for i, (_, row) in enumerate(nb_code.iterrows()):
                    cells_to_sort.append((row["cell_id"], code_pos[i]))

            # Add markdown cells with predicted rank
            for _, row in nb_md_preds.iterrows():
                cells_to_sort.append((row["cell_id"], row["pred_rank"]))

            # Sort by rank
            cells_to_sort.sort(key=lambda x: x[1])

            # Extract ordered IDs
            ordered_ids = [c[0] for c in cells_to_sort]
            submission_dict[nb_id] = ordered_ids

    # --------------------------------------------------------------------------
    # 7. Submission Generation & Metric Check
    # --------------------------------------------------------------------------
    print("\n--- Step 7: Submission & Metrics ---")

    # Generate submission dataframe
    df_sub = format_submission(submission_dict, save_path=Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())

    # Validate submission format
    assert "id" in df_sub.columns and "cell_order" in df_sub.columns
    assert len(df_sub) > 0

    # Demonstrate Metric Calculation (using Validation set as proxy for GT)
    # We need to construct a prediction dictionary for the validation set
    val_features["pred_rank"] = val_final_preds
    val_preds_dict = {}

    # Reconstruct validation orders
    # Note: This logic duplicates the test reconstruction logic, simplified here
    for nb_id in df_val["id"].unique():
        # Get GT code cells
        nb_code = df_val[(df_val["id"] == nb_id) & (df_val["cell_type"] == "code")]
        nb_md_preds = val_features[val_features["id"] == nb_id]

        cells = []
        n_code = len(nb_code)
        if n_code > 0:
            if n_code == 1:
                code_pos = [0.5]
            else:
                code_pos = np.linspace(0.0, 1.0, n_code)
            for i, (_, row) in enumerate(nb_code.iterrows()):
                cells.append((row["cell_id"], code_pos[i]))

        for _, row in nb_md_preds.iterrows():
            cells.append((row["cell_id"], row["pred_rank"]))

        cells.sort(key=lambda x: x[1])
        val_preds_dict[nb_id] = [c[0] for c in cells]

    # Create Ground Truth DataFrame for metric function
    # We need 'id' and 'cell_order' (space delimited string)
    # df_val was loaded via get_partition_data, but we need the original GT string.
    # We can reconstruct it from the 'rank' column in df_val or load metadata.
    # For this demo, we'll reconstruct from df_val assuming 'rank' is correct.
    gt_rows = []
    for nb_id, group in df_val.groupby("id"):
        sorted_group = group.sort_values("rank")
        order_str = " ".join(sorted_group["cell_id"].tolist())
        gt_rows.append({"id": nb_id, "cell_order": order_str})
    df_val_gt = pd.DataFrame(gt_rows)

    # Calculate Metric
    score = kendall_tau_metric(df_val_gt, val_preds_dict)
    print(f"Validation Kendall Tau Score (Debug Subset): {score:.4f}")

    print("\nPipeline demonstration completed successfully.")


if __name__ == "__main__":
    main()
