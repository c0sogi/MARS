import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, calc_kendall_tau, reconstruct_order
from library.semantic_model import FineTuner
from library.feature_engineering import generate_features
from library.ranker_model import LGBMRanker
from library.data_processing import load_notebook_data


def main():
    # 1. Setup
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Configure for fast baseline execution to meet time constraints
    # We use a subset of data for training/validation but will process full test set
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 3000
    Config.NUM_EPOCHS = 1

    set_seed(Config.SEED)

    print("--- Starting HC-SSR Pipeline ---")

    # 2. Semantic Backbone Fine-Tuning
    print("\n[Stage 1] Fine-tuning Semantic Backbone...")
    # Train the SentenceTransformer on relaxed (MD, Code) pairs
    ft_model = FineTuner()
    ft_model.train(load_cached_data=True)
    # Model is automatically saved to Config.BACKBONE_OUTPUT_DIR

    # 3. Feature Generation
    print("\n[Stage 2] Generating Features...")
    # Generate Train Features (using fine-tuned model)
    df_train_feats = generate_features(
        metadata_path=Config.TRAIN_METADATA_PATH,
        output_path=Config.TRAIN_FEATURES_PATH,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Generate Validation Features
    df_val_feats = generate_features(
        metadata_path=Config.VAL_METADATA_PATH,
        output_path=Config.VAL_FEATURES_PATH,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # 4. Ranker Training
    print("\n[Stage 3] Training Ranker...")
    ranker = LGBMRanker()
    ranker.train(df_train_feats, df_val_feats)

    # 5. Validation & Failure Analysis
    print("\n[Stage 4] Validation and Failure Analysis...")

    # Predict normalized ranks on validation set
    val_preds = ranker.predict(df_val_feats)
    df_val_feats["pred_rank"] = val_preds

    # Scale normalized rank [0, 1] back to integer index space of code cells
    # scaled_rank = pred_rank * n_code
    df_val_feats["scaled_rank"] = df_val_feats["pred_rank"] * df_val_feats["n_code"]

    # Load validation notebook structures (code cell IDs are needed for reconstruction)
    val_notebooks = load_notebook_data(Config.VAL_METADATA_PATH, debug=Config.DEBUG)

    val_predictions = {}
    val_feats_grp = df_val_feats.groupby("id")

    # Reconstruct orders for each notebook
    for nb_id, nb_data in val_notebooks.items():
        code_cells = list(nb_data["code"].keys())

        md_scores = {}
        if nb_id in val_feats_grp.groups:
            group = val_feats_grp.get_group(nb_id)
            # Map cell_id -> scaled_rank
            md_scores = dict(zip(group["cell_id"], group["scaled_rank"]))

        pred_order = reconstruct_order(code_cells, md_scores)
        val_predictions[nb_id] = pred_order

    # Calculate Kendall Tau Metric
    df_val_gt = pd.read_csv(Config.VAL_METADATA_PATH)
    # Filter GT to processed notebooks (handles debug sampling)
    df_val_gt = df_val_gt[df_val_gt["id"].isin(val_predictions.keys())]

    kendall_score = calc_kendall_tau(df_val_gt, val_predictions)
    print(f"Final Validation Metric: {kendall_score:.15f}")

    # Failure Analysis
    # Calculate absolute error: |pred_rank - true_rank|
    # Note: 'rank' in features is the normalized ground truth [0,1]
    df_val_feats["error"] = (df_val_feats["pred_rank"] - df_val_feats["rank"]).abs()

    print("Failure Analysis (Correlation with Error):")
    analysis_features = ["n_code", "md_len", "sim_max", "smoothed_best_match_loc"]
    correlations = (
        df_val_feats[analysis_features + ["error"]].corr()["error"].drop("error")
    )
    print(correlations)

    # 6. Submission
    SUBMISSION_THRESHOLD = 0.8061

    if kendall_score > SUBMISSION_THRESHOLD:
        print("\n[Stage 5] Generating Submission...")

        # Generate Test Features (Full Test Set)
        # We force debug=False to ensure we process all test notebooks
        df_test_feats = generate_features(
            metadata_path=Config.TEST_METADATA_PATH,
            output_path=Config.TEST_FEATURES_PATH,
            load_cached_data=True,
            debug=False,
        )

        # Predict
        test_preds = ranker.predict(df_test_feats)
        df_test_feats["pred_rank"] = test_preds
        df_test_feats["scaled_rank"] = (
            df_test_feats["pred_rank"] * df_test_feats["n_code"]
        )

        # Load test notebook structures
        test_notebooks = load_notebook_data(Config.TEST_METADATA_PATH, debug=False)

        submission_rows = []
        test_feats_grp = df_test_feats.groupby("id")

        # Iterate through all test notebooks defined in sample_submission to ensure correct order/completeness
        sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
        df_sample_sub = pd.read_csv(sample_sub_path)
        test_ids = df_sample_sub["id"].tolist()

        for nb_id in test_ids:
            if nb_id not in test_notebooks:
                # Fallback for missing data (should not happen)
                submission_rows.append({"id": nb_id, "cell_order": ""})
                continue

            nb_data = test_notebooks[nb_id]
            code_cells = list(nb_data["code"].keys())

            md_scores = {}
            if nb_id in test_feats_grp.groups:
                group = test_feats_grp.get_group(nb_id)
                md_scores = dict(zip(group["cell_id"], group["scaled_rank"]))

            final_order = reconstruct_order(code_cells, md_scores)
            submission_rows.append({"id": nb_id, "cell_order": final_order})

        # Save Submission
        df_submission = pd.DataFrame(submission_rows)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation score {kendall_score:.4f} is below threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
