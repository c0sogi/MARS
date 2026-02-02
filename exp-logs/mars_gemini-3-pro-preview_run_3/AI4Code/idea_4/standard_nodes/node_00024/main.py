import os
import pandas as pd
import numpy as np
import torch
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.backbone import SemanticModel
from library.feature_extractor import FeatureEngineer
from library.regressor import LGBMRanker
from library.data_loader import get_notebook_cells

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def reconstruct_orders(df_features, predictions, metadata_path):
    """
    Reconstructs the cell order dictionary {id: "cell_order_str"}
    from feature dataframe and model predictions.
    Used for validation scoring.
    """
    # Map predictions to (notebook_id, cell_id)
    df_pred = df_features.copy()
    df_pred["pred_rank"] = predictions

    # Create lookup: nb_id -> {cell_id: pred_rank}
    pred_lookup = {}
    for nb_id, group in df_pred.groupby("id"):
        pred_lookup[nb_id] = dict(zip(group["cell_id"], group["pred_rank"]))

    df_meta = pd.read_csv(metadata_path)
    reconstructed = {}

    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        file_path = row["file_path"]

        # Load notebook structure
        # In validation mode, we just need the list of cells to sort.
        # We don't pass cell_order string here because we want to simulate inference
        # (sorting based on our predictions, not ground truth ranks).
        nb_data = get_notebook_cells(nb_id, file_path)
        code_cells = nb_data["code_cells"]
        markdown_cells = nb_data["markdown_cells"]
        n_code = len(code_cells)

        if nb_id in pred_lookup:
            nb_preds = pred_lookup[nb_id]
            for md in markdown_cells:
                cell_id = md["id"]
                if cell_id in nb_preds:
                    # Convert normalized rank back to relative position
                    md["rank"] = nb_preds[cell_id] * n_code
                else:
                    md["rank"] = n_code
        else:
            # Fallback for notebooks not in features (e.g. empty)
            start_rank = n_code
            for i, md in enumerate(markdown_cells):
                md["rank"] = start_rank + i

        # Combine and sort
        all_cells = code_cells + markdown_cells
        all_cells.sort(key=lambda x: x["rank"])

        # Create order string
        reconstructed[nb_id] = " ".join([c["id"] for c in all_cells])

    return reconstructed


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    # We use DEBUG=True to limit data size, but increase sample size to 15,000
    # to ensure we meet the performance threshold (> 0.7633).
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 15000

    print("=" * 40)
    print("Starting DASAR Pipeline")
    print(f"Mode: {'DEBUG' if Config.DEBUG else 'FULL'}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print("=" * 40)

    # --------------------------------------------------------------------------
    # 2. Stage 1: Domain-Adaptive Semantic Backbone
    # --------------------------------------------------------------------------
    print("\n[Stage 1] Fine-tuning Semantic Backbone...")
    backbone = SemanticModel()

    # Fine-tune on (Markdown, Next_Code) pairs
    backbone.fine_tune(
        train_metadata_path=Config.TRAIN_METADATA_PATH,
        output_path=Config.FINE_TUNED_MODEL_PATH,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # --------------------------------------------------------------------------
    # 3. Stage 2: Feature Extraction
    # --------------------------------------------------------------------------
    print("\n[Stage 2] Extracting Features...")
    engineer = FeatureEngineer(backbone)

    # Extract Train Features
    train_df = engineer.extract_features(
        metadata_path=Config.TRAIN_METADATA_PATH,
        mode="train",
        cache_name="train_features",
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Extract Validation Features
    val_df = engineer.extract_features(
        metadata_path=Config.VAL_METADATA_PATH,
        mode="train",  # 'train' mode to calculate targets for validation evaluation
        cache_name="val_features",
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # --------------------------------------------------------------------------
    # 4. Stage 3: Regressor Training
    # --------------------------------------------------------------------------
    print("\n[Stage 3] Training Regressor...")
    ranker = LGBMRanker()
    ranker.train(train_df, val_df)

    # --------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n[Validation] Evaluating Model...")

    # Generate raw predictions (normalized ranks)
    val_preds = ranker.predict(val_df)

    # Reconstruct cell orders for Kendall Tau computation
    # We need to subset the metadata if we are in debug mode to match the features
    if Config.DEBUG:
        # Filter metadata to only include IDs present in val_df
        val_ids = set(val_df["id"].unique())
        temp_meta_path = os.path.join(Config.WORKING_DIR, "val_meta_debug.csv")
        full_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
        debug_val_meta = full_val_meta[full_val_meta["id"].isin(val_ids)]
        debug_val_meta.to_csv(temp_meta_path, index=False)
        eval_meta_path = temp_meta_path
        df_ground_truth = debug_val_meta
    else:
        eval_meta_path = Config.VAL_METADATA_PATH
        df_ground_truth = pd.read_csv(eval_meta_path)

    pred_orders = reconstruct_orders(val_df, val_preds, eval_meta_path)

    # Compute Metric
    kt_score = compute_kendall_tau(df_ground_truth, pred_orders)
    print(f"Final Validation Metric: {kt_score}")

    # Failure Analysis
    print("\n[Analysis] Performing Failure Analysis...")
    val_analysis = val_df.copy()
    val_analysis["pred"] = val_preds
    val_analysis["error"] = np.abs(val_analysis["target"] - val_analysis["pred"])

    analysis_cols = Config.FEATURES + ["error"]
    correlations = (
        val_analysis[analysis_cols].corr()["error"].sort_values(ascending=False)
    )

    print("Correlation between Features and Prediction Error:")
    print(correlations)

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7633

    if kt_score > THRESHOLD:
        print(
            f"\n[Submission] Metric {kt_score} > {THRESHOLD}. Generating submission..."
        )

        # Extract Test Features
        # Note: Test set is 20k samples. If DEBUG is True, we might be limiting this.
        # However, for submission we MUST predict on the full test set provided in metadata/test.csv.
        # We temporarily disable debug sampling for test feature extraction to ensure full coverage.

        test_df = engineer.extract_features(
            metadata_path=Config.TEST_METADATA_PATH,
            mode="test",
            cache_name="test_features",
            load_cached_data=True,
            debug=False,  # Force full test set processing
        )

        # Predict
        test_preds = ranker.predict(test_df)

        # Generate Submission File
        ranker.generate_submission(test_df, test_preds)

    else:
        print(f"\n[Submission] Metric {kt_score} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
