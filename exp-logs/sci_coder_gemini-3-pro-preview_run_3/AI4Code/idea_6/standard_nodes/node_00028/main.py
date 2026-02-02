import os
import sys
import pandas as pd
import numpy as np
import warnings
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library components
from library.config import Config
from library.utils import set_seed, read_notebook, kendall_tau_metric
from library.backbones import BackboneFineTuner
from library.features import DualViewFeatureExtractor
from library.regressor import RankRegressor, generate_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing DVSAR Pipeline...")

    # 2. Data Sampling for Fast Training
    # We create a sampled version of the training metadata to speed up fine-tuning and regressor training.
    # This ensures the script completes well within the time limit.
    print("Sampling training data...")
    full_train_df = pd.read_csv(Config.TRAIN_PATH)
    SAMPLE_SIZE = 5000

    if len(full_train_df) > SAMPLE_SIZE:
        sampled_train_df = full_train_df.sample(n=SAMPLE_SIZE, random_state=Config.SEED)
    else:
        sampled_train_df = full_train_df

    sampled_train_path = os.path.join(Config.WORKING_DIR, "train_sampled.csv")
    sampled_train_df.to_csv(sampled_train_path, index=False)
    print(f"Created sampled training set with {len(sampled_train_df)} notebooks.")

    # Update Config to point to the sampled dataset for training operations.
    # This affects both the BackboneFineTuner (which reads Config.TRAIN_PATH)
    # and our manual calls to feature extraction.
    Config.TRAIN_PATH = sampled_train_path

    # 3. Stage 1: Fine-Tune Backbones
    # We train both the Text-View and Code-View models using Contrastive Learning.
    print("\n=== Stage 1: Fine-Tuning Backbones ===")

    # Text Model
    text_tuner = BackboneFineTuner(Config.MODEL_TEXT, Config.TEXT_MODEL_SAVE_PATH)
    # Check if models already exist to avoid retraining if re-running
    if not os.path.exists(Config.TEXT_MODEL_SAVE_PATH):
        # load_cached_data=False ensures we generate pairs from the new sampled csv
        text_tuner.train(load_cached_data=False)
    else:
        print("Text model found, skipping training.")

    # Code Model
    code_tuner = BackboneFineTuner(Config.MODEL_CODE, Config.CODE_MODEL_SAVE_PATH)
    if not os.path.exists(Config.CODE_MODEL_SAVE_PATH):
        code_tuner.train(load_cached_data=False)
    else:
        print("Code model found, skipping training.")

    # 4. Stage 2: Feature Extraction
    print("\n=== Stage 2: Feature Extraction ===")
    extractor = DualViewFeatureExtractor()

    # Extract Training Features (Sampled)
    # We disable cache loading for train to ensure we get features for the sampled set
    # (and not a potentially existing full set cache from a previous run).
    print("Extracting training features...")
    df_train_features = extractor.extract_features(
        Config.TRAIN_PATH, mode="train", load_cached_data=False
    )

    # Extract Validation Features (Full)
    # We use the full validation set as required for rigorous evaluation.
    print("Extracting validation features...")
    df_val_features = extractor.extract_features(
        Config.VAL_PATH, mode="val", load_cached_data=True
    )

    # 5. Stage 3: Train Regressor
    print("\n=== Stage 3: Train Regressor ===")
    regressor = RankRegressor()
    regressor.train(df_train_features, df_val_features)

    # 6. Validation & Failure Analysis
    print("\n=== Validation and Failure Analysis ===")

    # Predict on Validation Set
    val_preds = regressor.predict(df_val_features)
    df_val_features["pred_rank"] = val_preds

    # Reconstruct Cell Orders for Kendall Tau Calculation
    df_val_meta = pd.read_csv(Config.VAL_PATH)
    val_submission_rows = []

    # Create a map of id -> DataFrame for fast lookup
    preds_grouped = df_val_features.groupby("id")

    print("Reconstructing validation orders...")
    for _, row in df_val_meta.iterrows():
        nb_id = row["id"]
        try:
            data = read_notebook(row["file_path"])
            cell_types = data.get("cell_type", {})
        except:
            continue

        all_cells = list(cell_types.keys())
        code_cells = [c for c in all_cells if cell_types[c] == "code"]
        n_code = len(code_cells)

        ranking = []

        # Place code cells at fixed pivot positions (0.5, 1.5, ...)
        # This acts as the skeleton of the notebook
        for i, cid in enumerate(code_cells):
            ranking.append((i + 0.5, cid))

        # Place markdown cells based on predicted rank
        if nb_id in preds_grouped.groups:
            group = preds_grouped.get_group(nb_id)
            # Map cell_id to predicted rank
            pred_map = dict(zip(group["cell_id"], group["pred_rank"]))

            for cid in all_cells:
                if cell_types[cid] == "markdown":
                    # Scale normalized rank back to absolute position
                    # Default to end if missing
                    rank = pred_map.get(cid, 1.0) * n_code
                    ranking.append((rank, cid))
        else:
            # Fallback if no markdown predictions found (e.g. no markdown cells or error)
            for cid in all_cells:
                if cell_types[cid] == "markdown":
                    ranking.append((n_code + 1.0, cid))

        # Sort by the calculated position
        ranking.sort(key=lambda x: x[0])
        ordered_ids = [x[1] for x in ranking]
        val_submission_rows.append({"id": nb_id, "cell_order": " ".join(ordered_ids)})

    df_val_pred = pd.DataFrame(val_submission_rows)
    df_val_gt = df_val_meta[["id", "cell_order"]]

    # Compute Metric
    kt_score = kendall_tau_metric(df_val_pred, df_val_gt)
    print(f"Final Validation Metric: {kt_score}")

    # Failure Analysis
    # Calculate absolute error
    df_val_features["abs_error"] = np.abs(
        df_val_features["target"] - df_val_features["pred_rank"]
    )

    # Correlate error with features to identify systematic issues
    features_to_analyze = ["n_code", "md_len", "tv_sim_max", "cv_sim_max"]
    correlations = (
        df_val_features[features_to_analyze + ["abs_error"]]
        .corr()["abs_error"]
        .drop("abs_error")
    )

    print("\nFailure Analysis - Correlation of Absolute Error with Features:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.8061
    if kt_score > THRESHOLD:
        print(
            f"\nValidation metric {kt_score} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Extract Test Features
        print("Extracting test features...")
        df_test_features = extractor.extract_features(
            Config.TEST_PATH, mode="test", load_cached_data=True
        )

        # Generate Submission
        generate_submission(regressor, df_test_features, Config.TEST_PATH)
    else:
        print(
            f"\nValidation metric {kt_score} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
