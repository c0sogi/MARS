import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.model_trainer import ModelTrainer
from library.feature_generator import FeatureFactory
from library.utils import kendall_tau, validate_ranks
from library.post_processor import RankSorter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_gpu():
    """Detects GPU and updates LightGBM configuration."""
    if torch.cuda.is_available():
        print("GPU detected. Configuring LightGBM to utilize GPU.")
        # Update Config params for GPU usage
        Config.LGBM_PARAMS["device"] = "gpu"
        Config.LGBM_PARAMS["gpu_platform_id"] = 0
        Config.LGBM_PARAMS["gpu_device_id"] = 0
    else:
        print("No GPU detected. Using CPU.")


def subsample_training_data(n_samples=20000):
    """
    Subsamples the training metadata to ensure fast baseline execution.
    Samples by ancestor_id to maintain data integrity.
    """
    print(f"Subsampling training data to {n_samples} notebooks...")
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    if len(df) <= n_samples:
        print("Dataset smaller than requested sample size. Using full dataset.")
        return

    # Get unique ancestors
    if "ancestor_id" in df.columns:
        ancestors = df["ancestor_id"].unique()
        # Sample ancestors
        if len(ancestors) > n_samples:
            # Assuming roughly 1 notebook per ancestor, sample n_samples ancestors
            # Using numpy choice for speed
            sampled_ancestors = np.random.choice(
                ancestors, size=n_samples, replace=False
            )
            df_sampled = df[df["ancestor_id"].isin(sampled_ancestors)]
        else:
            # Fallback if fewer ancestors than samples (unlikely given dataset stats)
            df_sampled = df.sample(n=n_samples, random_state=Config.RANDOM_STATE)
    else:
        df_sampled = df.sample(n=n_samples, random_state=Config.RANDOM_STATE)

    # Save sampled metadata
    sampled_path = os.path.join(Config.METADATA_DIR, "train_metadata_sampled.csv")
    df_sampled.to_csv(sampled_path, index=False)

    # Update Config to point to sampled file
    Config.TRAIN_METADATA_PATH = sampled_path
    print(f"Sampled metadata saved to {sampled_path}. Rows: {len(df_sampled)}")


def run_validation_inference(trainer, ridge_model, lgbm_model):
    """
    Runs inference on the validation set and computes Kendall Tau.
    """
    print("\n=== Running Validation Inference ===")
    factory = FeatureFactory()

    # 1. Generate Stage 1 Features for Validation
    print("Generating Stage 1 Validation Features...")
    X_val, y_val, groups_val = factory.build_stage1_dataset(
        split="val", load_cached_data=True
    )

    # 2. Predict with Ridge
    print("Predicting with Ridge...")
    ridge_preds = ridge_model.predict(X_val)

    # 3. Generate Stage 2 Features for Validation
    print("Generating Stage 2 Validation Features...")
    df_val_s2, y_val_s2, groups_s2 = factory.build_stage2_dataset(
        split="val", ridge_preds=ridge_preds, load_cached_data=True
    )

    # 4. Predict with LightGBM
    print("Predicting with LightGBM...")
    lgbm_preds = lgbm_model.predict(df_val_s2)
    lgbm_preds = validate_ranks(lgbm_preds)

    # 5. Reconstruct Notebook Orders
    print("Reconstructing Validation Notebook Orders...")
    # Load raw validation data to get code cells and structure
    df_md_val, df_code_val = factory.loader.load_data(
        split="val", load_cached_data=True
    )

    # Assign predictions
    df_md_val["pred_rank"] = lgbm_preds

    # Sort
    sorter = RankSorter()
    val_submission_df = sorter.sort_notebooks(
        df_md_val, df_code_val, load_cached_data=False
    )

    # 6. Compute Metric
    print("Computing Kendall Tau...")
    # Load Ground Truth
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create dictionaries for fast lookup
    ground_truths = (
        df_val_meta.set_index("id")["cell_order"].apply(lambda x: x.split()).to_dict()
    )
    predictions = (
        val_submission_df.set_index("id")["cell_order"]
        .apply(lambda x: x.split())
        .to_dict()
    )

    # Align lists
    gt_list = []
    pred_list = []
    ids = list(ground_truths.keys())

    for nb_id in ids:
        if nb_id in predictions:
            gt_list.append(ground_truths[nb_id])
            pred_list.append(predictions[nb_id])

    score = kendall_tau(gt_list, pred_list)
    print(f"Final Validation Metric: {score}")

    return score, df_val_s2, y_val_s2, lgbm_preds


def perform_failure_analysis(df_features, y_true, y_pred):
    """
    Analyzes correlation between features and prediction error.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Absolute Error
    errors = np.abs(y_true - y_pred)

    # Create a Series for correlation
    error_series = pd.Series(errors, name="MAE")

    # Compute correlations
    # We drop non-numeric columns if any (though stage2 features should be all numeric)
    numeric_feats = df_features.select_dtypes(include=[np.number])
    correlations = numeric_feats.corrwith(error_series).sort_values(
        ascending=False, key=abs
    )

    print("Top 10 Features Correlated with Error (MAE):")
    print(correlations.head(10))


def main():
    # 1. Setup
    set_seed(Config.RANDOM_STATE)
    configure_gpu()

    # 2. Subsample Data for Fast Baseline
    # Limit to 20,000 notebooks to ensure completion within time limits
    subsample_training_data(n_samples=20000)

    # 3. Train Models
    trainer = ModelTrainer()
    # load_cached_data=True allows using pre-computed features if they exist,
    # but since we changed metadata path, it will likely re-compute for the sampled set.
    ridge_model, lgbm_model = trainer.train(load_cached_data=True)

    # 4. Validation & Metric Calculation
    val_score, df_val_features, y_val_true, y_val_pred = run_validation_inference(
        trainer, ridge_model, lgbm_model
    )

    # 5. Failure Analysis
    perform_failure_analysis(df_val_features, y_val_true, y_val_pred)

    # 6. Submission Logic
    # Threshold from instructions
    THRESHOLD = 0.7959051868218839

    if val_score > THRESHOLD:
        print(
            f"\nValidation score {val_score:.6f} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        trainer.generate_submission(load_cached_data=True)
        print("Submission generation complete.")
    else:
        print(
            f"\nValidation score {val_score:.6f} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
