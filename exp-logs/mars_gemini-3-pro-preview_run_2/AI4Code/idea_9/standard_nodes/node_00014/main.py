import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.pipeline import RankingPipeline
from library.utils import compute_kendall_tau, seed_everything


def reconstruct_orders(df, preds):
    """
    Reconstructs the cell order for notebooks based on predicted ranks.
    Args:
        df (pd.DataFrame): DataFrame containing cell metadata (must include 'notebook_id', 'cell_id', 'cell_type').
        preds (np.array): Predicted ranks for markdown cells.
    Returns:
        pd.DataFrame: DataFrame with 'id' and 'cell_order' columns.
    """
    # Create a copy to avoid modifying original
    df_copy = df.copy()

    # Assign predictions to markdown cells
    md_mask = df_copy["cell_type"] == "markdown"
    df_copy.loc[md_mask, "pred_rank"] = preds

    submission_data = []

    # Group by notebook
    # Using observed=True for categorical groupby performance
    for nb_id, group in df_copy.groupby("notebook_id", observed=True):
        # Separate code and markdown
        code_cells = group[group["cell_type"] == "code"].copy()
        md_cells = group[group["cell_type"] == "markdown"].copy()

        # Assign ranks to code cells: 0.0 to 1.0 based on position
        n_code = len(code_cells)
        if n_code > 0:
            if n_code == 1:
                code_cells["pred_rank"] = 0.0
            else:
                code_cells["pred_rank"] = np.arange(n_code) / (n_code - 1)

        # Concatenate
        full_nb = pd.concat([code_cells, md_cells])

        # Sort by predicted rank
        full_nb = full_nb.sort_values("pred_rank")

        # Extract ID string
        cell_order = " ".join(full_nb["cell_id"].astype(str).tolist())

        submission_data.append({"id": nb_id, "cell_order": cell_order})

    return pd.DataFrame(submission_data)


def run_validation(pipeline):
    """
    Runs inference on the validation set and computes the metric.
    """
    print("--- Running Validation Inference ---")

    # Ensure validation data is loaded
    if pipeline.df_val is None:
        pipeline.load_data()

    df_val = pipeline.df_val

    # 1. Stage 1 Predictions (Ridge)
    val_ridge_preds = pipeline._predict_stage1(df_val)

    # 2. Stage 2 Features & Predictions (LGBM)
    # We use is_train=True to get y and groups, but we only need X for prediction
    # Note: build_stage2_dataset handles the anchor feature computation/loading
    X_val, y_val_true, _ = pipeline.build_stage2_dataset(
        df_val, "val", val_ridge_preds, is_train=True
    )

    val_final_preds = pipeline.stage2_model.predict(X_val)

    # 3. Reconstruct Orders
    val_predictions_df = reconstruct_orders(df_val, val_final_preds)

    # 4. Load Ground Truth
    # The ground truth cell_order is in the metadata file
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)
    ground_truth_df = val_metadata[["id", "cell_order"]]

    # 5. Compute Metric
    score = compute_kendall_tau(val_predictions_df, ground_truth_df)

    return score, df_val, val_final_preds, X_val


def perform_failure_analysis(df_val, val_preds, X_val):
    """
    Analyzes prediction errors on the validation set.
    """
    print("\n--- Failure Analysis ---")

    # Filter for markdown cells to align with predictions
    md_mask = df_val["cell_type"] == "markdown"
    df_md = df_val[md_mask].copy()

    # Add predictions and calculate error
    df_md["pred_rank"] = val_preds
    df_md["error"] = np.abs(df_md["norm_rank"] - df_md["pred_rank"])

    # Extract features for correlation
    # We need to reconstruct the feature dataframe or load specific features
    # X_val structure: [Ridge_Pred, Lex_Rank, Lex_Sim, Lat_Rank, Lat_Sim, LSA_0...LSA_127]

    # Extracting specific columns from X_val numpy array
    # Index 0: Ridge Prediction
    # Index 1: Lexical Anchor Rank
    # Index 2: Lexical Anchor Sim
    # Index 3: Latent Anchor Rank
    # Index 4: Latent Anchor Sim

    df_md["ridge_pred"] = X_val[:, 0]
    df_md["lex_sim"] = X_val[:, 2]
    df_md["lat_sim"] = X_val[:, 4]

    # Calculate simple text features
    df_md["char_len"] = df_md["source"].astype(str).str.len()

    # Correlations
    features_to_check = ["char_len", "lex_sim", "lat_sim", "ridge_pred"]

    print("Correlation between Error Magnitude and Features:")
    for feat in features_to_check:
        if feat in df_md.columns:
            corr, _ = pearsonr(df_md["error"], df_md[feat])
            print(f"  {feat}: {corr:.4f}")

    # Check error by rank position (are we worse at start or end?)
    corr_rank, _ = pearsonr(df_md["error"], df_md["norm_rank"])
    print(f"  norm_rank (Position): {corr_rank:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.RANDOM_SEED)
    Config.setup()

    # 2. Initialize Pipeline
    pipeline = RankingPipeline()

    # 3. Execute Training
    # This handles data loading, vectorizer fitting, Stage 1 training, and Stage 2 training
    pipeline.execute_training()

    # 4. Validation Assessment
    score, df_val, val_preds, X_val = run_validation(pipeline)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {score}")

    # 5. Failure Analysis
    perform_failure_analysis(df_val, val_preds, X_val)

    # 6. Submission Logic
    # Threshold defined in task
    THRESHOLD = 0.7959051868218839

    if score > THRESHOLD:
        print(
            f"\nValidation score ({score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        pipeline.predict_submission()
    else:
        print(
            f"\nValidation score ({score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
