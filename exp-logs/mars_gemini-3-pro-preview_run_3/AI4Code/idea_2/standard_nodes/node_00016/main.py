import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.feature_extractor import EmbeddingGenerator
from library.trainer import ModelTrainer
from library.inference import InferenceEngine
from library.utils import set_seed, compute_kendall_tau
from library.dataset import NotebookSequenceDataset


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Adjust Config for a fast baseline run
    Config.NUM_EPOCHS = 3  # Reduce epochs for speed
    TRAIN_LIMIT = 10000  # Limit training samples
    VAL_LIMIT = 2000  # Limit validation samples for quick check

    set_seed(Config.SEED)
    print("Configuration set for fast baseline.")
    print(f"Training on {TRAIN_LIMIT} notebooks for {Config.NUM_EPOCHS} epochs.")

    # =========================================================================
    # 2. Feature Extraction
    # =========================================================================
    print("\n[Step 1/5] Feature Extraction...")

    # Generate features for Training set (limited)
    # We force regeneration (load_cached_data=False) to ensure we use the limit
    gen = EmbeddingGenerator()
    gen.process_split("train", load_cached_data=False, debug_limit=TRAIN_LIMIT)

    # Generate features for Validation set (limited)
    gen.process_split("val", load_cached_data=False, debug_limit=VAL_LIMIT)

    # Note: Test features are generated on-demand by InferenceEngine if needed.

    # =========================================================================
    # 3. Model Training
    # =========================================================================
    print("\n[Step 2/5] Model Training...")

    # Initialize trainer with the debug limit to load the correct subset
    trainer = ModelTrainer(debug_limit=TRAIN_LIMIT)
    trainer.train()

    # =========================================================================
    # 4. Validation & Metric Calculation
    # =========================================================================
    print("\n[Step 3/5] Validation & Metric Calculation...")

    # Load validation features to get notebook structure
    val_features = pd.read_parquet(Config.VAL_CACHE_PATH)

    # Load ground truth metadata
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Filter metadata to match the limited validation set
    val_nb_ids = val_features["notebook_id"].unique()
    val_metadata = val_metadata[val_metadata["id"].isin(val_nb_ids)]

    # Build Ground Truth Dictionary: id -> list of cell_ids
    gt_dict = {
        row["id"]: row["cell_order"].split() for _, row in val_metadata.iterrows()
    }

    # Load tabular features for validation
    val_tabular = pd.read_parquet(Config.VAL_TABULAR_PATH)

    # Predict
    print("Running inference on validation set...")
    model = lgb.Booster(model_file=Config.MODEL_SAVE_PATH)

    feature_cols = [
        "n_code",
        "sim_max",
        "sim_mean",
        "sim_std",
        "best_match_loc",
        "center_of_mass",
    ]

    preds = model.predict(val_tabular[feature_cols])

    # Map predictions
    preds_map = {}
    for idx, row in val_tabular.iterrows():
        preds_map[(row["notebook_id"], row["cell_id"])] = float(preds[idx])

    # Reconstruct Orders and Compute Metric
    predicted_orders = {}
    notebook_metrics = []

    grouped = val_features.groupby("notebook_id")

    for nb_id, group in grouped:
        if nb_id not in gt_dict:
            continue

        code_cells = group[group["cell_type"] == "code"].sort_values("rank")
        code_ids = code_cells["cell_id"].tolist()
        n_code = len(code_ids)

        cells_with_score = []
        for i, cid in enumerate(code_ids):
            cells_with_score.append((cid, float(i)))

        md_cells = group[group["cell_type"] == "markdown"]
        for _, row in md_cells.iterrows():
            cid = row["cell_id"]
            pred_rel = preds_map.get((nb_id, cid), 0.5)
            score = pred_rel * n_code
            cells_with_score.append((cid, score))

        cells_with_score.sort(key=lambda x: x[1])
        pred_order = [x[0] for x in cells_with_score]
        predicted_orders[nb_id] = pred_order

        kt = compute_kendall_tau([pred_order], [gt_dict[nb_id]])
        notebook_metrics.append(
            {"id": nb_id, "n_code": n_code, "n_md": len(md_cells), "kendall_tau": kt}
        )

    final_metric = compute_kendall_tau(predicted_orders, gt_dict)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("\n[Step 4/5] Failure Analysis...")

    if len(notebook_metrics) > 0:
        df_metrics = pd.DataFrame(notebook_metrics)
        df_metrics["error"] = 1.0 - df_metrics["kendall_tau"]

        # Calculate correlations
        # We use numpy for correlation to avoid extra dependencies if scipy is missing,
        # though standard ML environments have it.
        if len(df_metrics) > 1:
            corr_code = np.corrcoef(df_metrics["error"], df_metrics["n_code"])[0, 1]
            corr_md = np.corrcoef(df_metrics["error"], df_metrics["n_md"])[0, 1]

            print(f"Correlation (Error vs n_code): {corr_code:.4f}")
            print(f"Correlation (Error vs n_md): {corr_md:.4f}")
        else:
            print("Not enough samples for correlation analysis.")
    else:
        print("No validation metrics collected.")

    # =========================================================================
    # 6. Submission
    # =========================================================================
    print("\n[Step 5/5] Submission Generation...")

    THRESHOLD = 0.7633

    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Proceeding with submission.")

        # Initialize InferenceEngine without limits to process the full test set
        engine = InferenceEngine(debug_limit=None)
        engine.generate_submission()

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
