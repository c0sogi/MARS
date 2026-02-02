import os
import sys
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_logger, Timer
from library.data import NotebookLoader
from library.features import FeaturePipeline
from library.models import Stage2LGBM, SubmissionGenerator
from library.metrics import compute_score

# Suppress warnings and tqdm if possible to keep output clean
import warnings

warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.RANDOM_STATE)
    logger = get_logger("runfile")

    # 2. Data Loading
    loader = NotebookLoader()

    # Load Train and Validation Data
    # Using load_cached_data=True to utilize pre-processed parquet files if they exist
    df_md_train, df_nb_train = loader.load_dataset("train", load_cached_data=True)
    df_md_val, df_nb_val = loader.load_dataset("val", load_cached_data=True)

    # 3. Feature Engineering
    fp = FeaturePipeline()

    # Fit pipeline on training data (TF-IDF, SVD, Global Ridge)
    fp.fit_pipeline(df_md_train)

    # Transform datasets to get Stage 2 features
    # This generates OOF Ridge preds for train, and global Ridge preds for val
    df_train_feats = fp.transform_pipeline(
        df_md_train, df_nb_train, "train", load_cached_data=True
    )
    df_val_feats = fp.transform_pipeline(
        df_md_val, df_nb_val, "val", load_cached_data=True
    )

    # 4. Stage 2 Training (LightGBM)
    model = Stage2LGBM()
    model.train(df_train_feats, df_val_feats)

    # 5. Validation Inference & Metric Calculation
    logger.info("Running validation inference...")
    val_preds = model.predict(df_val_feats)
    df_val_feats["pred_rank"] = val_preds

    # Reconstruct orders for validation to compute Kendall Tau
    # We need to combine code cells (fixed ranks) and markdown cells (predicted ranks)
    nb_code_map = df_nb_val.set_index("notebook_id")["code_ids"].to_dict()
    val_grouped = df_val_feats.groupby("notebook_id")

    val_predictions = {}
    all_val_ids = df_nb_val["notebook_id"].unique()

    for nb_id in all_val_ids:
        # Code cells: fixed ranks 0.0 to 1.0
        code_ids = nb_code_map.get(nb_id, [])
        n_code = len(code_ids)
        cells = []

        if n_code > 0:
            if n_code == 1:
                ranks = [0.0]
            else:
                ranks = np.linspace(0, 1, n_code)

            for cid, r in zip(code_ids, ranks):
                cells.append((cid, r))

        # Markdown cells: predicted ranks
        if nb_id in val_grouped.groups:
            g = val_grouped.get_group(nb_id)
            for _, row in g.iterrows():
                cells.append((row["cell_id"], row["pred_rank"]))

        # Sort by rank
        cells.sort(key=lambda x: x[1])

        # Create space-delimited string
        val_predictions[nb_id] = " ".join([c[0] for c in cells])

    # Load Ground Truth from metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val_metadata.csv")
    df_val_meta = pd.read_csv(val_meta_path)

    # Compute Metric
    final_score = compute_score(df_val_meta, val_predictions)
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    logger.info("Performing failure analysis...")
    # Calculate absolute error
    df_val_feats["error"] = np.abs(df_val_feats["pred_rank"] - df_val_feats["target"])

    # Correlate error with features
    # Filter for numeric columns only
    numeric_cols = df_val_feats.select_dtypes(include=[np.number]).columns
    # Exclude target and error itself from correlation check
    cols_to_check = [
        c
        for c in numeric_cols
        if c not in ["target", "error", "rank", "original_index"]
    ]

    correlations = (
        df_val_feats[cols_to_check]
        .corrwith(df_val_feats["error"])
        .sort_values(ascending=False)
    )

    print("Correlation between Error and Features (Top 5 Positive):")
    print(correlations.head(5))
    print("Correlation between Error and Features (Top 5 Negative):")
    print(correlations.tail(5))

    # 7. Submission
    THRESHOLD = 0.7959051868218839

    if final_score > THRESHOLD:
        logger.info(
            f"Validation score {final_score} > {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        df_md_test, df_nb_test = loader.load_dataset("test", load_cached_data=True)

        # Generate Test Features
        df_test_feats = fp.transform_pipeline(
            df_md_test, df_nb_test, "test", load_cached_data=True
        )

        # Predict
        test_preds = model.predict(df_test_feats)
        df_test_feats["pred_rank"] = test_preds

        # Generate Submission File
        sub_gen = SubmissionGenerator()
        sub_gen.generate(df_test_feats, df_nb_test)

    else:
        logger.info(
            f"Validation score {final_score} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
