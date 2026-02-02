import sys
import os
import gc
import numpy as np
import pandas as pd
import warnings
import torch
from datetime import timedelta

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_map12, Timer
from library.data_loader import DataLoader
from library.visual_encoder import VisualEncoder
from library.graph_engine import GraphEngine
from library.retrieval_system import SparseRetriever
from library.feature_engineering import FeatureEngine
from library.ranker import LGBMRanker

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data Loading
    # Load processed dataframes (mapped to integers)
    # Cite debug_lesson_3: Disable cache to prevent loading stale debug data
    data_loader = DataLoader()
    train_df, val_df, test_df, articles_df, customers_df = data_loader.load_data(
        load_cached_data=False
    )

    # 3. Visual Embeddings
    # Generate or load ResNet embeddings for articles
    visual_encoder = VisualEncoder()
    embeddings = visual_encoder.generate_embeddings(load_cached_data=False)

    # 4. Graph Construction
    # Build Short-term, Long-term, and Visual sparse graphs
    # Cite debug_lesson_1: Enforce temporal cutoff to prevent data leakage.
    max_date = train_df["t_dat"].max()
    graph_cutoff = max_date - timedelta(days=28)
    print(
        f"Splitting data for Graph Construction. Max Date: {max_date}, Cutoff: {graph_cutoff}"
    )

    train_graph_df = train_df[train_df["t_dat"] <= graph_cutoff].copy()

    graph_engine = GraphEngine()
    graph_engine.build_graphs(train_graph_df, embeddings, load_cached_data=False)

    del train_graph_df
    gc.collect()

    # 5. Retrieval System
    # Initialize the retriever which uses the built graphs
    retriever = SparseRetriever()

    # 6. Feature Engineering
    feature_engine = FeatureEngine()

    # Generate Training Data (Sliding Windows)
    # This creates multiple samples per user to train the ranker
    ranker_train_df = feature_engine.generate_train_data(
        retriever,
        data_loader,
        train_df,
        articles_df,
        customers_df,
        load_cached_data=False,
    )

    # Generate Validation Data (Hold-out set)
    ranker_val_df = feature_engine.generate_val_data(
        retriever, val_df, articles_df, customers_df, load_cached_data=False
    )

    # 7. Model Training
    ranker = LGBMRanker()
    ranker.train(ranker_train_df, ranker_val_df)

    # Free up memory
    del ranker_train_df
    gc.collect()

    # 8. Validation Inference & Metric Calculation
    # Predict scores for validation candidates
    val_preds_df = ranker.predict(ranker_val_df)

    # Prepare predictions for MAP@12 calculation
    # We need to select the top 12 items per customer
    val_preds_df = val_preds_df.sort_values(
        ["customer_id", "prediction_score"], ascending=[True, False]
    )
    top_k_preds = val_preds_df.groupby("customer_id").head(12)

    # Convert article IDs to space-separated strings for the metric function
    # Note: These are still mapped integers, but consistent with the validation ground truth
    preds_formatted = (
        top_k_preds.groupby("customer_id")["article_id"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index(name="prediction")
    )

    # Prepare Ground Truth from Validation DataFrame
    # Target is the last 7 days of the validation set (as defined in FeatureEngine logic)
    max_val_date = val_df["t_dat"].max()
    target_start = max_val_date - pd.Timedelta(days=Config.SLIDING_WINDOW_SIZE)
    val_target_df = val_df[val_df["t_dat"] >= target_start].copy()

    # Calculate MAP@12
    map12 = calculate_map12(preds_formatted, val_target_df)
    print(f"Final Validation Metric: {map12:.16f}")

    # 9. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate error magnitude. Since scores are raw logits, we apply sigmoid to get probabilities.
    val_preds_df["prob"] = 1 / (1 + np.exp(-val_preds_df["prediction_score"]))
    val_preds_df["error"] = (val_preds_df["label"] - val_preds_df["prob"]).abs()

    # correlate error with numerical features
    numeric_cols = val_preds_df.select_dtypes(include=[np.number]).columns
    exclude_cols = [
        "customer_id",
        "article_id",
        "label",
        "prediction_score",
        "prob",
        "error",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = (
        val_preds_df[feature_cols]
        .corrwith(val_preds_df["error"])
        .sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.head(10))

    # 10. Submission
    THRESHOLD = 0.026059042

    if map12 > THRESHOLD:
        print(
            f"Validation score {map12:.6f} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Generate Test Features
        ranker_test_df = feature_engine.generate_test_data(
            retriever,
            train_df,
            val_df,
            test_df,
            articles_df,
            customers_df,
            load_cached_data=False,
        )

        # Predict on Test Data
        test_preds_df = ranker.predict(ranker_test_df)

        # Generate Submission File
        # This handles mapping back to original IDs and filling with popular items
        ranker.generate_submission(test_preds_df, test_df, Config.CACHE_ARTICLE_MAP)

    else:
        print(
            f"Validation score {map12:.6f} does not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
