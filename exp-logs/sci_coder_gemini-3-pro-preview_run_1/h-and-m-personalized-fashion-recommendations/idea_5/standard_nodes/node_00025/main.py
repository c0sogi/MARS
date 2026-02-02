import pandas as pd
import numpy as np
import os
import random
import warnings
import sys

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library import evaluation
from library import inference_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Initialization
    set_seed(config.SEED)
    warnings.filterwarnings("ignore")

    print("Initializing Stratified Vectorized Hybrid-Graph Cascade...")
    recommender = inference_model.StratifiedRecommender()

    # 2. Load Resources (Data & Matrices)
    # This leverages the caching mechanism in data_processor and graph_engine
    print("Loading data and model resources...")
    train_df, val_df, test_df = recommender.load_resources(load_cached_data=True)

    # 3. Validation Inference
    print("Running inference on validation set...")
    # Extract unique customers from validation set to predict for
    val_customers = val_df[["customer_id"]].drop_duplicates().reset_index(drop=True)

    # Generate predictions
    val_pred_df = recommender.predict(val_customers)

    # 4. Metric Calculation
    print("Calculating MAP@12...")
    map_score = evaluation.calculate_map12(val_df, val_pred_df)
    print(f"Final Validation Metric: {map_score}")

    # 5. Failure Analysis
    print("Performing failure analysis...")

    # Prepare ground truth for analysis
    # Ensure article_ids are strings for comparison
    val_df_copy = val_df.copy()
    val_df_copy["article_id"] = val_df_copy["article_id"].astype(str)

    # Group actual purchases per customer
    ground_truth = (
        val_df_copy.groupby("customer_id")["article_id"].unique().reset_index()
    )
    ground_truth.columns = ["customer_id", "actual"]

    # Merge with predictions
    analysis_df = ground_truth.merge(val_pred_df, on="customer_id", how="left")
    analysis_df["prediction"] = analysis_df["prediction"].fillna("")

    # Convert prediction string to list
    analysis_df["predicted_list"] = analysis_df["prediction"].apply(
        lambda x: x.strip().split() if x.strip() else []
    )

    # Calculate Average Precision (AP) per user
    analysis_df["ap"] = analysis_df.apply(
        lambda row: evaluation.apk(row["actual"], row["predicted_list"], k=12), axis=1
    )

    # Define Error Magnitude
    analysis_df["error_magnitude"] = 1.0 - analysis_df["ap"]

    # Load Customer Features for Correlation Analysis
    customers_path = os.path.join(config.INPUT_DIR, "customers.csv")
    customers_df = pd.read_csv(customers_path)

    # Calculate History Length (Activity Level) from Training Data
    if "customer_id" in train_df.columns:
        history_counts = (
            train_df.groupby("customer_id").size().reset_index(name="history_len")
        )
    else:
        # Fallback if customer_id is not available directly (should not happen with provided loader)
        history_counts = pd.DataFrame(columns=["customer_id", "history_len"])

    # Merge features
    user_features = customers_df[["customer_id", "age"]].copy()
    user_features = user_features.merge(history_counts, on="customer_id", how="left")

    # Handle missing values for correlation calculation
    user_features["history_len"] = user_features["history_len"].fillna(0)
    user_features["age"] = user_features["age"].fillna(user_features["age"].mean())

    # Merge features into analysis dataframe
    analysis_df = analysis_df.merge(user_features, on="customer_id", how="left")

    # Calculate Correlations
    corr_age = analysis_df["error_magnitude"].corr(analysis_df["age"])
    corr_hist = analysis_df["error_magnitude"].corr(analysis_df["history_len"])

    print("Correlation between Model Error Magnitude (1-AP) and Input Features:")
    print(f"Feature: Age, Correlation: {corr_age}")
    print(f"Feature: History Length, Correlation: {corr_hist}")

    # 6. Submission Generation
    # Threshold defined in requirements
    SUBMISSION_THRESHOLD = 0.0265060791

    if map_score > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric {map_score} exceeds threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        recommender.run_submission()
    else:
        print(
            f"Validation metric {map_score} does not exceed threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
