import pandas as pd
import numpy as np
import os
import random
import warnings

# Import provided library modules
from library.utils import Timer, calculate_map12
from library.data_factory import DataManager
from library.tmvc_model import TMVCRecommender

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)

    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=== Starting H&M Recommendation Task Demo ===")

    # ---------------------------------------------------------
    # 1. Data Loading & Preparation
    # ---------------------------------------------------------
    with Timer("Data Preparation"):
        # Initialize DataManager
        dm = DataManager(
            input_dir=INPUT_DIR, metadata_dir=METADATA_DIR, cache_dir=CACHE_DIR
        )

        # Load full transaction history
        # This handles type optimization and caching automatically
        print("Loading transactions...")
        df_full = dm.load_data(load_cached=True)

        # Load auxiliary metadata for mappings
        print("Loading articles and customers...")
        articles_df = pd.read_csv(
            os.path.join(INPUT_DIR, "articles.csv"), dtype={"article_id": "int32"}
        )
        customers_df = pd.read_csv(os.path.join(INPUT_DIR, "customers.csv"))

        # Perform Time-Based Split (Train on T-1, Validate on T)
        # We use the last 7 days for validation
        train_df, val_df = dm.get_time_split(df_full, days=7)

        # Extract TMVC Subsets
        # Structure Window: Used for learning Item-Item Similarity (e.g., last 4 weeks)
        # Velocity Window: Used for trend modulation (e.g., last 1 week)
        # Full History: Used for User Habit profiling
        print("Generating TMVC temporal subsets...")
        df_structure, df_velocity, df_history = dm.get_windowed_subsets(
            train_df,
            structure_weeks=4,  # Using 4 weeks for demo speed (standard is often 16)
            velocity_weeks=1,
        )

    # ---------------------------------------------------------
    # 2. Model Initialization & Fitting
    # ---------------------------------------------------------
    with Timer("Model Training"):
        model = TMVCRecommender(cache_dir=CACHE_DIR)

        print("Fitting TMVC model...")
        # fit() builds the interaction matrix and similarity matrix
        # We set load_cached=False to demonstrate the computation logic
        model.fit(
            df_structure=df_structure,
            df_velocity=df_velocity,
            df_full_history=df_history,
            articles_df=articles_df,
            customers_df=customers_df,
            load_cached=False,
        )

    # ---------------------------------------------------------
    # 3. Validation & Scoring
    # ---------------------------------------------------------
    with Timer("Validation"):
        print("Predicting for validation customers...")
        # Identify customers in the validation set
        val_customers = val_df[["customer_id"]].drop_duplicates()

        # Generate predictions
        # Note: We use the training history (df_history) to predict validation behavior
        val_preds = model.predict(val_customers, df_history, batch_size=5000)

        # Calculate MAP@12
        score = calculate_map12(val_df, val_preds)
        print(f"Validation MAP@12 Score: {score:.6f}")

        # Basic assertion to verify the model is learning (random guess is ~0.0)
        if score <= 0.001:
            raise AssertionError(
                f"Model score {score} is too low. Check implementation."
            )

    # ---------------------------------------------------------
    # 4. Final Submission Generation
    # ---------------------------------------------------------
    with Timer("Submission Generation"):
        print("Loading test customer list...")
        test_customers = dm.load_test_customers()

        print(f"Generating predictions for {len(test_customers)} test customers...")
        # In a real scenario, we might retrain on full data (train + val).
        # Here we use the trained model to predict for the test set.
        final_preds = model.predict(test_customers, df_history, batch_size=5000)

        # Save submission
        submission_path = os.path.join(WORKING_DIR, "submission.csv")
        final_preds.to_csv(submission_path, index=False)
        print(f"Submission saved to: {submission_path}")

        # Verify output format
        assert "customer_id" in final_preds.columns
        assert "prediction" in final_preds.columns
        assert len(final_preds) == len(test_customers)

        # Check one prediction format
        sample_pred = final_preds.iloc[0]["prediction"]
        assert isinstance(sample_pred, str)
        # Should be space-separated article IDs
        assert len(sample_pred.split()) <= 12

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
