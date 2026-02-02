import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime, timedelta

# Import library components
from library.config import Config
from library.data_utils import DataManager
from library.similarity_engine import SimilarityEngine
from library.trend_engine import TrendEngine
from library.smdc_model import SMDCRecommender
from library.metrics import calculate_map12

# --- Configuration & Mock Data Setup ---


def setup_environment():
    """
    Sets up a temporary working environment with mock data to ensure
    the demonstration runs quickly and verifies logic without processing
    the massive full dataset.
    """
    # Define paths
    base_dir = "./working/demo_run"
    input_dir = os.path.join(base_dir, "input")
    cache_dir = os.path.join(base_dir, "cache")
    sub_dir = os.path.join(base_dir, "submission")

    # Clean up previous runs
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    os.makedirs(input_dir)
    os.makedirs(cache_dir)
    os.makedirs(sub_dir)

    # Monkey-patch Config to use our demo directories
    print(f"Configuring environment at {base_dir}...")
    Config.INPUT_DIR = input_dir
    Config.CACHE_DIR = cache_dir
    Config.SUBMISSION_DIR = sub_dir
    Config.SUBMISSION_PATH = os.path.join(sub_dir, "submission.csv")

    # --- Generate Mock Data ---

    # 1. Articles (50 items)
    # columns: article_id, product_code, ...
    n_articles = 50
    article_ids = np.arange(1000, 1000 + n_articles)
    product_codes = np.random.randint(
        100, 110, size=n_articles
    )  # Shared codes for variant sim

    articles_df = pd.DataFrame(
        {
            "article_id": article_ids,
            "product_code": product_codes,
            "prod_name": [f"Prod_{i}" for i in range(n_articles)],
            "product_type_no": np.random.randint(0, 5, size=n_articles),
            "detail_desc": ["desc"] * n_articles,
        }
    )
    articles_df.to_csv(os.path.join(input_dir, "articles.csv"), index=False)

    # 2. Customers (100 users)
    n_customers = 100
    customer_ids = [f"user_{i:04d}" for i in range(n_customers)]
    ages = np.random.randint(18, 70, size=n_customers)
    # Introduce some missing ages
    ages[0:5] = -1

    customers_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": [a if a != -1 else np.nan for a in ages],
            "postal_code": ["0000"] * n_customers,
        }
    )
    customers_df.to_csv(os.path.join(input_dir, "customers.csv"), index=False)

    # 3. Transactions (Generate history)
    # Create transactions over the last 10 weeks
    n_trans = 2000
    t_cust = np.random.choice(customer_ids, n_trans)
    t_art = np.random.choice(article_ids, n_trans)

    # Dates: from today back to 10 weeks ago
    max_date = datetime(2020, 9, 22)
    days_offset = np.random.randint(0, 70, n_trans)
    t_dates = [max_date - timedelta(days=int(d)) for d in days_offset]

    trans_df = pd.DataFrame(
        {
            "t_dat": t_dates,
            "customer_id": t_cust,
            "article_id": t_art,
            "price": np.random.rand(n_trans).astype(np.float32),
            "sales_channel_id": np.random.choice([1, 2], n_trans),
        }
    )
    trans_df.to_csv(os.path.join(input_dir, "transactions_train.csv"), index=False)

    # 4. Sample Submission
    # Just the customer list
    sub_df = pd.DataFrame(
        {"customer_id": customer_ids, "prediction": [""] * n_customers}
    )
    sub_df.to_csv(os.path.join(input_dir, "sample_submission.csv"), index=False)

    print("Mock data generated successfully.")
    return article_ids, customer_ids


# --- Test Functions ---


def test_data_manager():
    print("\n=== Testing DataManager ===")
    dm = DataManager()

    # Test loading for validation mode
    # load_cached_data=False forces processing from the raw CSVs we just made
    train_df, test_df, cust_df, art_df, indexer = dm.load_data(
        validate=True, load_cached_data=False
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Assertions
    assert not train_df.empty, "Train DataFrame should not be empty"
    assert not test_df.empty, "Test DataFrame should not be empty"
    assert "days_elapsed" in train_df.columns, "days_elapsed column missing"
    assert len(indexer.user_to_idx) > 0, "Indexer failed to map users"
    assert len(indexer.item_to_idx) > 0, "Indexer failed to map items"

    # Check Age Binning in customers
    assert "age_bin" in cust_df.columns, "Customer age binning failed"

    print("DataManager logic verified.")
    return train_df, art_df, indexer


def test_similarity_engine(train_df, art_df, indexer):
    print("\n=== Testing SimilarityEngine ===")
    sim_engine = SimilarityEngine()

    # 1. Build User-Item Matrix
    U = sim_engine.build_user_item_matrix(train_df, indexer, load_cached_data=False)

    assert sp.issparse(U), "U matrix should be sparse"
    assert U.shape == (
        len(indexer.user_to_idx),
        len(indexer.item_to_idx),
    ), f"U shape mismatch. Expected {(len(indexer.user_to_idx), len(indexer.item_to_idx))}, got {U.shape}"

    # 2. Build Hybrid Matrix
    S_hybrid = sim_engine.build_hybrid_matrix(
        U, art_df, indexer, load_cached_data=False
    )

    assert sp.issparse(S_hybrid), "S_hybrid should be sparse"
    assert S_hybrid.shape == (
        len(indexer.item_to_idx),
        len(indexer.item_to_idx),
    ), "S_hybrid shape mismatch"

    # Check diagonal is zero (as per logic)
    diag_sum = S_hybrid.diagonal().sum()
    assert diag_sum == 0, "Diagonal of similarity matrix should be zero"

    print("SimilarityEngine logic verified.")
    return U, S_hybrid


def test_trend_engine(train_df, customers_df, indexer):
    print("\n=== Testing TrendEngine ===")
    trend_engine = TrendEngine()

    # 1. Global Trends
    global_trends = trend_engine.get_global_trends(
        train_df, indexer, load_cached_data=False
    )

    assert isinstance(global_trends, np.ndarray), "Global trends should be numpy array"
    assert len(global_trends) == len(
        indexer.item_to_idx
    ), "Global trends dimension mismatch"
    assert global_trends.max() <= Config.SCALE_GLOBAL, "Global trends scaling issue"

    # 2. Cohort Trends
    cohort_trends = trend_engine.get_cohort_trends(
        train_df, customers_df, indexer, load_cached_data=False
    )

    assert isinstance(cohort_trends, dict), "Cohort trends should be a dictionary"
    assert len(cohort_trends) > 0, "No cohort trends calculated"
    # Check a random key
    key = list(cohort_trends.keys())[0]
    assert len(cohort_trends[key]) == len(
        indexer.item_to_idx
    ), "Cohort vector dimension mismatch"

    print("TrendEngine logic verified.")


def test_full_pipeline_and_submission():
    print("\n=== Testing Full SMDCRecommender Pipeline ===")

    # Instantiate model
    model = SMDCRecommender()

    # Fit model (Validate=False means submission mode)
    # This will use the cached data generated in previous steps if we set load_cached_data=True,
    # but to be safe and test integration, we let it reload from the cache we just created.
    model.fit(validate=False, load_cached_data=True)

    # Verify internal state
    assert model.U is not None, "Model U matrix not initialized"
    assert model.S_hybrid is not None, "Model S_hybrid not initialized"

    # Generate Submission
    model.generate_submission()

    # Check output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "customer_id" in sub_df.columns and "prediction" in sub_df.columns
    ), "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check prediction format (space separated string)
    sample_pred = sub_df.iloc[0]["prediction"]
    assert isinstance(sample_pred, str), "Prediction is not a string"
    items = sample_pred.split()
    assert len(items) <= 12, "Predicted more than 12 items"

    print("Full Pipeline verified. Submission generated.")


def test_metrics():
    print("\n=== Testing Metrics ===")

    # Case 1: Perfect Match
    gt = {"user1": [101, 102, 103]}
    preds = {"user1": "0000000101 0000000102 0000000103"}
    score = calculate_map12(gt, preds)
    # MAP should be 1.0
    assert np.isclose(score, 1.0), f"Expected MAP 1.0, got {score}"

    # Case 2: No Match
    gt = {"user1": [101]}
    preds = {"user1": "0000000999"}
    score = calculate_map12(gt, preds)
    assert score == 0.0, f"Expected MAP 0.0, got {score}"

    # Case 3: Partial Match with Ranking
    # GT: 101, 102. Pred: 102 (rank 1), 101 (rank 2)
    # P@1 (102): Correct? Yes. Precision=1/1.
    # P@2 (101): Correct? Yes. Precision=2/2.
    # AP = (1/1 + 2/2) / 2 = 1.0. Wait, order matters?
    # MAP formula: sum(P(k) * rel(k)) / min(m, 12)
    # k=1: item 102. rel=1. P(1)=1/1.
    # k=2: item 101. rel=1. P(2)=2/2.
    # Sum = 2. Div by 2. Result 1.0.
    # Let's try where order is wrong (irrelevant item first)
    # Pred: 999, 101. GT: 101.
    # k=1: 999. rel=0.
    # k=2: 101. rel=1. P(2)=1/2.
    # Sum = 0.5. Div by 1. Result 0.5.

    gt = {"user1": [101]}
    preds = {"user1": "0000000999 0000000101"}
    score = calculate_map12(gt, preds)
    assert np.isclose(score, 0.5), f"Expected MAP 0.5, got {score}"

    print("Metrics logic verified.")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)

    try:
        # 1. Setup Environment & Mock Data
        setup_environment()

        # 2. Test Data Loading
        # We need the customers_df for trend engine test later, so let's load it via DataManager
        dm = DataManager()
        train_df, test_df, customers_df, articles_df, indexer = dm.load_data(
            validate=True, load_cached_data=False
        )

        # 3. Test Similarity Engine
        U, S_hybrid = test_similarity_engine(train_df, articles_df, indexer)

        # 4. Test Trend Engine
        test_trend_engine(train_df, customers_df, indexer)

        # 5. Test Full Model
        test_full_pipeline_and_submission()

        # 6. Test Metrics
        test_metrics()

        print("\nAll tests passed successfully.")

    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        raise e
