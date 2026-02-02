import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

# Import library modules
from library.config import Config
from library.utils import set_seed, reduce_mem_usage
from library.metrics import calculate_map12
from library.data_loader import IdMapper, load_filtered_transactions
from library.model import TrendRepurchaseCascade


def demo_utils():
    print("\n=== 1. Demonstrating Utilities ===")

    # Test Reproducibility
    set_seed(42)
    rand_val_1 = np.random.rand()
    set_seed(42)
    rand_val_2 = np.random.rand()
    assert rand_val_1 == rand_val_2, "set_seed did not ensure reproducibility"
    print("Reproducibility check passed.")

    # Test Memory Reduction
    df = pd.DataFrame(
        {
            "a": np.random.rand(100).astype(np.float64),
            "b": np.random.randint(0, 100, 100).astype(np.int64),
        }
    )
    start_mem = df.memory_usage().sum()
    print(f"Original memory usage: {start_mem} bytes")

    df_reduced = reduce_mem_usage(df, verbose=True)
    end_mem = df_reduced.memory_usage().sum()

    assert df_reduced["a"].dtype == np.float16 or df_reduced["a"].dtype == np.float32
    assert df_reduced["b"].dtype == np.int8 or df_reduced["b"].dtype == np.int16
    assert end_mem < start_mem, "Memory usage was not reduced"
    print("Memory reduction logic verified.")


def demo_metrics():
    print("\n=== 2. Demonstrating Metrics (MAP@12) ===")

    # Scenario:
    # Customer 'u1' bought items [10, 20]
    # Prediction: '10 30 20'
    # Precision@1 (10): Hit (1/1)
    # Precision@2 (30): Miss
    # Precision@3 (20): Hit (2/3)
    # AP = (1/1 + 2/3) / min(2, 12) = (1 + 0.666) / 2 = 0.8333...

    val_df = pd.DataFrame({"customer_id": ["u1", "u1"], "article_id": [10, 20]})

    sub_df = pd.DataFrame({"customer_id": ["u1"], "prediction": ["10 30 20"]})

    score = calculate_map12(val_df, sub_df, k=12)
    expected_score = (1.0 + 2 / 3) / 2.0

    print(f"Calculated MAP@12: {score:.4f}")
    print(f"Expected MAP@12:   {expected_score:.4f}")

    assert np.isclose(score, expected_score, atol=1e-4), "MAP@12 calculation incorrect"
    print("Metric calculation verified.")


def demo_data_loader():
    print("\n=== 3. Demonstrating Data Loader ===")

    # Create dummy data in working directory to avoid loading large files
    working_dir = Config.WORKING_DIR
    os.makedirs(working_dir, exist_ok=True)

    dummy_cust_path = os.path.join(working_dir, "dummy_customers.csv")
    dummy_art_path = os.path.join(working_dir, "dummy_articles.csv")
    dummy_trans_path = os.path.join(working_dir, "dummy_transactions.csv")

    # 1. Create Dummy Metadata
    customers = pd.DataFrame({"customer_id": ["c1", "c2", "c3"]})
    articles = pd.DataFrame({"article_id": [101, 102, 103, 104]})  # int32

    customers.to_csv(dummy_cust_path, index=False)
    articles.to_csv(dummy_art_path, index=False)

    # 2. Create Dummy Transactions
    # c1 buys 101 today, c2 buys 102 yesterday
    today = datetime.now()
    dates = [today, today - timedelta(days=1), today - timedelta(days=10)]

    transactions = pd.DataFrame(
        {
            "t_dat": dates,
            "customer_id": ["c1", "c2", "c1"],
            "article_id": [101, 102, 103],
        }
    )
    transactions.to_csv(dummy_trans_path, index=False)

    # 3. Test IdMapper
    print("Testing IdMapper...")
    mapper = IdMapper()
    mapper.fit(customers, articles)

    # Transform
    c_idx = mapper.transform_customers(pd.DataFrame({"customer_id": ["c1"]}))
    a_idx = mapper.transform_articles(pd.DataFrame({"article_id": [101]}))

    assert len(c_idx) == 1
    assert len(a_idx) == 1
    print(f"Mapped c1 -> {c_idx.iloc[0]}, 101 -> {a_idx.iloc[0]}")

    # Inverse Transform
    orig_art = mapper.inverse_transform_articles([a_idx.iloc[0]])
    assert orig_art[0] == 101, "Inverse transform failed"

    # Save/Load
    cache_dir = os.path.join(working_dir, "test_cache")
    mapper.save(cache_dir)
    mapper_loaded = IdMapper()
    mapper_loaded.load(cache_dir)
    assert mapper_loaded.customer_to_idx == mapper.customer_to_idx
    print("IdMapper Save/Load verified.")

    # 4. Test load_filtered_transactions
    print("Testing load_filtered_transactions...")
    # Use a unique weeks number to avoid colliding with real cache
    df_loaded, mapper_loaded = load_filtered_transactions(
        train_files=[dummy_trans_path],
        customers_path=dummy_cust_path,
        articles_path=dummy_art_path,
        weeks=2,  # Should filter out the transaction from 10 days ago if weeks=1, but let's keep it safe
        load_cached_data=False,
    )

    assert "days_elapsed" in df_loaded.columns
    assert len(df_loaded) == 3, f"Expected 3 transactions, got {len(df_loaded)}"
    # Verify mapping is applied (IDs should be integers, likely small ones)
    assert df_loaded["customer_id"].dtype in [np.int8, np.int16, np.int32, np.int64]

    # Verify days_elapsed logic: Most recent date should be 0
    assert df_loaded["days_elapsed"].min() == 0

    print("Data loading and processing verified.")

    return df_loaded, mapper_loaded


def demo_model(df_train):
    print("\n=== 4. Demonstrating Model (TrendRepurchaseCascade) ===")

    # df_train has columns: customer_id, article_id, days_elapsed
    # Let's verify the model logic.
    # Logic:
    # 1. Repurchase: User history is prioritized.
    # 2. Trend: Global popularity (decayed) fills the rest.

    model = TrendRepurchaseCascade(top_k=3, decay_alpha=1.0)
    model.fit(df_train)

    # Check Global Trend
    # In our dummy data (from demo_data_loader):
    # c1 bought 101 (day 0) and 103 (day 10)
    # c2 bought 102 (day 1)
    #
    # Scores:
    # 101: 1 / (0 + 1) = 1.0
    # 102: 1 / (1 + 1) = 0.5
    # 103: 1 / (10 + 1) = 0.09
    # Trend order should be: 101, 102, 103

    trend = model.global_trend
    print(f"Global Trend Indices: {trend}")

    # Since we don't know the exact mapped indices without the mapper,
    # we rely on the fact that 101 was most recent -> highest score.
    # We can check predictions.

    # Predict for c1
    # History for c1: 101 (day 0), 103 (day 10). Order: 101, 103.
    # Prediction should be: [101, 103, 102] (102 filled from trend)

    # Get mapped ID for c1
    c1_id = df_train[df_train["days_elapsed"] == 0]["customer_id"].iloc[0]

    preds = model.predict([c1_id])
    pred_items = preds[0]

    print(f"Predictions for c1 (mapped): {pred_items}")

    assert len(pred_items) == 3, "Prediction length mismatch"

    # Verify history is first
    # The first item should be the one bought at day 0 (101)
    mapped_101 = df_train[(df_train["days_elapsed"] == 0)]["article_id"].iloc[0]
    assert pred_items[0] == mapped_101, "Most recent purchase not first in prediction"

    print("Model logic verified.")


def cleanup():
    print("\n=== Cleaning up temporary files ===")
    working_dir = Config.WORKING_DIR
    try:
        if os.path.exists(os.path.join(working_dir, "dummy_customers.csv")):
            os.remove(os.path.join(working_dir, "dummy_customers.csv"))
        if os.path.exists(os.path.join(working_dir, "dummy_articles.csv")):
            os.remove(os.path.join(working_dir, "dummy_articles.csv"))
        if os.path.exists(os.path.join(working_dir, "dummy_transactions.csv")):
            os.remove(os.path.join(working_dir, "dummy_transactions.csv"))
        if os.path.exists(os.path.join(working_dir, "test_cache")):
            shutil.rmtree(os.path.join(working_dir, "test_cache"))
        # Clean up the parquet cache created by load_filtered_transactions
        # It creates transactions_w2.parquet in Config.CACHE_DIR
        cache_file = os.path.join(Config.CACHE_DIR, "transactions_w2.parquet")
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except Exception as e:
        print(f"Cleanup warning: {e}")
    print("Cleanup complete.")


if __name__ == "__main__":
    try:
        demo_utils()
        demo_metrics()
        df_train, mapper = demo_data_loader()
        demo_model(df_train)
        print("\nAll demonstrations and assertions passed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise
    finally:
        cleanup()
