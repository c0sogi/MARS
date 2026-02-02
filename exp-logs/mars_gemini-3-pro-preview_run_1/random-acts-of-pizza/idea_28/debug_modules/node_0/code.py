import os
import numpy as np
import pandas as pd
import torch
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.data_utils import clean_text
from library.feature_engineering import SentimentExtractor
from library.models_mlp import PizzaDataset, DualQueryAttention
from library.train_eval import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_components():
    """
    Verifies the logic of individual components using dummy data.
    """
    print("=== Testing Individual Components ===")

    # 1. Test Data Utility: clean_text
    raw_text = "  Hello   WORLD!  "
    expected_text = "hello world!"
    cleaned = clean_text(raw_text)
    assert (
        cleaned == expected_text
    ), f"clean_text failed. Got '{cleaned}', expected '{expected_text}'"
    print("[Pass] clean_text")

    # 2. Test Feature Engineering: SentimentExtractor
    # Create a dummy dataframe with required columns
    df_dummy = pd.DataFrame(
        {
            "request_title": ["Great pizza", "Sad story"],
            "request_text_edit_aware": ["I am very happy", "I am destitute and hungry"],
        }
    )

    extractor = SentimentExtractor()
    # The transform method appends columns to the dataframe
    df_transformed = extractor.transform(df_dummy.copy())

    # Check if new columns exist and have valid values
    for col in [
        "title_polarity",
        "title_subjectivity",
        "body_polarity",
        "body_subjectivity",
    ]:
        assert col in df_transformed.columns, f"Missing column: {col}"
        assert df_transformed[col].dtype == float, f"Column {col} is not float"

    print("[Pass] SentimentExtractor")

    # 3. Test Model Component: PizzaDataset (MLP Data Loader)
    # Create dummy tensors
    batch_size = 4
    embed_dim = 16
    hist_len = 5
    meta_dim = 3

    title_emb = np.random.randn(batch_size, embed_dim).astype(np.float32)
    body_emb = np.random.randn(batch_size, embed_dim).astype(np.float32)
    hist_emb = np.random.randn(batch_size, hist_len, embed_dim).astype(np.float32)
    meta_feat = np.random.randn(batch_size, meta_dim).astype(np.float32)
    labels = np.random.randint(0, 2, batch_size).astype(np.float32)

    dataset = PizzaDataset(title_emb, body_emb, hist_emb, meta_feat, labels)
    sample = dataset[0]

    assert torch.is_tensor(sample["title"]), "Output is not a tensor"
    assert sample["title"].shape == (
        embed_dim,
    ), f"Incorrect shape: {sample['title'].shape}"
    assert "label" in sample, "Label missing from dataset item"
    print("[Pass] PizzaDataset")

    # 4. Test Model Component: DualQueryAttention (Neural Module)
    # Simulate a forward pass
    dq_attn = DualQueryAttention(embed_dim=embed_dim)

    query_tensor = torch.from_numpy(title_emb)  # (B, D)
    hist_tensor = torch.from_numpy(hist_emb)  # (B, L, D)

    context = dq_attn(query_tensor, hist_tensor)

    assert context.shape == (
        batch_size,
        embed_dim,
    ), f"Attention output shape mismatch: {context.shape}"
    print("[Pass] DualQueryAttention")


def run_fast_integration():
    """
    Runs the full training pipeline on a small subset of data to demonstrate usage and verify speed.
    """
    print("\n=== Running Fast Integration Pipeline ===")

    # Define temporary paths
    temp_dir = "./working/demo_data"
    temp_working_dir = "./working/demo_cache"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_working_dir, exist_ok=True)

    # 1. Prepare Mini-Dataset
    # We load the real metadata but sample a small balanced subset to ensure the model runs successfully.
    print("Preparing mini-dataset...")

    # Load Train
    df_train = pd.read_csv(Config.TRAIN_PATH)
    # Sample 10 positive and 10 negative examples to ensure class presence
    df_train_pos = df_train[df_train[Config.TARGET_COL] == True].head(10)
    df_train_neg = df_train[df_train[Config.TARGET_COL] == False].head(10)
    df_train_mini = pd.concat([df_train_pos, df_train_neg]).sample(
        frac=1, random_state=42
    )

    # Load Val
    df_val = pd.read_csv(Config.VAL_PATH)
    df_val_pos = df_val[df_val[Config.TARGET_COL] == True].head(5)
    df_val_neg = df_val[df_val[Config.TARGET_COL] == False].head(5)
    df_val_mini = pd.concat([df_val_pos, df_val_neg]).sample(frac=1, random_state=42)

    # Load Test
    df_test = pd.read_csv(Config.TEST_PATH).head(20)

    # Save Mini-CSVs
    mini_train_path = os.path.join(temp_dir, "train.csv")
    mini_val_path = os.path.join(temp_dir, "val.csv")
    mini_test_path = os.path.join(temp_dir, "test.csv")

    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test.to_csv(mini_test_path, index=False)

    # 2. Override Configuration
    # Point Config to our new mini files and temporary working directory
    Config.TRAIN_PATH = mini_train_path
    Config.VAL_PATH = mini_val_path
    Config.TEST_PATH = mini_test_path
    Config.WORKING_DIR = temp_working_dir
    Config.SUBMISSION_PATH = os.path.join(temp_dir, "submission.csv")

    # Optimize Hyperparameters for Speed
    Config.RF_PARAMS["n_estimators"] = 5  # Reduce trees
    Config.RF_PARAMS["n_jobs"] = 1  # Reduce overhead for small data
    Config.MLP_PARAMS["epochs"] = 1  # Single epoch
    Config.MLP_PARAMS["batch_size"] = 4  # Small batch

    # 3. Execute Pipeline
    print("Executing run_training()...")
    # load_cached_data=False ensures we process our new mini-dataset instead of loading old cache
    rf_auc, mlp_auc = run_training(load_cached_data=False, epochs=1)

    # 4. Verify Results
    print(f"\nPipeline Completed. RF AUC: {rf_auc:.4f}, MLP AUC: {mlp_auc:.4f}")

    # Check assertions
    assert 0.0 <= rf_auc <= 1.0, "RF AUC is invalid"
    assert 0.0 <= mlp_auc <= 1.0, "MLP AUC is invalid"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated"

    # Verify submission content
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(submission_df) == len(
        df_test
    ), f"Submission length mismatch. Expected {len(df_test)}, got {len(submission_df)}"
    assert (
        Config.TARGET_COL in submission_df.columns
    ), "Target column missing in submission"

    print("[Pass] Integration Pipeline")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Run tests
    test_components()
    run_fast_integration()
