import os
import pandas as pd
import numpy as np
from library.config import SUBMISSION_PATH, SEED
from library.utils import set_seed, jaccard
from library.data_loader import load_datasets
from library.model import SentimentRelevanceModel


def run_demo():
    # 1. Setup
    print("1. Setting up environment...")
    set_seed(SEED)

    # 2. Data Loading
    print("\n2. Loading datasets...")
    # We use debug=False to run on the full dataset because the statistical model is very fast.
    # This ensures we generate a valid submission for the full test set.
    # load_cached_data=False ensures we process the raw CSVs from metadata.
    train_df, val_df, test_df = load_datasets(load_cached_data=False, debug=False)

    print(f"   Train shape: {train_df.shape}")
    print(f"   Val shape:   {val_df.shape}")
    print(f"   Test shape:  {test_df.shape}")

    # Assertions to verify data loading
    assert not train_df.empty, "Training dataframe is empty"
    assert not val_df.empty, "Validation dataframe is empty"
    assert not test_df.empty, "Test dataframe is empty"
    assert "text" in train_df.columns
    assert "sentiment" in train_df.columns
    assert "selected_text" in train_df.columns

    # 3. Utility Verification (Jaccard)
    print("\n3. Verifying Jaccard utility...")
    s1 = "hello world"
    s2 = "hello world"
    s3 = "hello"
    s4 = "goodbye"

    score_perfect = jaccard(s1, s2)
    score_partial = jaccard(s1, s3)
    score_none = jaccard(s1, s4)

    print(f"   Jaccard('{s1}', '{s2}') = {score_perfect}")
    print(f"   Jaccard('{s1}', '{s3}') = {score_partial}")
    print(f"   Jaccard('{s1}', '{s4}') = {score_none}")

    assert score_perfect == 1.0, "Jaccard logic error: Identical strings should be 1.0"
    assert 0.0 < score_partial < 1.0, "Jaccard logic error: Partial match failed"
    assert score_none == 0.0, "Jaccard logic error: No overlap should be 0.0"

    # 4. Model Instantiation and Training
    print("\n4. Instantiating and fitting model...")
    model = SentimentRelevanceModel()

    # Force re-computation of weights (load_cached_data=False) to demonstrate the fitting logic
    # and ensure the code doesn't rely on pre-existing cache files for this run.
    model.fit(train_df, load_cached_data=False)

    # Verify model state (internal weights)
    assert len(model.pos_weights) > 0, "Positive weights not computed"
    assert len(model.neg_weights) > 0, "Negative weights not computed"
    print(
        f"   Learned {len(model.pos_weights)} positive tokens and {len(model.neg_weights)} negative tokens."
    )

    # 5. Evaluation
    print("\n5. Evaluating on validation set...")
    val_score = model.evaluate(val_df)
    print(f"   Validation Score: {val_score:.4f}")

    # Basic sanity check on score
    # (Statistical baselines for this task usually score > 0.5 due to the high volume of neutral tweets)
    assert val_score > 0.4, f"Model performance is suspiciously low: {val_score}"

    # 6. Submission Generation
    print("\n6. Generating submission...")
    model.generate_submission(test_df)

    # 7. Verify Submission
    print("\n7. Verifying submission file...")
    assert os.path.exists(SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(SUBMISSION_PATH)
    print(f"   Submission shape: {sub_df.shape}")
    print(f"   First few rows:\n{sub_df.head()}")

    # Check format requirements
    assert list(sub_df.columns) == [
        "textID",
        "selected_text",
    ], "Incorrect submission columns"
    assert len(sub_df) == len(test_df), "Submission row count mismatch"

    # Check for nulls (predictions should be strings, even if empty strings)
    assert (
        sub_df["selected_text"].isnull().sum() == 0
    ), "Submission contains null predictions"

    # Check that textID matches test set
    assert set(sub_df["textID"]) == set(
        test_df["textID"]
    ), "Submission textIDs do not match test set"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
