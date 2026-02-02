import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics import f1_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_dataset
from library.preprocessor import (
    get_text_features,
    get_target_matrix,
    TagEncoder,
    TextVectorizer,
)
from library.mnb_model import VectorizedMNB


def main():
    # 1. Setup
    print("=== Setting up environment ===")
    set_seed(42)

    # Define limits for the demo to ensure speed
    TRAIN_LIMIT = 10000
    VAL_LIMIT = 2000
    TEST_LIMIT = 1000

    # 2. Data Loading
    print("\n=== Loading Data ===")
    # Load training data subset
    # Note: limit is set, so these won't overwrite the full dataset cache
    train_df = load_dataset("train", limit=TRAIN_LIMIT, load_cached_data=False)
    val_df = load_dataset("val", limit=VAL_LIMIT, load_cached_data=False)
    test_df = load_dataset("test", limit=TEST_LIMIT, load_cached_data=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Validate Data Loading
    assert "text" in train_df.columns, "Train DF missing 'text' column"
    assert "tags_list" in train_df.columns, "Train DF missing 'tags_list' column"
    assert "text" in test_df.columns, "Test DF missing 'text' column"

    # 3. Preprocessing
    print("\n=== Preprocessing ===")

    # A. Text Features (Bag of Words)
    # Fit vectorizer on train, transform train
    X_train, vectorizer = get_text_features(
        train_df, "train", vectorizer=None, load_cached_data=False
    )

    # Transform val and test using the fitted vectorizer
    X_val, _ = get_text_features(
        val_df, "val", vectorizer=vectorizer, load_cached_data=False
    )
    X_test, _ = get_text_features(
        test_df, "test", vectorizer=vectorizer, load_cached_data=False
    )

    print(f"Text Feature Matrix (Train): {X_train.shape}")
    assert X_train.shape[0] == len(train_df)
    assert X_train.shape[1] == Config.VOCAB_SIZE

    # B. Target Features (One-Hot / Multi-label Binary)
    # Fit encoder on train, transform train
    Y_train, encoder = get_target_matrix(
        train_df, "train", encoder=None, load_cached_data=False
    )

    # Transform val using the fitted encoder
    Y_val, _ = get_target_matrix(val_df, "val", encoder=encoder, load_cached_data=False)

    print(f"Target Matrix (Train): {Y_train.shape}")
    assert Y_train.shape[0] == len(train_df)
    assert Y_train.shape[1] <= Config.TOP_K_TAGS

    # 4. Model Training
    print("\n=== Model Training ===")
    model = VectorizedMNB(alpha=1.0)
    model.fit(X_train, Y_train)

    # Validate Model Internal State
    assert model.is_fitted
    assert model.coef_.shape == (Y_train.shape[1], X_train.shape[1])

    # Test Model Saving and Loading
    model_path = os.path.join(Config.WORK_DIR, "demo_mnb_model.pkl")
    model.save(model_path)
    loaded_model = VectorizedMNB.load(model_path)
    assert loaded_model.is_fitted

    # 5. Validation / Evaluation
    print("\n=== Validation ===")
    # Predict scores (log-odds)
    scores_val = loaded_model.predict_scores(X_val)
    print(f"Scores shape: {scores_val.shape}")

    # Predict binary labels (threshold 0.0 implies prob > 0.5)
    preds_val_binary = loaded_model.predict(X_val, threshold=0.0)

    # Decode predictions to string tags
    preds_val_tags = encoder.inverse_transform(preds_val_binary)

    # Compare with ground truth for a few samples
    print("\nSample Predictions (Validation):")
    for i in range(3):
        truth = " ".join(val_df.iloc[i]["tags_list"])
        pred = preds_val_tags[i]
        print(f"  ID {val_df.iloc[i]['Id']}:")
        print(f"    True: {truth}")
        print(f"    Pred: {pred}")

    # Calculate Micro-F1 Score on the subset
    # Note: scikit-learn's f1_score handles sparse inputs
    f1 = f1_score(Y_val, preds_val_binary, average="micro")
    print(f"\nValidation Micro F1-Score (Subset): {f1:.4f}")

    # 6. Test Inference & Submission Generation
    print("\n=== Test Inference ===")
    preds_test_binary = loaded_model.predict(X_test, threshold=0.0)
    preds_test_tags = encoder.inverse_transform(preds_test_binary)

    # Construct Submission DataFrame
    submission_df = pd.DataFrame({"Id": test_df["Id"], "Tags": preds_test_tags})

    # Handle cases where no tags are predicted (fill with most common tag or leave empty)
    # For this demo, we verify the structure.
    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())

    # Final assertion on output format
    assert submission_df.columns.tolist() == ["Id", "Tags"]
    assert len(submission_df) == TEST_LIMIT

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
