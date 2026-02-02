import os
import sys
import pandas as pd
import logging
from library.utils import set_seed, setup_logger
from library.text_processing import RegexTransducer
from library.data_loader import load_dataset
from library.stats_model import HierarchicalLookupModel


def main():
    # 1. Setup and Reproducibility
    set_seed(42)
    logger = setup_logger("DemoScript")
    logger.info("Starting demonstration script...")

    # Define working directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission_demo.csv")

    # 2. Demonstrate and Verify RegexTransducer
    # The RegexTransducer is a fallback for OOV tokens. We verify it works as coded.
    logger.info("Verifying RegexTransducer logic...")
    rt = RegexTransducer()

    # Test Case 1: Money
    # Logic: $3.16 -> "three dollars, sixteen cents"
    # The provided code joins words with spaces, e.g., "twenty three" not "twenty-three".
    money_input = "$3.16"
    money_expected = "three dollars, sixteen cents"
    money_pred = rt.normalize(money_input)
    if money_pred != money_expected:
        raise AssertionError(
            f"RegexTransducer failed for {money_input}. Expected '{money_expected}', got '{money_pred}'"
        )

    # Test Case 2: Cardinal
    # Logic: 123 -> "one hundred twenty three"
    cardinal_input = "123"
    cardinal_expected = "one hundred twenty three"
    cardinal_pred = rt.normalize(cardinal_input)
    if cardinal_pred != cardinal_expected:
        raise AssertionError(
            f"RegexTransducer failed for {cardinal_input}. Expected '{cardinal_expected}', got '{cardinal_pred}'"
        )

    logger.info("RegexTransducer verification passed.")

    # 3. Demonstrate Data Loading
    # We load a small sample (1%) of the training data to verify the loader works
    logger.info("Testing Data Loader with sampling...")
    df_sample = load_dataset(
        split="train",
        base_dir="./metadata",
        cache_dir=CACHE_DIR,
        process_context=True,
        sample_ratio=0.01,
        seed=42,
    )

    # Verify DataFrame structure
    required_columns = ["sentence_id", "token_id", "before", "after", "prev_before"]
    for col in required_columns:
        if col not in df_sample.columns:
            raise AssertionError(f"Loaded dataset missing column: {col}")

    if len(df_sample) == 0:
        raise AssertionError("Loaded dataset is empty.")

    logger.info(f"Successfully loaded {len(df_sample)} rows (approx 1% sample).")

    # 4. Demonstrate Model Training (HierarchicalLookupModel)
    # We train on a 5% sample to keep execution fast while learning frequent patterns.
    logger.info("Initializing and training HierarchicalLookupModel...")
    model = HierarchicalLookupModel(cache_dir=CACHE_DIR)

    # Fit the model
    # Note: We disable loading cached data for the fit step to demonstrate the learning process
    # on the specific sample ratio, unless the cache matches exactly (which it won't initially).
    model.fit(train_split="train", load_cached_data=False, sample_ratio=0.05)

    # Verify model learned something
    if len(model.l1_lookup) == 0:
        raise AssertionError("Model failed to learn L1 unigram stats.")
    logger.info(
        f"Model trained. L1 Size: {len(model.l1_lookup)}, L2 Size: {len(model.l2_lookup)}"
    )

    # 5. Demonstrate Evaluation
    # Evaluate on the validation set. The evaluate method loads the full validation set.
    # Since lookup is fast, this is acceptable.
    logger.info("Evaluating model on validation set...")
    val_accuracy = model.evaluate(val_split="val", load_cached_data=True)

    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    # Basic sanity check: Accuracy should be high because >90% of tokens are identity mappings (PUNCT, PLAIN)
    # and the model defaults to identity if not found.
    if val_accuracy < 0.90:
        raise AssertionError(
            f"Validation accuracy {val_accuracy} is unexpectedly low (<0.90). Check model logic."
        )

    # 6. Demonstrate Submission Generation
    # Generate predictions for the test set
    logger.info("Generating submission file...")
    model.generate_submission(
        test_split="test", output_file=SUBMISSION_PATH, load_cached_data=True
    )

    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file was not created at {SUBMISSION_PATH}")

    # Verify submission format briefly
    df_sub = pd.read_csv(SUBMISSION_PATH)
    if "id" not in df_sub.columns or "after" not in df_sub.columns:
        raise AssertionError(
            "Submission file missing required columns 'id' or 'after'."
        )

    logger.info(f"Submission generated successfully with {len(df_sub)} rows.")
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
