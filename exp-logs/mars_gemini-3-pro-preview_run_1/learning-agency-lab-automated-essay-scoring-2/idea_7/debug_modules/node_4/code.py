import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import logging

# =============================================================================
# 1. Environment Setup & Mocking
# =============================================================================


# Mock tqdm to suppress progress bars from the library
class MockTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


# Inject mock into library.trainer before importing functions that use it
import library.trainer

library.trainer.tqdm = MockTqdm

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import preprocess_data, EssayDataset, MLMDataset, get_tokenizer
from library.model import CustomModel
from library.trainer import train_mlm, train_fold
from library.stacking import run_stacking, StackingModel

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Demo")


def main():
    seed_everything(42)
    logger.info("Starting Essay Scoring Pipeline Demo...")

    # =========================================================================
    # 2. Configuration Overrides for Speed & Demo
    # =========================================================================

    # Define a separate working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths
    Config.working_dir = DEMO_DIR
    Config.cache_dir = os.path.join(DEMO_DIR, "cache")
    Config.model_dir = os.path.join(DEMO_DIR, "checkpoints")
    os.makedirs(Config.model_dir, exist_ok=True)
    Config.mlm_model_dir = os.path.join(DEMO_DIR, "mlm_checkpoints")
    Config.submission_path = os.path.join(DEMO_DIR, "submission.csv")

    # Override Model & Training params for speed
    Config.model_name = (
        "microsoft/deberta-v3-xsmall"  # Use xsmall for fast download/inference
    )
    Config.num_folds = 2  # Only run 2 folds
    Config.epochs = 1
    Config.mlm_epochs = 1
    Config.train_batch_size = 2
    Config.eval_batch_size = 2
    Config.mlm_batch_size = 2
    Config.gradient_accumulation_steps = 1
    Config.max_length = 128  # Shorten sequence length for demo speed
    Config.debug = False  # We handle data subsampling manually below

    # Override LightGBM params for speed
    Config.lgbm_params["n_estimators"] = 10
    Config.lgbm_params["verbosity"] = -1

    # =========================================================================
    # 3. Data Preparation (Subsampling)
    # =========================================================================
    logger.info("Preparing subsampled data...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample tiny subsets: 40 train (enough for 2 folds of 20), 10 test
    demo_train = orig_train.sample(n=40, random_state=42).reset_index(drop=True)
    demo_test = orig_test.sample(n=10, random_state=42).reset_index(drop=True)

    # Save these subsets to the demo directory
    demo_train_path = os.path.join(DEMO_DIR, "train.csv")
    demo_test_path = os.path.join(DEMO_DIR, "test.csv")
    demo_train.to_csv(demo_train_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Point Config to these new files
    Config.train_path = demo_train_path
    Config.test_path = demo_test_path

    # Run Preprocessing (Feature Engineering + Fold Creation)
    # This will generate .parquet files in Config.cache_dir
    train_df, test_df = preprocess_data(load_cached_data=False)

    # Validation: Check if features and folds exist
    assert (
        "char_count" in train_df.columns
    ), "Feature Engineering failed: char_count missing"
    assert "fold" in train_df.columns, "Fold creation failed"
    assert len(train_df) == 40, "Train DataFrame size mismatch"
    logger.info("Data preprocessing verified.")

    # =========================================================================
    # 4. Dataset & Model Verification
    # =========================================================================
    logger.info("Verifying Dataset and Model classes...")

    tokenizer = get_tokenizer()

    # Test EssayDataset
    ds = EssayDataset(train_df, tokenizer, max_length=Config.max_length)
    item = ds[0]
    assert "input_ids" in item and "attention_mask" in item and "labels" in item
    assert item["input_ids"].shape == (Config.max_length,), "Incorrect input_ids shape"
    assert isinstance(item["labels"].item(), float), "Label should be float"

    # Test MLMDataset
    mlm_ds = MLMDataset(
        train_df["full_text"].values, tokenizer, max_length=Config.max_length
    )
    mlm_item = mlm_ds[0]
    assert "labels" in mlm_item
    assert mlm_item["labels"].shape == (
        Config.max_length,
    ), "Incorrect MLM labels shape"

    # Test CustomModel
    model = CustomModel(model_name=Config.model_name, pretrained=True)
    model.to(Config.device)
    model.eval()

    # Run a dummy forward pass
    with torch.no_grad():
        input_ids = item["input_ids"].unsqueeze(0).to(Config.device)
        mask = item["attention_mask"].unsqueeze(0).to(Config.device)
        output = model(input_ids, mask)

    assert output.shape == (1, 1), f"Model output shape mismatch: {output.shape}"
    logger.info("Dataset and Model logic verified.")

    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # 5. Stage 1: MLM Pre-training
    # =========================================================================
    logger.info("Running MLM Pre-training (Demo)...")
    mlm_path = train_mlm(train_df, test_df)

    assert os.path.exists(os.path.join(mlm_path, "config.json")), "MLM model not saved"
    logger.info("MLM Pre-training completed.")

    # =========================================================================
    # 6. Stage 2: Supervised Fine-Tuning (Cross-Validation)
    # =========================================================================
    logger.info("Running Supervised Training (2 Folds)...")

    oof_preds_list = []
    test_preds_accum = np.zeros(len(test_df))

    for fold in range(Config.num_folds):
        logger.info(f"  Training Fold {fold}...")
        ids, oof_preds, test_preds, score = train_fold(
            fold, train_df, test_df, mlm_model_path=mlm_path
        )

        # Verify outputs
        assert len(ids) == len(oof_preds), "OOF IDs and Preds length mismatch"
        assert len(test_preds) == len(test_df), "Test Preds length mismatch"

        # Collect OOF
        fold_oof = pd.DataFrame(
            {"essay_id": ids, "pred_score": oof_preds, "fold": fold}
        )
        oof_preds_list.append(fold_oof)

        # Accumulate Test Preds
        test_preds_accum += test_preds

    # Combine OOF
    oof_df = pd.concat(oof_preds_list, axis=0).reset_index(drop=True)

    # Average Test Preds
    avg_test_preds = test_preds_accum / Config.num_folds

    assert len(oof_df) == len(train_df), "OOF size does not match Train size"
    logger.info("Supervised Training completed.")

    # =========================================================================
    # 7. Stage 3: Stacking (Level 2)
    # =========================================================================
    logger.info("Running Stacking Ensemble...")

    # run_stacking internally loads train_processed.parquet from cache to get meta-features
    # It joins oof_df with meta-features
    submission = run_stacking(oof_df, avg_test_preds, load_cached_data=True)

    # Validation
    assert submission is not None, "Stacking failed to return submission"
    assert os.path.exists(Config.submission_path), "Submission file not found"
    assert len(submission) == len(test_df), "Submission length mismatch"
    assert "essay_id" in submission.columns and "score" in submission.columns

    # Check values are integers 1-6
    assert submission["score"].dtype == int or submission["score"].dtype == np.int64
    assert submission["score"].min() >= 1 and submission["score"].max() <= 6

    logger.info("Stacking completed.")
    logger.info(f"Final Submission Head:\n{submission.head()}")
    logger.info("Success! All components verified.")


if __name__ == "__main__":
    main()
