import os
import shutil
import warnings
import torch
from library.config import Config
from library.engine import run_full_pipeline

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clean_cache():
    """
    Removes cached parquet files to prevent dimension mismatches.
    Cite debug_lesson_1: Verify Cache Consistency Against Input Data Dimensions.
    """
    cache_dir = Config.WORKING_DIR
    print(f"Cleaning cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        # Remove specific parquet files to force regeneration
        for f in [
            "train_processed.parquet",
            "val_processed.parquet",
            "test_processed.parquet",
        ]:
            path = os.path.join(cache_dir, f)
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed stale cache file: {path}")


def main():
    # 1. Clear Cache
    # We ensure that the data loader regenerates the full dataset features
    # instead of potentially loading a subset from a previous debug run.
    clean_cache()

    # 2. Configure for Robust Execution
    # We use the full 5-fold cross-validation pipeline.
    # This utilizes DeBERTa, RoBERTa, and Statistical models.
    print("--- Starting Robust Full Pipeline ---")

    # Ensure the device is set correctly
    Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {Config.DEVICE}")

    # 3. Execute Pipeline
    # This function handles:
    # - 5-Fold CV
    # - OOF Prediction Generation
    # - Meta-Learner Training (Stacking)
    # - Final Submission Generation
    run_full_pipeline(debug=False, n_folds=5)


if __name__ == "__main__":
    main()
