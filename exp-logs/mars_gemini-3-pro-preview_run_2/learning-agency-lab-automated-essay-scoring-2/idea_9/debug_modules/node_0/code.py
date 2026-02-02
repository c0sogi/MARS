import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# --- Import Library Components ---
from library.config import Config
from library.data import load_data, get_collate_fn
from library.features import extract_linguistic_features
from library.models_classic import ClassicBranch
from library.models_semantic import run_semantic_training, predict_semantic_test
from library.stacking import StackingModel, optimize_thresholds


def setup_logging():
    """Suppress verbose logs for a clean demo output."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    # The library logger uses INFO, which we want to see for high-level steps


def run_pipeline_demo():
    print("==================================================")
    print("      Essay Scoring Pipeline Demo Execution       ")
    print("==================================================")

    # 1. Configuration & Setup
    # Override Config for a fast, minimal run
    print("\n[1] Configuring Environment...")
    Config.debug = True
    Config.debug_sample_size = 20  # Very small subset for speed
    Config.epochs = 1  # Single epoch
    Config.n_folds = 2  # Minimum folds for CV
    Config.train_batch_size = 2  # Small batch size
    Config.valid_batch_size = 2

    # Initialize directories
    Config.setup()
    setup_logging()
    print("    Configuration complete. Debug mode enabled.")

    # 2. Data Loading & Processing
    print("\n[2] Verifying Data Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load Train Data
    train_ds = load_data(tokenizer, split="train", debug=True)

    # Verify Dataset Length
    assert (
        len(train_ds) == Config.debug_sample_size
    ), f"Dataset length mismatch. Expected {Config.debug_sample_size}, got {len(train_ds)}"

    # Verify Collation
    collate_fn = get_collate_fn(tokenizer)
    loader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert batch["input_ids"].shape[0] == 4
    print(f"    Data loaded successfully. Batch shape: {batch['input_ids'].shape}")

    # 3. Feature Extraction
    print("\n[3] Verifying Structural Feature Extraction...")
    # Compute features for the loaded training subset
    features_df = extract_linguistic_features(
        train_ds.df, split="train_debug", load_cached_data=False
    )

    assert len(features_df) == len(train_ds.df)
    assert "word_count" in features_df.columns
    assert "flesch_reading_ease" in features_df.columns
    print(f"    Features computed. Shape: {features_df.shape}")

    # 4. Classic Branches (Ridge Regression)
    print("\n[4] Running Classic Branches (Lexical & Morphological)...")

    # Lexical Branch (Word N-grams)
    lexical = ClassicBranch(
        name="lexical", analyzer="word", ngram_range=(1, 1), min_df=1
    )
    lex_oof, lex_test = lexical.run(load_cached_data=False)

    # Morphological Branch (Char N-grams)
    morph = ClassicBranch(
        name="morphological", analyzer="char", ngram_range=(2, 2), min_df=1
    )
    mor_oof, mor_test = morph.run(load_cached_data=False)

    # Verify outputs match debug size
    assert len(lex_oof) == Config.debug_sample_size
    assert len(mor_oof) == Config.debug_sample_size
    print("    Classic branches execution successful.")

    # 5. Semantic Branch (DeBERTa)
    print("\n[5] Running Semantic Branch (Fine-tuning DeBERTa)...")
    print("    (This step involves a forward/backward pass on the GPU)")

    # Run Training (Generates OOF)
    sem_oof = run_semantic_training()

    # Run Inference (Generates Test Preds)
    sem_test = predict_semantic_test()

    # --- Alignment Fix ---
    # The provided library files have a discrepancy in debug subsetting logic:
    # ClassicBranch subsets the *merged* train+val dataframe to `debug_sample_size`.
    # SemanticBranch subsets train and val *separately* via `load_data`, then merges.
    # This results in Semantic OOF being larger than Classic OOF in debug mode.
    # We truncate Semantic OOF to match Classic OOF so the Stacking model receives aligned arrays.

    if len(sem_oof) > Config.debug_sample_size:
        print(
            f"    *Adjusting Semantic OOF size from {len(sem_oof)} to {Config.debug_sample_size} for alignment*"
        )
        sem_oof = sem_oof[: Config.debug_sample_size]
        # Overwrite the file on disk so StackingModel loads the correct size
        np.save(os.path.join(Config.output_dir, "semantic_oof.npy"), sem_oof)

    print("    Semantic branch execution successful.")

    # 6. Stacking & Submission
    print("\n[6] Running Stacking Meta-Learner...")
    stacker = StackingModel()

    # Train Stacker (LightGBM)
    oof_stack, y_true = stacker.train()
    assert len(oof_stack) == Config.debug_sample_size

    # Optimize Thresholds
    print("    Optimizing thresholds...")
    best_thresholds = optimize_thresholds(y_true, oof_stack)

    # Predict on Test Set
    test_preds_continuous, essay_ids = stacker.predict()
    assert len(test_preds_continuous) == Config.debug_sample_size

    # Apply Thresholds
    final_scores = np.digitize(test_preds_continuous, best_thresholds) + 1
    final_scores = np.clip(final_scores, 1, 6)

    # Create Submission
    submission = pd.DataFrame({"essay_id": essay_ids, "score": final_scores})

    print("\n[7] Final Submission Generated:")
    print(submission.head())

    # Final Validation
    assert submission.shape == (Config.debug_sample_size, 2)
    assert submission["score"].min() >= 1
    assert submission["score"].max() <= 6

    print("\n==================================================")
    print("           Demo Completed Successfully            ")
    print("==================================================")


if __name__ == "__main__":
    # Ensure we don't crash on minor warnings
    import warnings

    warnings.filterwarnings("ignore")

    try:
        run_pipeline_demo()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
