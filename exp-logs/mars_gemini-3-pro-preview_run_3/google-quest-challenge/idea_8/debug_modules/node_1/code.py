import os
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_tokenizer, get_dataloaders
from library.dapt_engine import run_dapt
from library.finetune_engine import train_fold
from library.ridge_topology import TopologyRidgeTrainer
from library.ensemble import MetaStacker


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo
    # --------------------------------------------------------------------------
    print(">>> Setting up Demo Configuration...")

    # Use a specific demo directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Use a smaller model for speed (DeBERTa-v3-xsmall)
    # This ensures the code runs quickly while testing the exact same architecture paths
    DEMO_MODEL = "microsoft/deberta-v3-xsmall"
    Config.MLM_MODEL_NAME = DEMO_MODEL
    Config.MODEL_A_NAME = DEMO_MODEL

    # Reduce training parameters
    Config.EPOCHS = 1
    Config.N_FOLDS = 2  # We will simulate 2 folds
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRAD_ACCUMULATION_STEPS = 1

    # Clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model: {Config.MLM_MODEL_NAME}")

    # --------------------------------------------------------------------------
    # 2. Verify Data Loading & Tokenization Logic
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Data Loader...")
    tokenizer = get_tokenizer(Config.MLM_MODEL_NAME)

    # Get debug dataloaders (subsets data to small count)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        load_cached_data=False,  # Force reload to test processing
        debug=True,
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    print(f"Batch Keys: {batch.keys()}")

    # Verify Shapes
    assert "input_ids" in batch
    assert "q_mask" in batch
    assert "a_mask" in batch
    assert batch["input_ids"].shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)

    # Verify Mask Logic: q_mask and a_mask should be binary
    # And they should generally not overlap for the actual text segments (though special tokens might be 0 in both)
    q_m = batch["q_mask"]
    a_m = batch["a_mask"]

    # Check that we have some 1s in both masks (assuming data has both Q and A)
    assert q_m.sum() > 0, "Question mask is empty!"
    assert a_m.sum() > 0, "Answer mask is empty!"

    print("Data Loader verification successful.")

    # --------------------------------------------------------------------------
    # 3. Phase 1: Domain Adaptive Pre-Training (DAPT)
    # --------------------------------------------------------------------------
    print("\n>>> Running Phase 1: DAPT...")
    # Run DAPT for 1 epoch on debug data
    dapt_model_path = run_dapt(load_cached_data=False, debug=True, epochs=1)

    assert os.path.exists(dapt_model_path), "DAPT model directory not created."
    assert os.path.exists(
        os.path.join(dapt_model_path, "config.json")
    ), "DAPT config missing."
    print(f"DAPT completed. Model saved to {dapt_model_path}")

    # --------------------------------------------------------------------------
    # 4. Phase 2: Supervised Fine-Tuning (Fold 0)
    # --------------------------------------------------------------------------
    print("\n>>> Running Phase 2: Fine-Tuning (Fold 0)...")
    # Train Fold 0 using the DAPT model
    ft_ckpt_path = train_fold(
        model_name_or_path=dapt_model_path,
        fold_idx=0,
        debug=True,
        load_cached_data=False,
        epochs=1,
    )

    assert os.path.exists(ft_ckpt_path), "Fine-tuned checkpoint not found."
    print(f"Fine-Tuning completed. Checkpoint: {ft_ckpt_path}")

    # --------------------------------------------------------------------------
    # 5. Phase 3: Feature Extraction & Ridge Regression (Fold 0)
    # --------------------------------------------------------------------------
    print("\n>>> Running Phase 3: Ridge Topology Training (Fold 0)...")

    model_tag = "demo_model"
    ridge_trainer = TopologyRidgeTrainer(
        model_tag=model_tag, fold_idx=0, base_model_name=Config.MLM_MODEL_NAME
    )

    # This extracts features, trains Ridge heads, and generates OOF/Test preds
    test_preds = ridge_trainer.train_and_predict(
        checkpoint_path=ft_ckpt_path, debug=True, load_cached_data=False
    )

    # Verify Outputs
    oof_preds_path = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold0_oof_preds.npy"
    )
    test_preds_path = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold0_test_preds.npy"
    )

    assert os.path.exists(oof_preds_path), "OOF predictions missing."
    assert os.path.exists(test_preds_path), "Test predictions missing."

    # Verify Value Range [0, 1]
    oof_preds = np.load(oof_preds_path)
    assert (
        oof_preds.min() >= 0.0 and oof_preds.max() <= 1.0
    ), "OOF preds out of range [0,1]"
    assert (
        test_preds.min() >= 0.0 and test_preds.max() <= 1.0
    ), "Test preds out of range [0,1]"

    print("Ridge Training completed successfully.")

    # --------------------------------------------------------------------------
    # 6. Simulate Fold 1 (for Ensemble Demo)
    # --------------------------------------------------------------------------
    print("\n>>> Simulating Fold 1 Artifacts...")
    # To demonstrate the Stacker without running training twice, we copy Fold 0 files to Fold 1
    files_to_copy = [
        f"{model_tag}_fold0_val_targets.npy",
        f"{model_tag}_fold0_oof_preds.npy",
        f"{model_tag}_fold0_test_preds.npy",
        f"{model_tag}_fold0_test_ids.npy",
    ]

    for fname in files_to_copy:
        src = os.path.join(Config.WORKING_DIR, fname)
        dst = os.path.join(Config.WORKING_DIR, fname.replace("fold0", "fold1"))
        shutil.copy(src, dst)

    print("Fold 1 simulation completed.")

    # --------------------------------------------------------------------------
    # 7. Phase 4: Meta-Ensemble Stacking
    # --------------------------------------------------------------------------
    print("\n>>> Running Phase 4: Meta-Stacking...")

    stacker = MetaStacker(model_tags=[model_tag])
    stacker.run()

    # --------------------------------------------------------------------------
    # 8. Final Verification
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Submission...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["qa_id"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check values
    # In debug mode, test set is 50 rows
    assert len(df_sub) == 50, f"Expected 50 rows in debug submission, got {len(df_sub)}"

    # Check probability range
    numeric_cols = df_sub.iloc[:, 1:]
    assert (numeric_cols.values >= 0).all() and (
        numeric_cols.values <= 1
    ).all(), "Submission values out of range"

    print("\n" + "=" * 40)
    print(" DEMO RUN COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()
