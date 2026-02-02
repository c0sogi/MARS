import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import logging

# 1. Suppress Progress Bars and Warnings
# Monkey-patch tqdm to disable progress bars from the library modules
import tqdm.auto


class SilentTqdm(tqdm.auto.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


tqdm.auto.tqdm = SilentTqdm

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)

# 2. Import Library Modules
from library.config import Config
from library.utils import seed_everything, compute_qwk, CacheManager
from library.data import load_processed_data, get_folds, EssayDataset
from library.model_backbone import EssayBackbone
from library.engine import train_one_epoch, valid_one_epoch, extract_embeddings
from library.model_head import StackingTrainer
from library.pipeline import run_cv, generate_submission
from transformers import AutoTokenizer


def run_demo():
    print("=== Starting Essay Scoring Solution Demo ===")

    # -------------------------------------------------------------------------
    # 3. Configure for Speed (Debug Mode)
    # -------------------------------------------------------------------------
    print("\n[Configuration] Patching Config for fast demonstration...")
    Config.debug = True  # Limits dataset to 100 rows
    Config.epochs = 1
    Config.n_folds = 2  # Reduce folds
    Config.batch_size = 2
    Config.accum_iter = 1
    Config.working_dir = "./working/demo_run"

    # Update derived paths
    Config.cache_dir = os.path.join(Config.working_dir, "cache")
    Config.checkpoint_dir = os.path.join(Config.working_dir, "checkpoints")
    Config.output_dir = os.path.join(Config.working_dir, "output")
    Config.submission_dir = os.path.join(Config.working_dir, "submission")
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Create directories
    Config.create_directories()

    # Patch LightGBM params for speed
    Config.lgbm_params["n_estimators"] = 10
    Config.lgbm_params["early_stopping_rounds"] = 5
    Config.lgbm_params["verbosity"] = -1

    seed_everything(Config.seed)
    print("Config patched and directories created.")

    # -------------------------------------------------------------------------
    # 4. Verify Utils
    # -------------------------------------------------------------------------
    print("\n[Utils] Verifying metric and cache...")

    # Test QWK
    y_true = np.array([1, 2, 3, 4, 5, 6])
    y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0, 5.9])
    score = compute_qwk(y_true, y_pred)
    assert score > 0.9, f"QWK calculation seems incorrect, got {score}"
    print(f"QWK Metric Check: Passed (Score: {score:.4f})")

    # Test CacheManager
    cm = CacheManager(Config.cache_dir)
    dummy_data = np.random.rand(5, 5)
    cm.save(dummy_data, "test_cache", ext="npy")
    loaded_data = cm.load("test_cache", ext="npy")
    assert np.allclose(dummy_data, loaded_data), "CacheManager save/load failed."
    print("CacheManager Check: Passed")

    # -------------------------------------------------------------------------
    # 5. Verify Data Processing
    # -------------------------------------------------------------------------
    print("\n[Data] Verifying data loading and processing...")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load Train Data (Debug mode = 100 samples)
    train_ds = load_processed_data(tokenizer, mode="train", load_cached_data=False)

    assert (
        len(train_ds) == 100
    ), f"Expected 100 samples in debug mode, got {len(train_ds)}"
    assert train_ds.scores is not None, "Train dataset should have scores."

    sample_item = train_ds[0]
    assert "input_ids" in sample_item
    assert "meta_features" in sample_item
    assert (
        sample_item["meta_features"].shape[0] == 5
    ), "Meta features should have 5 dimensions."

    print(f"Dataset Loaded. Size: {len(train_ds)}")
    print("Data Processing Check: Passed")

    # -------------------------------------------------------------------------
    # 6. Verify Backbone Model
    # -------------------------------------------------------------------------
    print("\n[Backbone] Verifying DeBERTa model architecture...")

    device = Config.device
    model = EssayBackbone(pretrained=True)
    model.to(device)
    model.eval()

    # Create dummy batch
    batch_size = 2
    seq_len = 128
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones((batch_size, seq_len)).to(device)

    with torch.no_grad():
        # Test 1: Score Prediction
        logits = model(dummy_input_ids, dummy_mask, return_embedding=False)
        assert logits.shape == (
            batch_size,
            1,
        ), f"Expected output shape {(batch_size, 1)}, got {logits.shape}"

        # Test 2: Embedding Extraction
        embeddings = model(dummy_input_ids, dummy_mask, return_embedding=True)
        # DeBERTa-v3-large hidden size is 1024
        expected_dim = model.config.hidden_size
        assert embeddings.shape == (
            batch_size,
            expected_dim,
        ), f"Expected embedding shape {(batch_size, expected_dim)}, got {embeddings.shape}"

    print("Backbone Model Check: Passed")

    # Clean up GPU memory
    del model, dummy_input_ids, dummy_mask
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 7. Verify Stacking Head (LightGBM)
    # -------------------------------------------------------------------------
    print("\n[Head] Verifying Stacking Trainer (LightGBM)...")

    # Generate dummy data for stacking
    n_samples = 50
    hidden_dim = 1024
    n_meta = 5

    X_emb = np.random.rand(n_samples, hidden_dim).astype(np.float32)
    X_meta = np.random.rand(n_samples, n_meta).astype(np.float32)
    y = np.random.randint(1, 7, size=n_samples).astype(np.float32)

    stacker = StackingTrainer()

    # Fit
    stacker.fit(X_emb, X_meta, y)
    assert stacker.model is not None, "Model should be trained."

    # Predict
    preds = stacker.predict(X_emb, X_meta)
    assert preds.shape == (n_samples,), "Prediction shape mismatch."

    # Save/Load
    stacker.save(Config.output_dir, "test_lgbm.txt")
    stacker_loaded = StackingTrainer()
    stacker_loaded.load(os.path.join(Config.output_dir, "test_lgbm.txt"))
    preds_loaded = stacker_loaded.predict(X_emb, X_meta)

    assert np.allclose(preds, preds_loaded), "Model save/load inconsistency."
    print("Stacking Head Check: Passed")

    # -------------------------------------------------------------------------
    # 8. Run Full Pipeline
    # -------------------------------------------------------------------------
    print("\n[Pipeline] Running End-to-End Pipeline (CV + Inference)...")

    # Step 1: Run Cross-Validation
    # This will train backbone on 2 folds (debug mode), save checkpoints,
    # extract OOF embeddings, and train the stacking head.
    print("--- Running CV ---")
    run_cv()

    # Verify outputs exist
    assert os.path.exists(
        os.path.join(Config.checkpoint_dir, "backbone_fold_0.pth")
    ), "Fold 0 checkpoint missing."
    assert os.path.exists(
        os.path.join(Config.output_dir, "lgbm_stacking.txt")
    ), "Stacking model missing."

    # Step 2: Generate Submission
    # This loads test data, generates ensemble embeddings, predicts, and saves CSV.
    print("--- Generating Submission ---")
    generate_submission()

    assert os.path.exists(Config.submission_path), "Submission file was not created."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission generated with shape: {sub_df.shape}")
    print(sub_df.head())

    # Check if scores are valid integers 1-6
    valid_scores = sub_df["score"].isin([1, 2, 3, 4, 5, 6]).all()
    assert valid_scores, "Submission contains invalid scores (must be integers 1-6)."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
