import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.models import UNetLocalizer, FractureEncoder, AnatomicalTransformer
from library.engine import (
    train_segmentor,
    train_encoder,
    extract_features,
    train_aggregator,
)
from library.inference import InferencePipeline


def clean_working_directory():
    """
    Cleans up the working directory to ensure the demo runs from scratch.
    """
    # Remove checkpoints to force training
    if os.path.exists(Config.CHECKPOINT_DIR):
        shutil.rmtree(Config.CHECKPOINT_DIR)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Remove cache to force feature extraction
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Remove submission
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)


def verify_models():
    """
    Instantiates models and passes dummy data to verify shapes and logic.
    """
    print("Verifying Model Architectures...")
    device = Config.DEVICE

    # 1. UNetLocalizer
    # Input: (B, 1, H, W) -> Output: (B, Num_Classes, H, W)
    model_seg = UNetLocalizer(num_classes=8, pretrained=False).to(device)
    dummy_input_seg = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        output_seg = model_seg(dummy_input_seg)

    assert output_seg.shape == (
        2,
        8,
        256,
        256,
    ), f"UNet output shape mismatch. Expected (2, 8, 256, 256), got {output_seg.shape}"
    print("  [PASS] UNetLocalizer")

    # 2. FractureEncoder
    # Input: (B, 4, H, W) -> Output: (B, Feature_Dim)
    # Note: 4 channels = 3 RGB + 1 Mask
    model_enc = FractureEncoder(pretrained=False).to(device)
    dummy_input_enc = torch.randn(2, 4, 256, 256).to(device)
    with torch.no_grad():
        output_enc = model_enc(dummy_input_enc)

    # Feature dim depends on backbone, usually 1280 for EfficientNetV2-S
    expected_dim = model_enc.feature_dim
    assert output_enc.shape == (
        2,
        expected_dim,
    ), f"Encoder output shape mismatch. Expected (2, {expected_dim}), got {output_enc.shape}"
    print("  [PASS] FractureEncoder")

    # 3. AnatomicalTransformer
    # Input: Features (B, Seq, Dim), AnatIDs (B, Seq), Mask (B, Seq)
    # Output: Logits (B, 8)
    model_agg = AnatomicalTransformer().to(device)
    # Ensure hidden dims match for the test
    model_agg.input_dim = expected_dim
    # Re-init projection layer to match the encoder dim if different from default
    model_agg.feature_proj = torch.nn.Linear(expected_dim, model_agg.hidden_dim).to(
        device
    )

    seq_len = 10
    dummy_feats = torch.randn(2, seq_len, expected_dim).to(device)
    dummy_anat = torch.randint(0, 8, (2, seq_len)).to(device)
    dummy_mask = torch.ones(2, seq_len).to(device)

    with torch.no_grad():
        output_agg = model_agg(dummy_feats, dummy_anat, dummy_mask)

    assert output_agg.shape == (
        2,
        8,
    ), f"Transformer output shape mismatch. Expected (2, 8), got {output_agg.shape}"
    print("  [PASS] AnatomicalTransformer")
    print("All models verified successfully.\n")


def run_pipeline_demo():
    """
    Runs the full training and inference pipeline using a tiny subset of data.
    """
    print("Starting Pipeline Demonstration...")

    # --- 1. Train Segmentor (Stage 1) ---
    # This will train on the small subset defined by Config.DEBUG=True
    try:
        train_segmentor(load_cached_data=False)
        assert os.path.exists(
            Config.SEG_MODEL_PATH
        ), "Stage 1 model checkpoint not found after training."
    except Exception as e:
        print(f"Stage 1 Training Failed: {e}")
        raise e

    # --- 2. Train Encoder (Stage 2) ---
    try:
        train_encoder(load_cached_data=False)
        assert os.path.exists(
            Config.ENC_MODEL_PATH
        ), "Stage 2 model checkpoint not found after training."
    except Exception as e:
        print(f"Stage 2 Training Failed: {e}")
        raise e

    # --- 3. Feature Extraction ---
    # Uses the trained models to create .npy cache files
    try:
        extract_features(load_cached_data=False)
        assert os.path.exists(
            Config.TRAIN_FEATURES_CACHE
        ), "Train features cache not found."
        assert os.path.exists(
            Config.VAL_FEATURES_CACHE
        ), "Val features cache not found."
        assert os.path.exists(
            Config.TEST_FEATURES_CACHE
        ), "Test features cache not found."
    except Exception as e:
        print(f"Feature Extraction Failed: {e}")
        raise e

    # --- 4. Train Aggregator (Stage 3) ---
    try:
        train_aggregator(load_cached_data=False)
        assert os.path.exists(
            Config.AGG_MODEL_PATH
        ), "Stage 3 model checkpoint not found after training."
    except Exception as e:
        print(f"Stage 3 Training Failed: {e}")
        raise e

    # --- 5. Inference ---
    print("\nRunning Inference Pipeline...")
    pipeline = InferencePipeline()

    # We force `load_cached_data=False` to exercise the extraction logic inside inference,
    # or `True` to use what we just generated.
    # Since `extract_features` generated the test cache already, we can use it.
    # However, InferencePipeline uses a specific cache file "test_inference_data.npy"
    # which is different from "test_features.npy" generated by extract_features (logic varies slightly in provided files).
    # So we run predict with load_cached_data=False to force it to generate its own cache.

    # Clean specific inference cache if it exists to verify generation
    inference_cache = os.path.join(Config.CACHE_DIR, "test_inference_data.npy")
    if os.path.exists(inference_cache):
        os.remove(inference_cache)

    sub_df = pipeline.predict(load_cached_data=False)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    assert len(sub_df) > 0, "Submission dataframe is empty."
    assert (
        "row_id" in sub_df.columns and "fractured" in sub_df.columns
    ), "Submission columns mismatch."

    # Check row count: 5 studies (DEBUG) * 8 rows per study = 40 rows
    # The provided test_metadata might have more, but DEBUG limits iteration in inference.py
    expected_rows = 5 * 8
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission (5 debug studies), got {len(sub_df)}."

    print(f"\nPipeline Demonstration Completed Successfully.")
    print(f"Submission Head:\n{sub_df.head()}")


if __name__ == "__main__":
    # 1. Suppress Warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # 2. Configure for Speed/Demo
    # We monkey-patch the Config class to run minimal iterations on a subset of data.
    Config.DEBUG = True  # Forces data loaders to use only 5 samples
    Config.SEG_EPOCHS = 1
    Config.ENC_EPOCHS = 1
    Config.AGG_EPOCHS = 1

    Config.SEG_BATCH_SIZE = 2
    Config.ENC_BATCH_SIZE = 2
    Config.AGG_BATCH_SIZE = 2

    # 3. Set Seeds
    seed_everything(Config.SEED)

    # 4. Cleanup
    clean_working_directory()

    # 5. Verify Models
    verify_models()

    # 6. Run Full Pipeline
    run_pipeline_demo()
