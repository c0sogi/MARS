import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import transformers
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import library components
import library.config as config
import library.utils as utils
import library.data as data_lib
import library.model as model_lib
import library.engine as engine_lib


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    print("=== Setting up Environment ===")
    # Suppress warnings and verbose logs for clean output
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set seed for reproducibility
    utils.seed_everything(config.SEED)

    device = config.DEVICE
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Optimization: Patch Data Loader for Speed
    # --------------------------------------------------------------------------
    print("=== Patching Data Loader for Fast Demonstration ===")
    # The default library processes the full 1.4M dataset. We patch the loader
    # used by library.data to return only 500 rows. This allows the demo to
    # run in seconds.

    original_loader = utils.load_processed_data

    def fast_loader_proxy(split, load_cached_data=True):
        # Force reload from source to apply slicing, ignoring potentially huge cached files
        # unless we already created our small cached files in this run.
        # For this demo, we rely on the fact that we are slicing the dataframe.
        print(f"  Loading and slicing data for split: {split}")
        df = original_loader(split, load_cached_data=False)
        return df.head(500)  # Slice to 500 samples

    # Apply the patch to the library.data module where it is used
    data_lib.load_processed_data = fast_loader_proxy

    # --------------------------------------------------------------------------
    # 3. Data Preparation
    # --------------------------------------------------------------------------
    print("=== Preparing Datasets ===")

    # Instantiate Datasets (this will trigger tokenization of the 500 samples)
    train_dataset = data_lib.ToxicityDataset("train", load_cached_data=False)
    val_dataset = data_lib.ToxicityDataset("validation", load_cached_data=False)
    test_dataset = data_lib.ToxicityDataset("test", load_cached_data=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    # Verify dataset logic
    assert len(train_dataset) == 500, "Train dataset should be sliced to 500"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # --------------------------------------------------------------------------
    # 4. Model Initialization
    # --------------------------------------------------------------------------
    print("=== Initializing Model ===")
    model = model_lib.MultiTaskRoBERTa(
        model_name=config.MODEL_NAME,
        dropout_rate=config.DROPOUT,
        num_identities=len(config.IDENTITY_COLUMNS),
    )
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.LR)

    # --------------------------------------------------------------------------
    # 5. Training Loop (Demo)
    # --------------------------------------------------------------------------
    print("=== Starting Training (1 Epoch) ===")

    # Run training for 1 epoch
    avg_loss = engine_lib.train_fn(train_loader, model, optimizer, device)

    print(f"Epoch 1 Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # --------------------------------------------------------------------------
    # 6. Evaluation & Metrics
    # --------------------------------------------------------------------------
    print("=== Running Evaluation ===")

    # Get predictions
    val_preds, val_targets, val_identities = engine_lib.eval_fn(
        val_loader, model, device
    )

    # Verify output shapes
    assert len(val_preds) == len(val_dataset), "Prediction count mismatch"
    assert len(val_targets) == len(val_dataset), "Target count mismatch"

    # Reconstruct Identity DataFrame for the Evaluator
    # val_identities is a list of lists/arrays. Convert to DF with correct columns.
    val_identity_df = pd.DataFrame(val_identities, columns=config.IDENTITY_COLUMNS)

    # Initialize Evaluator
    evaluator = utils.JigsawEvaluator(val_targets, val_preds, val_identity_df)

    # Calculate Metrics
    final_score, overall_auc, sub_auc, bpsn_auc, bnsp_auc = evaluator.get_final_metric()

    print(f"Validation Metrics:")
    print(f"  Overall AUC: {overall_auc:.4f}")
    print(f"  Subgroup AUC (Mean): {sub_auc:.4f}")
    print(f"  BPSN AUC (Mean): {bpsn_auc:.4f}")
    print(f"  BNSP AUC (Mean): {bnsp_auc:.4f}")
    print(f"  Final Weighted Score: {final_score:.4f}")

    # Note: With only 500 samples, some identity subgroups might be empty,
    # resulting in NaNs which the evaluator handles (returns NaN or 0.5 fallback).
    # We just check that the code ran without error.

    # --------------------------------------------------------------------------
    # 7. Inference & Submission
    # --------------------------------------------------------------------------
    print("=== Running Inference ===")

    test_ids, test_preds = engine_lib.inference_fn(test_loader, model, device)

    assert len(test_ids) == len(test_dataset), "Test ID count mismatch"
    assert len(test_preds) == len(test_dataset), "Test prediction count mismatch"

    print("=== Saving Submission ===")
    engine_lib.save_submission(test_ids, test_preds)

    # Verify file creation
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    # Verify file content format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    assert list(sub_df.columns) == ["id", "prediction"], "Submission columns mismatch"
    assert len(sub_df) == 500, "Submission row count mismatch"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
