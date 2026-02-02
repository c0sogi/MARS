import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import gc

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_score
from library.data import load_and_process_data, InsultDataset, get_dataloaders
from library.model import InsultModel
from library.awp import AWP
from library.tapt import run_tapt
from library.train import run_training
from library.inference import predict


def run_demo():
    print("=" * 50)
    print("STARTING DEMO RUN")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Use a tiny model to ensure rapid execution
    Config.model_name = "prajjwal1/bert-tiny"

    # Enable debug mode to use a small subset of data
    Config.debug = True
    Config.debug_subset_size = 50  # Very small subset

    # Reduce training parameters
    Config.epochs = 2
    Config.tapt_epochs = 1
    Config.n_folds = 2
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.tapt_batch_size = 4

    # Output directory for demo
    Config.output_dir = "./working/demo_run"
    Config.submission_path = os.path.join(Config.output_dir, "submission.csv")
    Config.tapt_output_dir = os.path.join(Config.output_dir, "tapt_model")

    # Setup directories
    Config.setup()
    seed_everything(Config.seed)

    print(f"Model: {Config.model_name}")
    print(f"Debug Mode: {Config.debug}")
    print(f"Output Dir: {Config.output_dir}")

    # ---------------------------------------------------------
    # 2. Data Loading and Processing Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load data
    df_train, df_val, df_test = load_and_process_data(load_cached_data=False)

    # Verify data loading
    assert len(df_train) > 0, "Training data is empty"
    assert len(df_val) > 0, "Validation data is empty"
    assert len(df_test) > 0, "Test data is empty"
    assert "Comment" in df_train.columns, "Comment column missing"
    assert "Insult" in df_train.columns, "Target column missing"

    print(
        f"Train rows: {len(df_train)}, Val rows: {len(df_val)}, Test rows: {len(df_test)}"
    )

    # Verify Dataset Class
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    ds = InsultDataset(df_train.head(10), tokenizer, max_len=32)
    item = ds[0]

    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 32, "Incorrect sequence length"
    assert isinstance(item["labels"], torch.Tensor), "Labels should be a tensor"

    print("Data processing logic verified.")

    # ---------------------------------------------------------
    # 3. Model Initialization and Forward Pass Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Logic...")

    device = Config.device
    model = InsultModel(pretrained=True)
    model.to(device)
    model.train()

    # Create a dummy batch
    batch_size = 2
    dummy_input = torch.randint(0, 100, (batch_size, 32)).to(device)
    dummy_mask = torch.ones((batch_size, 32)).to(device)

    # Forward pass
    logits = model(dummy_input, dummy_mask)

    assert logits.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {logits.shape}"
    print("Model forward pass successful.")

    # ---------------------------------------------------------
    # 4. Adversarial Weight Perturbation (AWP) Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying AWP Logic...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    awp = AWP(model, optimizer, adv_lr=0.1, adv_eps=0.1, start_epoch=0)

    # Save original weight of a parameter
    param_name = "fc.weight"
    orig_weight = model.fc.weight.data.clone()

    # Dummy loss and backward to generate gradients
    loss = logits.mean()
    loss.backward()

    # Attack
    awp.attack()
    perturbed_weight = model.fc.weight.data.clone()

    # Verify weights changed
    diff = torch.norm(perturbed_weight - orig_weight)
    assert diff > 0, "AWP attack did not perturb weights"
    print(f"AWP perturbation magnitude: {diff.item():.6f}")

    # Restore
    awp.restore()
    restored_weight = model.fc.weight.data.clone()

    # Verify weights restored
    assert torch.allclose(
        orig_weight, restored_weight
    ), "AWP restore failed to recover original weights"
    print("AWP logic verified.")

    # Clean up
    del model, optimizer, awp, logits, loss
    torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # 5. Task-Adaptive Pre-Training (TAPT) Execution
    # ---------------------------------------------------------
    print("\n[5] Running TAPT (Demo)...")

    # This will run TAPT for 1 epoch on the tiny subset
    try:
        run_tapt()
        tapt_model_path = os.path.join(Config.tapt_output_dir, "tapt_model.pth")
        assert os.path.exists(tapt_model_path), "TAPT model file was not created"
        print("TAPT execution successful.")
    except Exception as e:
        print(f"TAPT execution failed: {e}")
        raise e

    # ---------------------------------------------------------
    # 6. Full Training Pipeline Execution
    # ---------------------------------------------------------
    print("\n[6] Running Training Pipeline (Demo)...")

    # This runs the K-Fold training loop
    # It uses the TAPT model generated in step 5 if Config.use_tapt is True
    try:
        run_training()

        # Verify output files
        for fold in range(Config.n_folds):
            model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")
            assert os.path.exists(model_path), f"Model for fold {fold} missing"

        assert os.path.exists(
            Config.submission_path
        ), "Submission file missing after training"

        # Check submission content
        sub_df = pd.read_csv(Config.submission_path)
        assert (
            Config.target_col in sub_df.columns
        ), "Prediction column missing in submission"
        assert (
            len(sub_df) == Config.debug_subset_size
        ), f"Submission size mismatch. Expected {Config.debug_subset_size}, got {len(sub_df)}"

        print("Training pipeline successful.")

    except Exception as e:
        print(f"Training pipeline failed: {e}")
        raise e

    # ---------------------------------------------------------
    # 7. Inference Pipeline Execution
    # ---------------------------------------------------------
    print("\n[7] Running Inference Pipeline (Demo)...")

    # Rename previous submission to avoid overwrite confusion during verification
    os.rename(Config.submission_path, Config.submission_path + ".train_generated")

    try:
        predict(load_cached_data=True)

        assert os.path.exists(
            Config.submission_path
        ), "Submission file missing after inference"

        sub_df = pd.read_csv(Config.submission_path)
        preds = sub_df[Config.target_col].values

        # Verify predictions range
        assert (preds >= 0).all() and (
            preds <= 1
        ).all(), "Predictions out of range [0, 1]"

        print("Inference pipeline successful.")
        print(f"Sample Predictions:\n{sub_df[[Config.target_col]].head()}")

    except Exception as e:
        print(f"Inference pipeline failed: {e}")
        raise e

    print("\n" + "=" * 50)
    print("DEMO RUN COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
