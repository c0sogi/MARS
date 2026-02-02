import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import warnings
from torch.utils.data import DataLoader
from transformers import logging as transformers_logging

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from provided library
from library.config import CFG
from library.utils import seed_everything, get_logger
from library.data_processing import (
    get_data,
    get_tokenizer,
    JigsawDataset,
    get_loaders,
    _preprocess_df,
)
from library.model import JigsawModel
from library.losses import HybridLoss
from library.awp import AWP
from library.trainer import compute_bias_metrics


def demo_pipeline():
    print("Initializing demonstration...")

    # 1. Override Configuration for Speed and Debugging
    # We modify the global CFG class attributes directly to run a small-scale test.
    print("1. Configuring environment...")
    CFG.debug = True
    CFG.debug_sample_size = 64  # Small sample for quick execution
    CFG.train_batch_size = 4
    CFG.valid_batch_size = 4
    CFG.epochs = 1
    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    seed_everything(CFG.seed)

    # 2. Data Processing Demonstration
    print("2. Testing Data Processing...")

    # Load Tokenizer
    tokenizer = get_tokenizer()
    assert tokenizer is not None, "Tokenizer failed to load."

    # Load Data (Force reload from metadata to verify processing logic)
    # We disable loading from cache to ensure the processing logic runs.
    train_df, val_df, test_df = get_data(load_cached_data=False, debug=True)

    # Verify DataFrames
    assert (
        len(train_df) == CFG.debug_sample_size
    ), f"Train DF size mismatch: {len(train_df)}"
    assert "loss_weight" in train_df.columns, "Loss weights not generated in Train DF."
    assert "binary_target" in train_df.columns, "Binary target not present."

    # Verify Dataset Class
    ds = JigsawDataset(train_df, tokenizer, max_len=128, is_test=False)
    sample = ds[0]

    # Check sample keys and shapes
    required_keys = [
        "input_ids",
        "attention_mask",
        "target",
        "aux_labels",
        "loss_weight",
    ]
    for key in required_keys:
        assert key in sample, f"Dataset sample missing key: {key}"

    # input_ids should be (max_len,)
    assert sample["input_ids"].shape == (128,), "Incorrect input_ids shape."
    # aux_labels should be (9 identities + 1 attack) = 10
    assert sample["aux_labels"].shape == (
        10,
    ), f"Incorrect aux_labels shape: {sample['aux_labels'].shape}"

    print("   Data Processing verified successfully.")

    # 3. Model Architecture Demonstration
    print("3. Testing Model Architecture...")

    # Initialize Model
    model = JigsawModel(pretrained=True)
    model.to(CFG.device)
    model.train()

    # Create a small batch
    loader = DataLoader(ds, batch_size=CFG.train_batch_size, shuffle=False)
    batch = next(iter(loader))

    input_ids = batch["input_ids"].to(CFG.device)
    attention_mask = batch["attention_mask"].to(CFG.device)
    targets = batch["target"].to(CFG.device)
    aux_labels = batch["aux_labels"].to(CFG.device)
    weights = batch["loss_weight"].to(CFG.device)

    # Forward Pass
    outputs = model(input_ids, attention_mask)

    # Verify Output Structure
    assert "logits" in outputs
    assert "aux_identity" in outputs
    assert "aux_attack" in outputs

    # Verify Shapes
    # logits: (batch_size, 1)
    assert outputs["logits"].shape == (
        CFG.train_batch_size,
        1,
    ), "Logits shape mismatch."
    # aux_identity: (batch_size, 9)
    assert outputs["aux_identity"].shape == (
        CFG.train_batch_size,
        9,
    ), "Aux Identity shape mismatch."

    print("   Model forward pass verified successfully.")

    # 4. Loss Function Demonstration
    print("4. Testing Hybrid Loss...")

    criterion = HybridLoss()
    loss = criterion(outputs, targets, aux_labels, weights)

    # Verify Loss
    assert torch.is_tensor(loss), "Loss is not a tensor."
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."

    print(f"   Loss calculated: {loss.item():.4f}")

    # 5. Adversarial Weight Perturbation (AWP) Demonstration
    print("5. Testing AWP (Adversarial Weight Perturbation)...")

    # AWP requires an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Initialize AWP
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-2, start_epoch=0)

    # We need gradients to perform an attack
    loss.backward()

    # Capture original weight of a specific parameter (e.g., toxicity head weight)
    # We pick a leaf parameter that requires grad
    param_name = "toxicity_head.weight"
    orig_param = dict(model.named_parameters())[param_name].clone()

    # Perform Attack
    awp.attack()

    # Check if weight changed
    attacked_param = dict(model.named_parameters())[param_name]
    assert not torch.equal(
        orig_param, attacked_param
    ), "AWP Attack failed: Weights did not change."

    # Restore Weights
    awp.restore()
    restored_param = dict(model.named_parameters())[param_name]

    # Check if weights are restored (allow for tiny float errors, though usually exact)
    assert torch.allclose(
        orig_param, restored_param, atol=1e-7
    ), "AWP Restore failed: Weights did not match original."

    print("   AWP attack and restore verified successfully.")

    # 6. Metric Computation Demonstration
    print("6. Testing Bias Metrics...")

    # Generate synthetic predictions for the validation dataframe loaded earlier
    # val_df has 64 rows (debug size)
    # We create random predictions between 0 and 1
    np.random.seed(CFG.seed)
    dummy_preds = np.random.rand(len(val_df))

    # Ensure we have enough variety in targets for AUC calculation
    # In a tiny debug sample of 64, we might not have all classes/identities populated perfectly.
    # We manually inject some diversity into the dataframe for this test to avoid AUC errors (only one class present).
    val_df.loc[0:10, "target"] = 0.8  # Toxic
    val_df.loc[11:20, "target"] = 0.1  # Non-toxic
    val_df.loc[0:5, "male"] = 1.0  # Mention male
    val_df.loc[11:15, "male"] = 1.0  # Mention male

    # Compute Metrics
    score, overall_auc, sub_auc, bpsn_auc, bnsp_auc = compute_bias_metrics(
        val_df, dummy_preds
    )

    print(f"   Computed Score: {score:.4f}")
    print(f"   Overall AUC: {overall_auc:.4f}")
    print(f"   Subgroup AUC (Mean): {sub_auc:.4f}")

    # Assertions
    assert isinstance(score, float), "Score is not a float."
    assert 0.0 <= overall_auc <= 1.0, "Overall AUC out of range."

    print("   Metrics calculation verified successfully.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    demo_pipeline()
