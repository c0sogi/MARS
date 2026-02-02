import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as transformers_logging

# Suppress verbose transformer warnings
transformers_logging.set_verbosity_error()

# Import library components
from library.config import Config
from library.utils import set_seed, JigsawMetric
from library.losses import JigsawLoss
from library.model import ToxicityModel, AWP
from library.data import ToxicityDataset, load_data, get_weighted_loader, DataMiner
from library.engine import train_fn, eval_fn, inference_fn


# ==========================================
# 1. Configuration & Setup
# ==========================================
def setup_demo_config():
    """
    Overrides Config parameters to use a tiny model and small dataset
    for rapid demonstration and verification.
    """
    print("[Demo] Configuring environment for rapid execution...")

    # Use a tiny BERT model for speed
    Config.model_name = "prajjwal1/bert-tiny"
    Config.tokenizer_name = "prajjwal1/bert-tiny"

    # Reduce hyperparameters
    Config.max_len = 64
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.epochs = 1
    Config.accumulate_grad_batches = 1
    Config.awp_start_epoch = 0  # Enable AWP immediately for testing

    # Set debug flags
    Config.debug = True
    Config.debug_subset_size = 50  # Only use 50 samples

    # Set device
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Demo] Device: {Config.device}")

    set_seed(Config.seed)


# ==========================================
# 2. Data Pipeline Verification
# ==========================================
def verify_data_pipeline():
    print("\n[Demo] Verifying Data Pipeline...")

    # Load raw data (this uses the provided metadata files)
    # We manually slice here because load_data loads the full file
    full_train_df = load_data("train", load_cached_data=False)
    train_df = full_train_df.head(Config.debug_subset_size).copy()

    full_val_df = load_data("val", load_cached_data=False)
    val_df = full_val_df.head(Config.debug_subset_size).copy()

    print(f"  Train subset shape: {train_df.shape}")
    print(f"  Val subset shape: {val_df.shape}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)

    # Create DataLoader
    train_loader = get_weighted_loader(
        train_df, tokenizer, Config.train_batch_size, is_test=False
    )

    # Fetch a batch to verify structure
    batch = next(iter(train_loader))

    # Assertions
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "target" in batch
    assert "identity_target" in batch
    assert "aux_target" in batch
    assert "weight" in batch

    # Check shapes
    B = Config.train_batch_size
    assert batch["input_ids"].shape == (B, Config.max_len)
    assert batch["target"].shape == (B,) or batch["target"].shape == (B, 1)
    assert batch["identity_target"].shape[1] == len(Config.identity_cols)
    assert batch["aux_target"].shape[1] == len(Config.aux_cols)

    print("  Data Pipeline verified successfully.")
    return train_loader, val_df, tokenizer


# ==========================================
# 3. Model & Loss Verification
# ==========================================
def verify_model_and_loss(train_loader):
    print("\n[Demo] Verifying Model and Loss Function...")

    # Initialize Model
    model = ToxicityModel(model_name=Config.model_name, pretrained=True)
    model.to(Config.device)

    # Initialize Loss
    loss_fn = JigsawLoss()

    # Get a batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(Config.device)
    attention_mask = batch["attention_mask"].to(Config.device)
    targets = batch["target"].to(Config.device).view(-1, 1)
    identity_targets = batch["identity_target"].to(Config.device)
    aux_targets = batch["aux_target"].to(Config.device)
    weights = batch["weight"].to(Config.device).view(-1, 1)

    # Forward Pass
    tox_logits, ident_logits, aux_logits = model(input_ids, attention_mask)

    # Assert Output Shapes
    B = input_ids.shape[0]
    assert tox_logits.shape == (B, 1), f"Expected (B, 1), got {tox_logits.shape}"
    assert ident_logits.shape == (B, len(Config.identity_cols))
    assert aux_logits.shape == (B, len(Config.aux_cols))

    # Loss Calculation
    combined_aux_logits = torch.cat([ident_logits, aux_logits], dim=1)
    combined_aux_targets = torch.cat([identity_targets, aux_targets], dim=1)

    loss, loss_dict = loss_fn(
        toxicity_logits=tox_logits,
        toxicity_targets=targets,
        aux_logits=combined_aux_logits,
        aux_targets=combined_aux_targets,
        sample_weights=weights,
    )

    # Assert Loss correctness
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss).any()
    assert "loss_toxicity" in loss_dict
    assert "loss_rank" in loss_dict

    print(f"  Forward pass successful. Total Loss: {loss.item():.4f}")
    print("  Model and Loss verified successfully.")

    return model, loss_fn


# ==========================================
# 4. Training Engine Verification (with AWP)
# ==========================================
def verify_training_engine(model, train_loader, loss_fn):
    print("\n[Demo] Verifying Training Engine (with AWP)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = None  # Skip scheduler for simple demo

    # Initialize AWP
    awp = AWP(model, optimizer, adv_lr=0.1, adv_eps=0.01, start_epoch=0)

    # Run one epoch (which is just a few batches due to subset)
    losses = train_fn(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=Config.device,
        awp=awp,
        epoch=0,
    )

    assert losses["total"] > 0
    print(f"  Training step complete. Avg Loss: {losses['total']:.4f}")
    print("  Training Engine verified successfully.")


# ==========================================
# 5. Metric Logic Verification
# ==========================================
def verify_metric_logic():
    print("\n[Demo] Verifying Jigsaw Metric Logic...")

    metric = JigsawMetric()

    # Create synthetic dataframe
    # Columns: binary_target, prediction, [identity_cols...]
    data = {
        Config.binary_target_col: [1, 0, 1, 0],  # Toxic, Non-Toxic, Toxic, Non-Toxic
        "prediction": [
            0.9,
            0.1,
            0.8,
            0.6,
        ],  # Good preds, except last one (False Positive)
        "male": [0, 0, 1, 1],  # Identity present in last two
        "female": [0, 0, 0, 0],
    }
    # Fill other identities with 0
    for col in Config.identity_cols:
        if col not in data:
            data[col] = [0, 0, 0, 0]

    df = pd.DataFrame(data)

    # The last sample (Target=0, Pred=0.6, Male=1) is a "Background Positive, Subgroup Negative" error?
    # No, Target=0 means Negative. Pred=0.6 means model thinks it's Positive.
    # Identity=1.
    # This is a Non-Toxic comment mentioning identity, predicted as Toxic.
    # This hurts BPSN AUC (Background Positive?? No, BPSN = Background Positive, Subgroup Negative).
    # Wait, BPSN definition: "restrict to non-toxic examples that mention identity AND toxic examples that do not".
    # A low BPSN score means model confuses non-toxic identity mentions with toxic non-mentions.

    results = metric.compute(df, df["prediction"].values)

    assert "final_score" in results
    assert "overall_auc" in results
    assert "bpsn_auc_mean" in results
    assert 0.0 <= results["final_score"] <= 1.0

    print(f"  Metric computed: {results['final_score']:.4f}")
    print("  Metric Logic verified successfully.")


# ==========================================
# 6. Data Miner Verification
# ==========================================
def verify_data_miner():
    print("\n[Demo] Verifying Data Miner (Hard Negative Mining)...")

    miner = DataMiner()

    # Create dummy train data
    train_df = pd.DataFrame(
        {
            "id": [1, 2],
            "comment_text": ["text1", "text2"],
            "target": [0.1, 0.9],
            "binary_target": [0, 1],
            "weight": [1.0, 1.0],
        }
    )
    # Add identity cols to train
    for col in Config.identity_cols:
        train_df[col] = 0.0
    for col in Config.aux_cols:
        train_df[col] = 0.0

    # Create dummy scout predictions
    # Case: Low toxicity prediction, High identity probability -> Hard Negative?
    # No, Hard Negative usually means: Non-Toxic Label but Model predicts High Toxicity (False Positive).
    # OR: Model predicts Low Toxicity but it has Identity?
    # Let's check DataMiner code logic:
    # mask = (max_identity_prob > threshold) & (toxicity_prob < threshold)
    # This captures: "Model is confident it's an identity, and confident it's NOT toxic".
    # These are "Safe Identity Mentions".
    # If we add them to training with Target=0, we reinforce that Identity != Toxic.

    scout_preds_df = pd.DataFrame(
        {
            "id": [101, 102],
            "comment_text": ["I am a man", "I hate you"],
            "prediction": [0.05, 0.95],  # Low tox, High tox
        }
    )

    # Identity preds:
    # 101: High prob for 'male' (index 0)
    # 102: Low prob for all
    scout_identity_preds = np.zeros((2, len(Config.identity_cols)))
    scout_identity_preds[0, 0] = 0.9  # High 'male'

    # Run augmentation
    aug_df = miner.augment_training_data(train_df, scout_preds_df, scout_identity_preds)

    # Expectation: Row 101 should be added because identity > 0.5 and tox < 0.1
    # Row 102 (High tox) should not be added by this specific logic (which looks for non-toxic identities)

    assert len(aug_df) == 3  # 2 original + 1 mined
    assert aug_df.iloc[-1]["weight"] == 5.0  # Check weight assignment
    assert aug_df.iloc[-1]["male"] == 1.0  # Check pseudo-label

    print(f"  Mined {len(aug_df) - len(train_df)} samples.")
    print("  Data Miner verified successfully.")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    t0 = time.time()

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Data
        train_loader, val_df, tokenizer = verify_data_pipeline()

        # 3. Model & Loss
        model, loss_fn = verify_model_and_loss(train_loader)

        # 4. Training
        verify_training_engine(model, train_loader, loss_fn)

        # 5. Evaluation (Engine)
        # Create val loader
        val_dataset = ToxicityDataset(val_df, tokenizer, Config.max_len, is_test=False)
        val_loader = DataLoader(val_dataset, batch_size=Config.valid_batch_size)

        print("\n[Demo] Verifying Evaluation Engine...")
        val_losses, val_metrics, preds = eval_fn(
            model, val_loader, loss_fn, Config.device, val_df
        )
        print(f"  Val AUC: {val_metrics['overall_auc']:.4f}")

        # 6. Metric Logic
        verify_metric_logic()

        # 7. Data Miner
        verify_data_miner()

        print("\n==========================================")
        print(f"SUCCESS: All components verified in {time.time() - t0:.2f} seconds.")
        print("==========================================")

    except AssertionError as e:
        print(f"\n[FAILURE] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILURE] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
