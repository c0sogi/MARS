import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import CFG
from library.data import JigsawDataset, MLMDataset
from library.model import JigsawModel
from library.losses import JigsawLoss
from library.utils import seed_everything, JigsawMetrics, AWP, EMA
from library.engine import train_mlm, train_epoch, valid_epoch, inference


def demo_main():
    # 1. Setup and Configuration Overrides for Speed
    print(">>> Setting up demonstration...")
    seed_everything(CFG.seed)

    # Override CFG for speed
    CFG.train_batch_size = 4
    CFG.valid_batch_size = 8
    CFG.max_len = 64  # Reduce sequence length for speed
    device = CFG.device

    # 2. Data Preparation (Using Subsets)
    print("\n>>> Loading and preprocessing data subsets...")

    # Load raw metadata
    train_df_full = pd.read_csv(CFG.train_path)
    val_df_full = pd.read_csv(CFG.val_path)
    test_df_full = pd.read_csv(CFG.test_path)

    # Slice to tiny subsets
    N_SAMPLES = 50
    train_subset = train_df_full.head(N_SAMPLES).copy()
    val_subset = val_df_full.head(N_SAMPLES).copy()
    test_subset = test_df_full.head(N_SAMPLES).copy()

    # Replicate Preprocessing Logic from library.data.preprocess_data
    # (We do this manually here to avoid processing the full 1.4M rows in the library function)

    # Train Preprocessing
    train_subset[CFG.binary_target_col] = (train_subset[CFG.target_col] >= 0.5).astype(
        int
    )
    identity_sub = train_subset[CFG.identity_cols].fillna(0.0)
    train_subset["identity_present"] = (identity_sub.max(axis=1) >= 0.5).astype(int)

    # Bias Trap Weighting
    is_toxic = train_subset[CFG.binary_target_col] == 1
    has_identity = train_subset["identity_present"] == 1
    bias_trap_mask = has_identity  # Simplification for demo: treat any identity mention as requiring attention

    train_subset["loss_weight"] = 1.0
    train_subset.loc[bias_trap_mask, "loss_weight"] = CFG.bias_sample_weight
    train_subset["sampler_weight"] = 1.0
    train_subset.loc[bias_trap_mask, "sampler_weight"] = CFG.bias_sample_weight

    # Val Preprocessing
    val_subset[CFG.binary_target_col] = (val_subset[CFG.target_col] >= 0.5).astype(int)

    print(
        f"Subset shapes: Train={train_subset.shape}, Val={val_subset.shape}, Test={test_subset.shape}"
    )

    # 3. Tokenizer and Dataset Initialization
    print("\n>>> Initializing Tokenizer and Datasets...")
    # Using a smaller fast tokenizer if possible, but we stick to config model for correctness
    # We assume internet is available or model is cached. If not, this might hang,
    # but the prompt says "All packages are already installed", implying model weights might be too.
    # If debert-v3-large is too slow to load, we'd normally swap, but we must use provided lib.
    try:
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    except Exception as e:
        print(
            f"Warning: Could not load {CFG.model_name} tokenizer. Using 'bert-base-uncased' for demo."
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Instantiate Datasets
    train_ds = JigsawDataset(train_subset, tokenizer, CFG.max_len)
    val_ds = JigsawDataset(val_subset, tokenizer, CFG.max_len)
    test_ds = JigsawDataset(test_subset, tokenizer, CFG.max_len, is_test=True)

    # Verify Dataset Output
    sample = train_ds[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "target",
        "identities",
        "attack",
        "sample_weights",
    ]
    for k in required_keys:
        assert k in sample, f"Dataset sample missing key: {k}"
    print("Dataset verification passed.")

    # Create Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=CFG.train_batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=CFG.valid_batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=CFG.valid_batch_size, shuffle=False
    )

    # 4. Model Initialization
    print("\n>>> Initializing Model...")
    # We use pretrained=False to avoid downloading 800MB+ model weights during this quick demo
    # In a real run, pretrained=True
    model = JigsawModel(pretrained=False)
    model.to(device)

    # Verify Forward Pass
    dummy_input = sample["input_ids"].unsqueeze(0).to(device)
    dummy_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(dummy_input, dummy_mask)

    assert "logits" in outputs
    assert "aux_identity_logits" in outputs
    assert "aux_attack_logits" in outputs
    assert outputs["logits"].shape == (1, 1)
    assert outputs["aux_identity_logits"].shape == (1, len(CFG.identity_cols))
    print("Model forward pass verification passed.")

    # 5. Loss Function Verification
    print("\n>>> Verifying Loss Function...")
    criterion = JigsawLoss()

    # Create dummy targets matching batch size 1
    dummy_targets = {
        "target": torch.tensor([[1.0]]).to(device),
        "identities": torch.randn(1, len(CFG.identity_cols)).to(device),
        "attack": torch.tensor([[0.0]]).to(device),
        "sample_weights": torch.tensor([[1.0]]).to(device),
    }

    # Test General Stage Loss
    loss, loss_dict = criterion(outputs, dummy_targets, stage="general")
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"General Stage Loss: {loss.item():.4f}")

    # Test Robust Stage Loss (Ranking)
    # Ranking loss requires >1 sample to find pairs, so we expect it to be 0 or handle single sample gracefully
    loss_robust, loss_dict_robust = criterion(outputs, dummy_targets, stage="robust")
    print(f"Robust Stage Loss (1 sample): {loss_robust.item():.4f}")

    # 6. Training Loop Simulation
    print("\n>>> Simulating Training Stages...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=10
    )

    # --- Stage 1: DAPT (MLM) ---
    print("--- Simulating Stage 1: DAPT ---")
    # Create MLM Dataset from text
    mlm_texts = train_subset["comment_text"].fillna("").values
    mlm_ds = MLMDataset(mlm_texts, tokenizer, CFG.max_len)

    # Need collator for MLM
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    mlm_loader = torch.utils.data.DataLoader(
        mlm_ds, batch_size=2, collate_fn=data_collator
    )

    # We can't easily use the JigsawModel for MLM training directly as it wraps the backbone
    # and adds classification heads. The library `train_mlm` expects a model that returns `outputs.loss`.
    # Usually, one would use AutoModelForMaskedLM. For this demo, we skip the actual MLM training call
    # because JigsawModel is not an MLM model. We will assume the user swaps the model class for Stage 1.
    # Instead, we verify the data flow for MLM.
    batch = next(iter(mlm_loader))
    assert "input_ids" in batch
    assert "labels" in batch
    print("MLM Data Loading verified.")

    # --- Stage 2: General Fine-Tuning ---
    print("--- Simulating Stage 2: General Fine-Tuning ---")
    avg_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        epoch=0,
        stage="general",
        awp=None,
        ema=None,
    )
    print(f"Stage 2 Epoch complete. Avg Loss: {avg_loss:.4f}")

    # --- Stage 3: Robust Optimization (AWP) ---
    print("--- Simulating Stage 3: Robust Optimization (AWP) ---")
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-2)
    avg_loss_robust = train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        epoch=0,
        stage="robust",
        awp=awp,
        ema=None,
    )
    print(f"Stage 3 Epoch complete. Avg Loss: {avg_loss_robust:.4f}")

    # 7. Validation and Metrics
    print("\n>>> Running Validation...")
    ema = EMA(model)
    val_loss, val_score = valid_epoch(model, val_loader, criterion, device, ema=ema)
    print(f"Validation complete. Loss: {val_loss:.4f}, Score: {val_score:.4f}")

    # Verify Metric Logic explicitly
    print("Verifying Metric Logic...")
    metrics_calc = JigsawMetrics()
    # Create a perfect prediction scenario
    y_true_df = val_subset.copy()
    # Ensure at least one identity is present for metric calculation stability
    y_true_df.loc[0, CFG.identity_cols[0]] = 1.0

    # Perfect predictions (match binary target)
    perfect_preds = y_true_df[CFG.binary_target_col].values.astype(float)
    score, _, _, _, _ = metrics_calc.get_final_metric(y_true_df, perfect_preds)
    # Score should be 1.0 for perfect predictions
    assert (
        score == 1.0
    ), f"Metric logic failed: Perfect predictions should yield score 1.0, got {score}"
    print("Metric logic verification passed.")

    # 8. Inference
    print("\n>>> Running Inference...")
    predictions = inference(model, test_loader, device, ema=ema)

    # Create Submission
    submission = pd.DataFrame(
        {"id": test_subset["id"], "prediction": predictions.flatten()}
    )

    output_path = os.path.join(CFG.output_dir, "submission.csv")
    # submission.to_csv(output_path, index=False) # Not writing to disk to avoid clutter, just print
    print(f"Inference complete. Predictions shape: {predictions.shape}")
    print("Sample Submission Head:")
    print(submission.head())

    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    demo_main()
