import os
import sys
import torch
import pandas as pd
import numpy as np
import transformers
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.nn import MSELoss

# Import library modules
from library.config import CFG
from library.utils import seed_everything, get_device
from library.cpc_utils import get_cpc_texts
from library.data import (
    preprocess_data,
    prepare_loaders,
    prepare_inference_loader,
    PearsonDataset,
)
from library.model import CustomModel, get_optimizer_params
from library.engine import ModelEMA, train_fn, valid_fn, inference_fn


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Initializing configuration...")

    # Set seed for reproducibility
    seed_everything(CFG.seed)

    # Get compute device
    device = get_device()
    print(f"Device: {device}")

    # Suppress verbose transformer logging
    transformers.logging.set_verbosity_error()

    # Modify CFG for the demonstration to ensure speed and isolation
    CFG.working_dir = "./working/demo_run"
    CFG.cpc_cache_path = os.path.join(CFG.working_dir, "cpc_texts.parquet")
    CFG.num_epochs = 1
    CFG.n_folds = 2  # We won't run the full loop, but setting this ensures logic holds
    CFG.debug = True  # Enables debug subsets in prepare_loaders
    CFG.print_freq = 10

    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    # ==========================================
    # 2. CPC Utils Demonstration
    # ==========================================
    print("\n[Demo] Parsing CPC Contexts...")
    # This parses description.md or loads from cache
    cpc_texts = get_cpc_texts(CFG, load_cached_data=False)

    # Verification
    assert isinstance(cpc_texts, dict), "CPC texts should be a dictionary."
    assert len(cpc_texts) > 0, "CPC texts dictionary is empty."
    print(f"Successfully loaded {len(cpc_texts)} CPC context descriptions.")

    # ==========================================
    # 3. Data Preprocessing Demonstration
    # ==========================================
    print("\n[Demo] Preprocessing Data...")
    # This creates train_folds.parquet and test_processed.parquet
    train_df, test_df = preprocess_data(CFG, load_cached_data=False)

    # Verification
    assert not train_df.empty, "Processed train DataFrame is empty."
    assert not test_df.empty, "Processed test DataFrame is empty."
    assert (
        "context_text" in train_df.columns
    ), "context_text column missing in train_df."
    assert "fold" in train_df.columns, "fold column missing in train_df."
    print(f"Data processed. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ==========================================
    # 4. Dataset and DataLoader Demonstration
    # ==========================================
    print("\n[Demo] Preparing Datasets and Loaders...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # Verify Dataset Class logic manually
    sample_data = train_df.head(5).copy()
    dataset = PearsonDataset(sample_data, tokenizer, CFG.max_len, is_train=True)
    sample_item = dataset[0]

    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "labels" in sample_item
    assert isinstance(sample_item["labels"], torch.Tensor)
    print("PearsonDataset item structure verified.")

    # Create Loaders (using debug=True to get small subsets: 100 train, 50 val)
    train_loader, val_loader = prepare_loaders(
        fold=0, tokenizer=tokenizer, cfg=CFG, debug=True
    )

    # Verify Loader
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == CFG.batch_size
    print(f"DataLoader created. Batch size: {batch['input_ids'].shape[0]}")

    # ==========================================
    # 5. Model Initialization Demonstration
    # ==========================================
    print("\n[Demo] Initializing Model...")
    model = CustomModel(CFG, pretrained=True)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        output = model(input_ids, attention_mask, token_type_ids)

    assert output.shape == (
        CFG.batch_size,
    ), f"Output shape mismatch. Expected ({CFG.batch_size},), got {output.shape}"
    print("Model forward pass successful.")

    # ==========================================
    # 6. Training Loop Demonstration
    # ==========================================
    print("\n[Demo] Running Training Loop (1 Epoch, Debug Subset)...")

    # Setup Optimizer with LLRD
    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=CFG.learning_rate,
        decoder_lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    optimizer = AdamW(
        optimizer_params, lr=CFG.learning_rate, eps=CFG.eps, betas=CFG.betas
    )

    # Setup Scheduler
    num_train_steps = len(train_loader) * CFG.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * CFG.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Setup EMA
    model_ema = ModelEMA(model, decay=CFG.ema_decay)

    # Loss Function (MSE for Pearson correlation proxy)
    criterion = MSELoss()

    # Train Function
    avg_loss = train_fn(
        train_loader,
        model,
        criterion,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        cfg=CFG,
        model_ema=model_ema,
    )

    assert not np.isnan(avg_loss), "Training loss returned NaN."
    print(f"Training Epoch Completed. Average Loss: {avg_loss:.4f}")

    # ==========================================
    # 7. Validation Demonstration
    # ==========================================
    print("\n[Demo] Running Validation...")
    val_loss, val_score = valid_fn(
        val_loader, model, criterion, device, CFG, model_ema=model_ema
    )

    assert isinstance(val_score, float)
    assert -1.0 <= val_score <= 1.0, f"Pearson score {val_score} out of range [-1, 1]"
    print(f"Validation Completed. Loss: {val_loss:.4f}, Pearson Score: {val_score:.4f}")

    # ==========================================
    # 8. Inference Demonstration
    # ==========================================
    print("\n[Demo] Running Inference on Test Set...")
    # prepare_inference_loader loads the full test set (3648 samples)
    # This is fast enough for a demo on GPU.
    test_loader = prepare_inference_loader(tokenizer, CFG)

    predictions = inference_fn(test_loader, model, device, model_ema=model_ema)

    assert len(predictions) == len(
        test_df
    ), "Number of predictions does not match test set size."
    print(f"Inference generated {len(predictions)} predictions.")

    # Optional: Save dummy submission
    submission = pd.DataFrame({"id": test_df["id"], "score": predictions})
    sub_path = os.path.join(CFG.working_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Sample submission saved to {sub_path}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
