import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.cpc_mapping import get_cpc_texts
from library.dataset import PearsonDataset
from library.model import CustomModel
from library.engine import get_optimizer_params, train_fn, valid_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DemoConfig(Config):
    """
    Configuration overrides for the demonstration run.
    Optimized for speed and resource usage.
    """

    # Use a separate directory for demo outputs
    working_dir = "./working/demo_run"
    output_dir = working_dir
    model_dir = os.path.join(output_dir, "models")
    predictions_dir = os.path.join(output_dir, "predictions")
    submission_path = os.path.join(output_dir, "submission.csv")

    # Execution constraints
    debug = True
    debug_sample_size = 50  # Only use 50 samples for demo
    epochs = 1
    max_length = 64  # Reduced from 256 for speed
    train_batch_size = 4
    valid_batch_size = 8

    # Model settings
    # We keep the model_name as is, assuming environment handles it.
    # If download is slow, this is the only bottleneck.

    # Hardware
    num_workers = 2


def main():
    # 1. Setup
    cfg = DemoConfig()
    cfg.setup()  # Create directories
    seed_everything(cfg.seed)

    logger = get_logger(os.path.join(cfg.working_dir, "demo.log"))
    logger.info("Starting Semantic Similarity Demo...")
    logger.info(f"Device: {cfg.device}")

    # 2. Load Data
    logger.info("Loading metadata...")
    # We read the full CSVs but will slice them manually to ensure the Dataset class
    # doesn't try to process the whole file even if debug is on (depending on implementation).
    # The provided Dataset class uses cfg.debug logic, but passing a small DF is safer for speed.
    df_train = pd.read_csv(cfg.train_path)
    df_val = pd.read_csv(cfg.val_path)
    df_test = pd.read_csv(cfg.test_path)

    # Slice for demo speed
    df_train = df_train.head(cfg.debug_sample_size).reset_index(drop=True)
    df_val = df_val.head(cfg.debug_sample_size).reset_index(drop=True)
    # Test set is small enough, but let's slice it too for consistency
    df_test = df_test.head(cfg.debug_sample_size).reset_index(drop=True)

    logger.info(f"Train shape: {df_train.shape}")
    logger.info(f"Val shape: {df_val.shape}")
    logger.info(f"Test shape: {df_test.shape}")

    # 3. CPC Mapping
    logger.info("Generating CPC Context Map...")
    cpc_texts = get_cpc_texts(cfg)

    # Verification: Check if mapping is populated
    assert len(cpc_texts) > 0, "CPC Context mapping is empty!"
    sample_code = list(cpc_texts.keys())[0]
    assert isinstance(cpc_texts[sample_code], str), "CPC description should be a string"
    logger.info(f"Loaded {len(cpc_texts)} CPC descriptions.")

    # 4. Tokenizer
    logger.info(f"Loading Tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 5. Dataset & DataLoader
    logger.info("Creating Datasets...")
    # Note: We set load_cached_data=False to ensure we demonstrate processing logic
    train_dataset = PearsonDataset(
        df_train, tokenizer, cpc_texts, mode="train", cfg=cfg, load_cached_data=False
    )
    val_dataset = PearsonDataset(
        df_val, tokenizer, cpc_texts, mode="val", cfg=cfg, load_cached_data=False
    )

    # Verification: Check dataset item structure
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "labels" in sample_item
    assert sample_item["input_ids"].shape[0] == cfg.max_length
    assert isinstance(sample_item["labels"], torch.Tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 6. Model Initialization
    logger.info("Initializing Model...")
    model = CustomModel(cfg, pretrained=True)
    model.to(cfg.device)

    # Verification: Dummy Forward Pass
    logger.info("Running dummy forward pass...")
    dummy_batch = next(iter(train_loader))
    for k, v in dummy_batch.items():
        dummy_batch[k] = v.to(cfg.device)

    with torch.no_grad():
        dummy_output = model(
            input_ids=dummy_batch["input_ids"],
            attention_mask=dummy_batch["attention_mask"],
            labels=dummy_batch["labels"],
        )

    assert "logits" in dummy_output
    assert "loss" in dummy_output
    assert dummy_output["logits"].shape == (
        cfg.train_batch_size,
    ), f"Expected logits shape ({cfg.train_batch_size},), got {dummy_output['logits'].shape}"
    logger.info("Model forward pass successful.")

    # 7. Optimizer & Scheduler
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=cfg.learning_rate,
        decoder_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_parameters)

    num_training_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * cfg.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # 8. Training Loop
    logger.info("Starting Training...")
    best_pearson = -1.0

    for epoch in range(cfg.epochs):
        # Train
        avg_loss = train_fn(
            train_loader,
            model,
            optimizer,
            epoch,
            scheduler,
            cfg.device,
            cfg,
            logger=logger,
        )
        assert not np.isnan(avg_loss), "Training loss is NaN!"

        # Validation
        val_loss, val_pearson, val_preds = valid_fn(
            valid_loader, model, cfg.device, cfg, logger=logger
        )
        assert not np.isnan(val_loss), "Validation loss is NaN!"
        assert -1.0 <= val_pearson <= 1.0, f"Pearson score {val_pearson} out of range!"

        logger.info(
            f"Epoch {epoch+1} Results - Loss: {val_loss:.4f}, Pearson: {val_pearson:.4f}"
        )

        # Save best model (simplified logic)
        if val_pearson > best_pearson:
            best_pearson = val_pearson
            torch.save(
                model.state_dict(), os.path.join(cfg.model_dir, "best_model.pth")
            )

    # 9. Inference on Test Set
    logger.info("Running Inference on Test Set...")
    test_dataset = PearsonDataset(
        df_test, tokenizer, cpc_texts, mode="test", cfg=cfg, load_cached_data=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Load best model weights
    model.load_state_dict(
        torch.load(
            os.path.join(cfg.model_dir, "best_model.pth"), map_location=cfg.device
        )
    )
    model.eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(cfg.device)
            mask = batch["attention_mask"].to(cfg.device)

            # Forward pass (no labels)
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                outputs = model(input_ids, mask)

            logits = outputs["logits"].view(-1).cpu().numpy()
            test_preds.append(logits)

    test_predictions = np.concatenate(test_preds)

    # 10. Create Submission
    submission = pd.DataFrame({"id": df_test["id"], "score": test_predictions})

    # Verification: Check submission format
    assert len(submission) == len(df_test)
    assert "id" in submission.columns and "score" in submission.columns

    submission.to_csv(cfg.submission_path, index=False)
    logger.info(f"Submission saved to {cfg.submission_path}")
    logger.info("Demo Completed Successfully.")


if __name__ == "__main__":
    main()
