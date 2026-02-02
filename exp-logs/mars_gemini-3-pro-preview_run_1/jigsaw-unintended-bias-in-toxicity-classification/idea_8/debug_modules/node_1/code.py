import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.data import get_dataloaders, load_and_process_data
from library.model import ToxicityModel
from library.loss import CompositeLoss, AWP
from library.metrics import JigsawEvaluator
from library.engine import train_fn, eval_fn, inference_fn


def main():
    print("Initializing Configuration...")
    # Initialize Config with debug=True to use a subset of data (default 5000)
    # We further reduce this subset size manually for this demonstration to ensure rapid execution.
    config = Config(debug=True, epochs=1, train_batch_size=4)
    config.output_dir = "./working/demo_run"
    config.train_subset_size = 64  # Extremely small subset for quick demo verification

    # Ensure clean slate
    if os.path.exists(config.output_dir):
        shutil.rmtree(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)
    device = get_device()
    print(f"Device: {device}")

    # Setup Logger
    logger = get_logger(os.path.join(config.output_dir, "demo.log"))
    logger.info("Starting Demo Script...")

    # ==========================================
    # 1. Data Loading Verification
    # ==========================================
    logger.info("Step 1: Loading Data...")
    # get_dataloaders handles loading, processing (weight calculation), and caching
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    logger.info(f"Train Batch Keys: {batch.keys()}")

    # Assertions for batch structure
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "target" in batch
    assert "aux_targets" in batch
    assert batch["input_ids"].shape[0] == config.train_batch_size
    assert batch["input_ids"].shape[1] <= config.max_len

    logger.info("Data Loading Verified.")

    # ==========================================
    # 2. Model Initialization & Forward Pass
    # ==========================================
    logger.info("Step 2: Initializing Model...")
    model = ToxicityModel(
        model_name=config.model_name,
        num_identity_classes=len(config.aux_identity_cols),
        num_aux_attack_classes=1 if config.aux_attack_col else 0,
    )
    model.to(device)

    # Move batch to device
    batch_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Forward pass
    logger.info("Running Forward Pass...")
    outputs = model(batch_device["input_ids"], batch_device["attention_mask"])

    # Verify Output Shapes
    # Toxicity head: (batch_size, 1)
    assert outputs["toxicity"].shape == (config.train_batch_size, 1)
    # Identity head: (batch_size, num_identity_classes)
    assert outputs["identity"].shape == (
        config.train_batch_size,
        len(config.aux_identity_cols),
    )
    # Attack head: (batch_size, 1)
    assert outputs["attack"].shape == (config.train_batch_size, 1)

    logger.info("Model Forward Pass Verified.")

    # ==========================================
    # 3. Loss Calculation Verification
    # ==========================================
    logger.info("Step 3: Calculating Loss...")
    criterion = CompositeLoss(config)

    loss, loss_dict = criterion(outputs, batch_device)

    logger.info(f"Loss Values: {loss_dict}")
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert "loss_total" in loss_dict
    assert "loss_rank" in loss_dict

    logger.info("Loss Calculation Verified.")

    # ==========================================
    # 4. Optimizer & AWP Setup
    # ==========================================
    logger.info("Step 4: Setting up Optimizer and AWP...")
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Scheduler
    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # AWP
    awp = AWP(model, optimizer, adv_lr=config.awp_lr, adv_eps=config.awp_eps)

    # Verify AWP methods run without error
    awp.attack_step()
    awp.restore()
    logger.info("Optimizer and AWP Setup Verified.")

    # ==========================================
    # 5. Training Loop (Engine Integration)
    # ==========================================
    logger.info("Step 5: Running Training Loop (1 Epoch)...")

    # We use the provided train_fn from library.engine
    # config.awp_start_epoch is usually 2, so AWP won't trigger in epoch 0, which is fine for speed.
    avg_loss = train_fn(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        awp,
        device,
        config,
        epoch=0,
        logger=logger,
    )

    assert avg_loss > 0
    logger.info(f"Training Loop Verified. Avg Loss: {avg_loss:.4f}")

    # ==========================================
    # 6. Evaluation & Metrics
    # ==========================================
    logger.info("Step 6: Running Evaluation...")

    # Load validation dataframe for metrics
    val_df = load_and_process_data(config, mode="val", load_cached_data=True)
    # Slice val_df to match the size of val_loader if debug mode truncated the loader?
    # In this demo setup, get_dataloaders doesn't truncate val/test based on debug flag in config.py logic,
    # only train is truncated. However, for speed, let's just run eval on the full val set
    # (it's fast enough for inference) or we can manually slice val_loader.
    # To be safe on time, we will just run the standard eval_fn.

    # For the purpose of this demo, we will limit the validation loop by manually slicing the loader
    # inside the loop if we were writing it, but eval_fn iterates the whole loader.
    # Given the constraints, we'll let it run. If it's too slow, we'd mock it, but
    # val set inference is usually fast. To be absolutely safe, let's mock a smaller val loader.

    # Create a tiny val loader for demo speed
    tiny_val_df = val_df.iloc[:100]
    from library.data import ToxicityDataset, collate_fn
    from torch.utils.data import DataLoader

    tiny_val_dataset = ToxicityDataset(
        tiny_val_df, train_loader.dataset.tokenizer, config, is_test=False
    )
    tiny_val_loader = DataLoader(
        tiny_val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_preds = eval_fn(model, tiny_val_loader, device)

    assert len(val_preds) == 100
    assert val_preds.min() >= 0.0 and val_preds.max() <= 1.0

    # Calculate Metrics
    evaluator = JigsawEvaluator(config)
    score, metrics = evaluator.evaluate(tiny_val_df, val_preds)

    logger.info(f"Validation Score: {score}")
    logger.info(f"Metrics: {metrics}")

    assert "final_score" in metrics
    assert "overall_auc" in metrics

    logger.info("Evaluation Verified.")

    # ==========================================
    # 7. Inference & Submission
    # ==========================================
    logger.info("Step 7: Running Inference on Test Set...")

    # Similarly, create a tiny test loader for speed
    test_df = load_and_process_data(config, mode="test", load_cached_data=True)
    tiny_test_df = test_df.iloc[:100]

    tiny_test_dataset = ToxicityDataset(
        tiny_test_df, train_loader.dataset.tokenizer, config, is_test=True
    )
    tiny_test_loader = DataLoader(
        tiny_test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    ids, preds = inference_fn(model, tiny_test_loader, device)

    assert len(ids) == 100
    assert len(preds) == 100

    # Create submission file
    submission_df = pd.DataFrame({"id": ids, "prediction": preds})
    submission_path = os.path.join(config.output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path)
    logger.info(f"Submission saved to {submission_path}")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
