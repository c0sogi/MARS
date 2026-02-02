import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import library components
from library.config import cfg
from library.utils import seed_everything, get_logger
from library.dataset import prepare_data, PearsonDataset, CollateFn
from library.model import CustomModel
from library.loss import CompositeLoss
from library.engine import get_optimizer_params, train_fn, valid_fn, AWP


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    # Ensure reproducibility
    seed_everything(cfg.seed)

    # Initialize Logger
    logger = get_logger(os.path.join(cfg.working_dir, "demo_script.log"))
    logger.info("Starting Phrase Matching Demo Script...")

    # Override Config for Speed and Demonstration
    logger.info("Overriding configuration for fast demonstration...")
    cfg.debug = True  # Use small subset of data
    cfg.epochs = 1
    cfg.batch_size = 4
    cfg.print_freq = 5
    cfg.working_dir = "./working/demo_execution"
    cfg.train_cache_path = os.path.join(cfg.working_dir, "train_cache.parquet")
    cfg.val_cache_path = os.path.join(cfg.working_dir, "val_cache.parquet")
    cfg.test_cache_path = os.path.join(cfg.working_dir, "test_cache.parquet")

    os.makedirs(cfg.working_dir, exist_ok=True)

    device = cfg.device
    logger.info(f"Device: {device}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    logger.info("Step 2: Preparing Data...")

    # Load Train and Val data (Debug mode will sample 100 rows)
    # Note: prepare_data handles CPC mapping internally
    df_train = prepare_data("train", load_cached_data=False, debug=True)
    df_val = prepare_data("val", load_cached_data=False, debug=True)

    logger.info(f"Train Data Shape: {df_train.shape}")
    logger.info(f"Val Data Shape: {df_val.shape}")

    # Verification: Check if context text was merged correctly
    assert (
        "context_text" in df_train.columns
    ), "context_text column missing in train data"
    sample_context = df_train.iloc[0]["context_text"]
    logger.info(f"Sample Context Text: {sample_context}")
    assert (
        isinstance(sample_context, str) and len(sample_context) > 0
    ), "Context text is empty"

    # ==========================================
    # 3. Tokenizer & Dataset
    # ==========================================
    logger.info("Step 3: Initializing Tokenizer and Dataset...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Create Datasets
    train_dataset = PearsonDataset(df_train, tokenizer, max_len=cfg.max_len)
    val_dataset = PearsonDataset(df_val, tokenizer, max_len=cfg.max_len)

    # Verification: Check Dataset Item
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "labels" in sample_item
    assert "class_labels" in sample_item
    logger.info("Dataset item keys verified.")

    # Create DataLoaders
    collate_fn = CollateFn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for simple script execution
        pin_memory=True,
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Verification: Check Batch Structure
    sample_batch = next(iter(train_loader))
    logger.info(f"Batch Input IDs Shape: {sample_batch['input_ids'].shape}")
    assert sample_batch["input_ids"].shape[0] == cfg.batch_size, "Batch size mismatch"
    assert sample_batch["labels"].shape[0] == cfg.batch_size, "Labels shape mismatch"

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    logger.info("Step 4: Initializing Model...")

    model = CustomModel()
    model.to(device)

    # Verification: Forward Pass
    logger.info("Verifying Forward Pass...")
    model.eval()
    with torch.no_grad():
        # Move batch to device
        inputs = {
            k: v.to(device)
            for k, v in sample_batch.items()
            if isinstance(v, torch.Tensor)
        }
        outputs = model(inputs["input_ids"], inputs["attention_mask"])

    assert "logits" in outputs
    assert "class_logits" in outputs
    assert outputs["logits"].shape == (
        cfg.batch_size,
        1,
    ), "Regression logits shape incorrect"
    assert outputs["class_logits"].shape == (
        cfg.batch_size,
        cfg.num_classes,
    ), "Class logits shape incorrect"
    logger.info("Forward pass successful.")

    # ==========================================
    # 5. Loss Function
    # ==========================================
    logger.info("Step 5: Initializing Loss Function...")

    criterion = CompositeLoss()

    # Verification: Loss Calculation
    # We need to pass the batch dictionary (containing labels) to the criterion
    # Ensure inputs dictionary has labels on device
    inputs["labels"] = sample_batch["labels"].to(device)
    inputs["class_labels"] = sample_batch["class_labels"].to(device)

    loss_dict = criterion(outputs, inputs)
    logger.info(f"Calculated Loss: {loss_dict['loss'].item():.4f}")
    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN"
    assert (
        "mse" in loss_dict and "pearson" in loss_dict and "ce" in loss_dict
    ), "Missing loss components"

    # ==========================================
    # 6. Training Loop (Demo)
    # ==========================================
    logger.info("Step 6: Running Training Loop (1 Epoch)...")

    # Optimizer
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=cfg.learning_rate,
        decoder_lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=cfg.learning_rate, eps=cfg.eps, betas=cfg.betas
    )

    # Scheduler
    num_train_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * cfg.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # AWP (Adversarial Weight Perturbation)
    awp = AWP(model, optimizer, adv_lr=cfg.awp_lr, adv_eps=cfg.awp_eps)

    # Train for 1 Epoch
    avg_loss = train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
    )
    logger.info(f"Training Epoch 0 Complete. Avg Loss: {avg_loss:.4f}")

    # ==========================================
    # 7. Validation Loop
    # ==========================================
    logger.info("Step 7: Running Validation...")

    val_loss, val_score, val_preds = valid_fn(
        valid_loader=valid_loader, model=model, criterion=criterion, device=device
    )
    logger.info(
        f"Validation Complete. Loss: {val_loss:.4f}, Pearson Score: {val_score:.4f}"
    )

    assert len(val_preds) == len(
        df_val
    ), "Number of predictions does not match validation set size"

    # ==========================================
    # 8. Inference on Test Set
    # ==========================================
    logger.info("Step 8: Inference on Test Set...")

    # Prepare Test Data
    df_test = prepare_data("test", load_cached_data=False, debug=True)
    test_dataset = PearsonDataset(df_test, tokenizer, max_len=cfg.max_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Inference Loop (Reusing valid_fn logic partially)
    model.eval()
    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(batch["input_ids"], batch["attention_mask"])

            preds = outputs["logits"].view(-1).cpu().numpy()
            test_preds.append(preds)

    test_predictions = np.concatenate(test_preds)

    # Create Submission
    submission = pd.DataFrame({"id": df_test["id"], "score": test_predictions})

    # Clip scores to valid range [0, 1] as per metric definition
    submission["score"] = submission["score"].clip(0, 1)

    submission_path = os.path.join(cfg.working_dir, "submission_demo.csv")
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
    logger.info(f"Submission Head:\n{submission.head()}")

    logger.info("Demo Script Completed Successfully.")


if __name__ == "__main__":
    run_demo()
