import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, setup_logger, get_score
from library.cpc_utils import get_cpc_texts
from library.dataset import get_folds, CPCDataset, get_collate_fn
from library.model import CustomModel
from library.loss import HybridPearsonLoss
from library.engine import get_optimizer_params, AWP, EMA, train_fn, valid_fn


def run_demo():
    # ====================================================
    # 1. Configuration & Setup
    # ====================================================
    print(">>> [1/6] Configuring Demo Environment...")

    # Override CFG for speed and demo purposes
    CFG.debug = True
    CFG.epochs = 2
    CFG.train_batch_size = 8
    CFG.valid_batch_size = 16
    CFG.max_len = 64  # Shorter sequence length for speed
    CFG.output_dir = "./working/demo_execution"
    CFG.model_name = "microsoft/deberta-v3-base"  # Use base model for faster demo loading if possible, else fallback to large
    # Note: If base isn't cached/available, this might download.
    # Reverting to what is likely available or defined in config if needed,
    # but 'microsoft/deberta-v3-large' is heavy.
    # Let's stick to the config default to avoid download errors if 'base' isn't there,
    # but we will limit the number of steps significantly.
    CFG.model_name = "microsoft/deberta-v3-large"

    # Ensure reproducibility
    seed_everything(CFG.seed)

    # Setup Logger
    logger = setup_logger(os.path.join(CFG.output_dir, "demo.log"))
    logger.info("Starting End-to-End Demo...")

    # ====================================================
    # 2. Data Preparation
    # ====================================================
    logger.info(">>> [2/6] Preparing Data...")

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    except Exception as e:
        logger.warning(f"Could not load {CFG.model_name} from Hub/Cache. Error: {e}")
        # Fallback for offline environments if specific path needed,
        # but here we assume standard cache or internet access as per prompt context.
        raise e

    # Load CPC Context Texts
    cpc_texts = get_cpc_texts(CFG)
    assert isinstance(cpc_texts, dict), "CPC texts should be a dictionary"
    assert len(cpc_texts) > 0, "CPC texts should not be empty"
    logger.info(f"Loaded {len(cpc_texts)} CPC context descriptions.")

    # Load Folds
    # This reads ./metadata/train.csv and creates folds
    df = get_folds(CFG, load_cached_data=False)

    # SUBSET DATA FOR SPEED
    # Take 50 samples for training, 20 for validation
    train_subset = df[df["fold"] != 0].head(50).reset_index(drop=True)
    valid_subset = df[df["fold"] == 0].head(20).reset_index(drop=True)

    logger.info(f"Train subset shape: {train_subset.shape}")
    logger.info(f"Valid subset shape: {valid_subset.shape}")

    # Create Datasets
    train_dataset = CPCDataset(train_subset, tokenizer, cpc_texts, CFG.max_len)
    valid_dataset = CPCDataset(valid_subset, tokenizer, cpc_texts, CFG.max_len)

    # Create DataLoaders
    collate_fn = get_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Verify DataLoader output
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "labels" in sample_batch
    assert "ids" in sample_batch
    logger.info("DataLoader verification passed.")

    # ====================================================
    # 3. Model Initialization
    # ====================================================
    logger.info(">>> [3/6] Initializing Model...")

    device = CFG.device
    model = CustomModel(CFG, pretrained=True)
    model.to(device)

    # Dummy Forward Pass Check
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in sample_batch.items() if k != "ids"}
        outputs = model(**inputs)

    # Check Output Shapes
    # Score: (batch_size, 1)
    # Logits: (batch_size, 5)
    batch_size = sample_batch["input_ids"].size(0)
    assert outputs["score"].shape == (
        batch_size,
        1,
    ), f"Expected score shape ({batch_size}, 1), got {outputs['score'].shape}"
    assert outputs["logits"].shape == (
        batch_size,
        5,
    ), f"Expected logits shape ({batch_size}, 5), got {outputs['logits'].shape}"
    logger.info("Model forward pass shape verification passed.")

    # ====================================================
    # 4. Training Components Setup
    # ====================================================
    logger.info(">>> [4/6] Setting up Loss, Optimizer, AWP, EMA...")

    # Loss
    criterion = HybridPearsonLoss(CFG)

    # Optimizer with LLRD
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr,
        decoder_lr=CFG.head_lr,
        weight_decay=CFG.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=CFG.learning_rate, eps=1e-6)

    # Scheduler
    num_train_steps = int(len(train_subset) / CFG.train_batch_size * CFG.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps,
        num_cycles=CFG.num_cycles,
    )

    # AWP (Adversarial Weight Perturbation)
    awp = None
    if CFG.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=CFG.awp_lr,
            adv_eps=CFG.awp_eps,
            start_epoch=CFG.awp_start_epoch,
        )
        logger.info("AWP initialized.")

    # EMA (Exponential Moving Average)
    ema = None
    if CFG.use_ema:
        ema = EMA(model, CFG.ema_decay)
        ema.register()
        logger.info("EMA initialized.")

    # ====================================================
    # 5. Execution Loop (Train & Validate)
    # ====================================================
    logger.info(">>> [5/6] Starting Training Loop...")

    best_score = -1.0

    for epoch in range(CFG.epochs):
        start_time = pd.Timestamp.now()

        # Train
        avg_loss = train_fn(
            fold=0,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler,
            device=device,
            awp=awp,
            ema=ema,
        )

        # Validate
        val_loss, val_score, val_preds = valid_fn(
            valid_loader=valid_loader,
            model=model,
            criterion=criterion,
            device=device,
            ema=ema,
        )

        elapsed = pd.Timestamp.now() - start_time
        logger.info(
            f"Epoch {epoch+1}/{CFG.epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Pearson: {val_score:.4f} | "
            f"Time: {elapsed}"
        )

        # Assertions for sanity check
        assert not np.isnan(avg_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert (
            -1.0 <= val_score <= 1.0
        ), f"Pearson score {val_score} out of range [-1, 1]"

        if val_score > best_score:
            best_score = val_score
            # In a real run, we would save the model here
            # torch.save(model.state_dict(), os.path.join(CFG.output_dir, "best_model.pth"))

    logger.info(f"Training finished. Best Validation Score: {best_score:.4f}")

    # ====================================================
    # 6. Inference Demonstration
    # ====================================================
    logger.info(">>> [6/6] Running Inference on Test Metadata...")

    # Load Test Metadata
    test_df = pd.read_csv(CFG.test_path)
    # Subset for speed
    test_subset = test_df.head(20).reset_index(drop=True)

    test_dataset = CPCDataset(
        test_subset, tokenizer, cpc_texts, CFG.max_len, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model.eval()
    if ema:
        ema.apply_shadow()

    test_preds = []
    test_ids = []

    with torch.no_grad():
        for inputs in test_loader:
            ids = inputs["ids"]
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "ids"}

            outputs = model(**inputs)
            scores = outputs["score"].view(-1).cpu().numpy()

            test_preds.extend(scores)
            test_ids.extend(ids)

    if ema:
        ema.restore()

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "score": test_preds})

    # Verify Submission Format
    assert len(submission) == len(test_subset)
    assert "id" in submission.columns
    assert "score" in submission.columns
    assert submission["score"].dtype == float or submission["score"].dtype == np.float32

    # Save demo submission
    submission_path = os.path.join(CFG.output_dir, "submission_demo.csv")
    submission.to_csv(submission_path, index=False)
    logger.info(f"Demo submission saved to {submission_path}")
    logger.info(">>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
