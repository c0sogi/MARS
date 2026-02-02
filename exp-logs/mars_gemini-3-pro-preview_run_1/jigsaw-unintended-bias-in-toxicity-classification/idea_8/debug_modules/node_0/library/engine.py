import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.data import get_dataloaders, load_and_process_data
from library.model import ToxicityModel
from library.loss import CompositeLoss, AWP
from library.metrics import JigsawEvaluator


def move_batch_to_device(batch, device):
    """
    Moves all tensors in a batch dictionary to the specified device.
    """
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def train_fn(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    awp,
    device,
    config,
    epoch,
    logger,
):
    """
    Executes one epoch of training with AWP support.
    """
    model.train()

    running_loss = 0.0
    dataset_size = 0

    # Progress bar
    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch+1}/{config.epochs}",
    )

    for step, batch in pbar:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["input_ids"].size(0)

        # --- Forward Pass 1 (Clean) ---
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss, loss_dict = criterion(outputs, batch)

        # --- Backward Pass 1 ---
        loss.backward()

        # --- Adversarial Weight Perturbation (AWP) ---
        # We only apply AWP after the model has stabilized (awp_start_epoch)
        if config.use_awp and epoch >= config.awp_start_epoch:
            # 1. Save current weights and apply perturbation based on current gradients
            awp.attack_step()

            # 2. Forward Pass 2 (Adversarial)
            # Compute loss on the perturbed model
            outputs_adv = model(batch["input_ids"], batch["attention_mask"])
            loss_adv, _ = criterion(outputs_adv, batch)

            # 3. Backward Pass 2
            # We want to optimize the model to be robust to perturbations.
            # Therefore, we use the gradients from the adversarial loss.
            optimizer.zero_grad()  # Clear gradients from the clean pass
            loss_adv.backward()  # Populate gradients from the adversarial pass

            # 4. Restore original weights
            # The gradients from loss_adv.backward() remain in param.grad
            awp.restore()

        # --- Optimization Step ---
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Update progress bar with current loss and LR
        current_loss = running_loss / dataset_size
        pbar.set_postfix(
            loss=f"{current_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

    avg_loss = running_loss / dataset_size
    logger.info(f"Epoch {epoch+1} Training Loss: {avg_loss:.6f}")

    return avg_loss


def eval_fn(model, val_loader, device):
    """
    Runs inference on the validation set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = move_batch_to_device(batch, device)

            outputs = model(batch["input_ids"], batch["attention_mask"])

            # We only need the primary toxicity logits for evaluation
            logits = outputs["toxicity"].view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)

    return np.concatenate(preds)


def inference_fn(model, test_loader, device):
    """
    Runs inference on the test set.
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch = move_batch_to_device(batch, device)

            outputs = model(batch["input_ids"], batch["attention_mask"])
            logits = outputs["toxicity"].view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.append(probs)
            ids.append(batch["id"].cpu().numpy())

    return np.concatenate(ids), np.concatenate(preds)


def run():
    """
    Main execution function.
    """
    # 1. Configuration & Setup
    config = Config()
    seed_everything(config.seed)
    device = get_device()

    # Ensure output directories exist
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    logger = get_logger(os.path.join(config.output_dir, "train.log"))
    logger.info("Starting Engine...")

    # 2. Data Loading
    logger.info("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Load validation dataframe explicitly for metrics calculation
    # We use the same cache logic as the dataloaders
    val_df = load_and_process_data(config, mode="val", load_cached_data=True)

    # 3. Model Initialization
    logger.info(f"Initializing Model: {config.model_name}")
    model = ToxicityModel(
        model_name=config.model_name,
        num_identity_classes=len(config.aux_identity_cols),
        num_aux_attack_classes=1 if config.aux_attack_col else 0,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
        num_training_steps=num_training_steps,
    )

    # 5. Loss & AWP
    criterion = CompositeLoss(config)
    awp = AWP(model, optimizer, adv_lr=config.awp_lr, adv_eps=config.awp_eps)

    # 6. Evaluator
    evaluator = JigsawEvaluator(config)

    # 7. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(config.output_dir, "best_model.bin")

    for epoch in range(config.epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{config.epochs} =====")

        # Train
        train_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            awp,
            device,
            config,
            epoch,
            logger,
        )

        # Validate
        logger.info("Running Validation...")
        val_preds = eval_fn(model, val_loader, device)

        # Compute Metrics
        # val_preds aligns with val_df because val_loader is not shuffled and comes from same source
        score, metrics_dict = evaluator.evaluate(val_df, val_preds)

        logger.info(f"Validation Score: {score}")
        logger.info(f"Metrics: {metrics_dict}")

        # Checkpoint
        if score > best_score:
            logger.info(f"Score improved from {best_score} to {score}. Saving model...")
            best_score = score
            torch.save(model.state_dict(), best_model_path)
        else:
            logger.info(f"Score did not improve from {best_score}.")

    # 8. Inference
    logger.info("\nLoading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    ids, preds = inference_fn(model, test_loader, device)

    # 9. Submission
    submission_df = pd.DataFrame({"id": ids, "prediction": preds})

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")

    # Clean up
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
