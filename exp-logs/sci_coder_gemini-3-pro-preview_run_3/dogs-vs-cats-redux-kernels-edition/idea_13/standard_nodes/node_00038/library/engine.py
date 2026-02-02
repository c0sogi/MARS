import os
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.data import get_dataloaders
from library.modeling import create_model, get_optimizer_params, Trainer, predict


def train_model(config_name: str, load_cached_data: bool = True, patience: int = 3):
    """
    Trains a model based on the provided configuration name using the Trainer class.
    Implements Early Stopping and saves the best model checkpoint.

    Args:
        config_name (str): Key matching a configuration in Config.MODEL_CONFIGS.
        load_cached_data (bool): Whether to use cached metadata.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        float: The best validation loss achieved.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    cfg = Config.MODEL_CONFIGS[config_name]

    print(f"Starting training for configuration: {config_name}")

    # 2. Data
    # We only need train and val loaders for training
    train_loader, val_loader, _ = get_dataloaders(
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        load_cached_data=load_cached_data,
    )

    # 3. Model
    model = create_model(
        model_name=cfg["model_name"],
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        img_size=cfg["img_size"],
    )
    model.to(Config.DEVICE)

    # 4. Optimizer & Scheduler
    # Use get_optimizer_params to handle LLRD and Weight Decay properly
    optimizer_params = get_optimizer_params(
        model,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        use_llrd=cfg["use_llrd"],
        llrd_decay=cfg["llrd_decay"],
    )

    optimizer = AdamW(
        optimizer_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"])

    # 5. Trainer Initialization
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        config_name=config_name,
    )

    # 6. Training Loop with Early Stopping
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # Run one epoch
        train_loss = trainer.train_one_epoch(epoch)
        val_loss, metric_log_loss = trainer.validate()

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch}/{cfg['epochs']} - Train Loss: {train_loss:.15f} - Val Loss: {val_loss:.15f} - Val LogLoss: {metric_log_loss:.15f}"
        )

        # Checkpoint & Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            # Save Checkpoint
            save_path = os.path.join(Config.WORKING_DIR, f"{config_name}_best.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "config_name": config_name,
                },
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_loss:.15f}"
                )
                break

    return best_loss


def generate_submission(model_keys: list, load_cached_data: bool = True):
    """
    Generates predictions for the test set using an ensemble of trained models.
    Saves the result to submission.csv.

    Args:
        model_keys (list): List of configuration keys to include in the ensemble.
        load_cached_data (bool): Whether to use cached metadata.
    """
    seed_everything(Config.SEED)

    ensemble_predictions = {}  # Dict[id, List[float]]

    # Iterate over each model configuration
    for key in model_keys:
        print(f"Generating predictions for model: {key}")
        cfg = Config.MODEL_CONFIGS[key]

        # 1. Load Test Data (Specific to model resolution)
        _, _, test_loader = get_dataloaders(
            img_size=cfg["img_size"],
            batch_size=cfg["batch_size"],
            load_cached_data=load_cached_data,
        )

        # 2. Initialize Model
        model = create_model(
            model_name=cfg["model_name"],
            num_classes=Config.NUM_CLASSES,
            pretrained=False,  # We will load our own weights
            img_size=cfg["img_size"],
        )
        model.to(Config.DEVICE)

        # 3. Load Weights
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"{key}_best.pth")
        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint not found for {key} at {checkpoint_path}. Skipping."
            )
            continue

        load_checkpoint(checkpoint_path, model, device=Config.DEVICE)

        # 4. Predict (with TTA if enabled in Config)
        # predict returns dict {id: probability}
        preds = predict(model, test_loader, Config.DEVICE, tta_flip=Config.TTA_FLIP)

        # 5. Accumulate
        for img_id, prob in preds.items():
            if img_id not in ensemble_predictions:
                ensemble_predictions[img_id] = []
            ensemble_predictions[img_id].append(prob)

    # Aggregate Predictions (Mean)
    final_rows = []

    if not ensemble_predictions:
        print("Error: No predictions generated.")
        return

    # Sort IDs for clean output
    sorted_ids = sorted(ensemble_predictions.keys())

    for img_id in sorted_ids:
        probs = ensemble_predictions[img_id]
        avg_prob = sum(probs) / len(probs)
        final_rows.append({"id": int(img_id), "label": avg_prob})

    # Create DataFrame and Save
    submission_df = pd.DataFrame(final_rows)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
