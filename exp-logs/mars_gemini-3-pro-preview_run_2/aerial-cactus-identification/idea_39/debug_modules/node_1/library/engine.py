import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.utils import set_seed, get_device, AverageMeter
from library.dataset import get_dataloaders
from library.model import (
    UltraWideDBBResNeXt,
    train_one_epoch,
    validate,
    predict_with_tta,
)


def inference_with_tta(model, loader, device):
    """
    Performs Test Time Augmentation (TTA) during inference.
    Wraps the library function to provide a consistent interface.

    Args:
        model: The trained PyTorch model (in deploy mode).
        loader: DataLoader for the test set.
        device: The computing device.

    Returns:
        tuple: (ids, predictions)
    """
    return predict_with_tta(model, loader, device)


def run_training_pipeline(
    epochs=20,
    batch_size=64,
    seeds=[0, 1, 2, 3, 4],
    patience=5,
    load_cached_data=True,
    base_lr=1e-3,
    weight_decay=1e-4,
    submission_dir="./submission",
    working_dir="./working/idea_39",
):
    """
    Executes the full training and inference pipeline.

    Args:
        epochs (int): Maximum number of training epochs per seed.
        batch_size (int): Batch size for DataLoaders.
        seeds (list): List of random seeds for ensemble training.
        patience (int): Early stopping patience (epochs without improvement).
        load_cached_data (bool): Whether to load pre-processed data from cache.
        base_lr (float): Initial learning rate.
        weight_decay (float): Weight decay for the optimizer.
        submission_dir (str): Directory to save the submission file.
        working_dir (str): Directory for caching intermediate files.
    """
    device = get_device()
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)

    all_preds = []
    test_ids = None

    for seed in seeds:
        print(f"Starting training for Seed {seed}")
        set_seed(seed)

        # 1. Data Loading
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=batch_size,
            num_workers=2,
            load_cached_data=load_cached_data,
            seed=seed,
        )

        # 2. Model Initialization
        # Using Ultra-Wide DBB-SE-ResNeXt with 32 groups
        model = UltraWideDBBResNeXt(groups=32).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # 3. Training Loop with Early Stopping
        best_auc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            # Print metrics with full precision
            print(
                f"Seed {seed} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1} for Seed {seed}"
                    )
                    break

        print(f"Seed {seed} Best Val AUC: {best_auc:.10f}")

        # 4. Inference
        # Load best weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Switch to efficient inference mode (fuse DBB branches)
        model.switch_to_deploy()

        # Predict with Test Time Augmentation
        ids, preds = inference_with_tta(model, test_loader, device)

        if test_ids is None:
            test_ids = ids

        all_preds.append(preds)

    # 5. Aggregation
    # Arithmetic mean of predictions across all seeds
    final_preds = np.mean(all_preds, axis=0)

    # 6. Submission
    submission_path = os.path.join(submission_dir, "submission.csv")
    df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
