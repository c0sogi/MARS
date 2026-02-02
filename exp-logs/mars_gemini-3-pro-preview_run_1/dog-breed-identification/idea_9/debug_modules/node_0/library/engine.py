import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import log_loss, accuracy_score
import torch.nn.functional as F

from library.config import Config
from library.utils import seed_everything
from library.dataset import DogDataset, get_transforms
from library.model import create_model
from library.loss import DistillationLoss


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    device: torch.device,
    criterion: DistillationLoss,
    epoch: int,
    config: Config,
):
    """
    Trains the model for one epoch.
    Handles both standard Hard Label training and Soft Target Distillation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Check for soft targets (Distillation)
        teacher_logits = None
        if "soft_target" in batch:
            teacher_logits = batch["soft_target"].to(device)

        optimizer.zero_grad()

        student_logits = model(images)

        # Compute loss (DistillationLoss handles None teacher_logits gracefully)
        loss = criterion(student_logits, labels, teacher_logits)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: DistillationLoss,
):
    """
    Evaluates the model on the validation set.
    Returns Loss, LogLoss, Accuracy, and Logits (for OOF generation).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)

            # Validation always uses hard labels (teacher_logits=None)
            loss = criterion(logits, labels, teacher_logits=None)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # Compute Probabilities for Metrics
    all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()

    # Calculate Metrics
    # Log Loss
    try:
        # Provide labels explicitly to handle cases where a batch might miss a class
        metric_log_loss = log_loss(
            all_labels, all_probs, labels=list(range(all_probs.shape[1]))
        )
    except ValueError:
        metric_log_loss = avg_loss

    # Accuracy
    preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, preds)

    return avg_loss, metric_log_loss, acc, all_logits


def predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Generates predictions for the test set.
    Applies Test Time Augmentation (Horizontal Flip).
    Returns probabilities and IDs.
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            ids = batch["id"]

            # Original Prediction
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.softmax(logits_flipped, dim=1)

            # Average Probabilities
            avg_probs = (probs + probs_flipped) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_probs), all_ids


def run_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    data_dict: dict,
    teacher_oof_df: pd.DataFrame,
    config: Config,
):
    """
    Runs the training and evaluation for a single fold.
    Implements SWA and Knowledge Distillation logic.
    """
    device = config.device
    seed_everything(config.seed + fold_idx)

    print(f"--- Starting Fold {fold_idx} ---")

    # 1. Prepare Data
    train_images = data_dict["train_images"][train_idx]
    train_ids = data_dict["train_ids"][train_idx]
    train_labels = data_dict["train_labels"][train_idx]

    val_images = data_dict["train_images"][val_idx]
    val_ids = data_dict["train_ids"][val_idx]
    val_labels = data_dict["train_labels"][val_idx]

    # 2. Prepare Soft Targets (if Teacher OOF is provided)
    soft_targets = None
    if teacher_oof_df is not None:
        print(f"Fold {fold_idx}: Loading Soft Targets for Distillation...")
        # Ensure column order matches label_map
        sorted_breeds = sorted(data_dict["label_map"], key=data_dict["label_map"].get)

        # Index by ID for fast retrieval
        teacher_df_indexed = teacher_oof_df.set_index("id")

        # Retrieve logits for current train IDs
        # Ensure IDs are strings for lookup
        train_ids_str = [str(x) for x in train_ids]

        try:
            matched_logits = teacher_df_indexed.loc[train_ids_str, sorted_breeds].values
            soft_targets = matched_logits.astype(np.float32)
        except KeyError as e:
            raise KeyError(
                f"Fold {fold_idx}: Failed to match teacher logits. Missing ID: {e}"
            )

    # 3. Create Datasets and Loaders
    train_dataset = DogDataset(
        train_images,
        train_ids,
        train_labels,
        soft_targets,
        get_transforms(config, "train"),
    )
    val_dataset = DogDataset(
        val_images, val_ids, val_labels, None, get_transforms(config, "val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = create_model(config, pretrained=config.pretrained)
    model.to(device)

    # 5. Optimizer (Discriminative Learning Rates)
    backbone_ids = list(map(id, model.backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.lr_backbone},
            {"params": head_params, "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )

    # 6. Schedulers
    # Main scheduler for standard training phase
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    # SWA Setup
    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)

    criterion = DistillationLoss(config)

    best_log_loss = float("inf")
    best_model_path = os.path.join(
        config.working_dir, f"{config.model_name}_fold_{fold_idx}.pth"
    )

    # 7. Training Loop
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, criterion, epoch, config
        )

        # SWA Logic
        is_swa_phase = config.use_swa and (epoch >= config.swa_start_epoch)

        if is_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            # Evaluate current model to monitor progress (SWA model is evaluated at end)
            val_loss, val_ll, val_acc, _ = evaluate(
                model, val_loader, device, criterion
            )
            print(
                f"Fold {fold_idx} Epoch {epoch} [SWA] | Train: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LogLoss: {val_ll:.6f} | Acc: {val_acc:.6f}"
            )
        else:
            scheduler.step()
            val_loss, val_ll, val_acc, _ = evaluate(
                model, val_loader, device, criterion
            )
            print(
                f"Fold {fold_idx} Epoch {epoch} | Train: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LogLoss: {val_ll:.6f} | Acc: {val_acc:.6f}"
            )

            # Save best model (Pre-SWA)
            if val_ll < best_log_loss:
                best_log_loss = val_ll
                # We save the model state here. If SWA is enabled, this might be overwritten by SWA model later.
                # However, if SWA fails to improve, we might want this.
                # For this strategy, we prioritize SWA result if enabled.
                if not config.use_swa:
                    torch.save(model.state_dict(), best_model_path)

    # 8. Finalize Model
    final_model = model
    if config.use_swa:
        print(f"Fold {fold_idx}: Finalizing SWA Model (Updating Batch Norm)...")
        update_bn(train_loader, swa_model, device=device)
        final_model = swa_model
        # Save SWA model as the fold model
        torch.save(final_model.state_dict(), best_model_path)
    elif best_log_loss < float("inf"):
        # Reload best model if not using SWA
        model.load_state_dict(torch.load(best_model_path))
        final_model = model

    # 9. Final Validation & OOF Generation
    _, final_ll, final_acc, final_logits = evaluate(
        final_model, val_loader, device, criterion
    )
    print(
        f"Fold {fold_idx} Final Result | LogLoss: {final_ll:.10f} | Acc: {final_acc:.10f}"
    )

    # Construct OOF DataFrame with Logits (for Distillation or Stacking)
    sorted_breeds = sorted(data_dict["label_map"], key=data_dict["label_map"].get)
    oof_df = pd.DataFrame(final_logits, columns=sorted_breeds)
    oof_df.insert(0, "id", [str(x) for x in val_ids])

    return oof_df, final_ll


def predict_and_submit(config: Config, data_dict: dict, fold_models: list):
    """
    Generates predictions for the test set using an ensemble of fold models.
    Saves the result to submission.csv.
    """
    print(f"\nGenerating predictions with {len(fold_models)} models...")

    device = config.device
    test_images = data_dict["test_images"]
    test_ids = data_dict["test_ids"]

    test_dataset = DogDataset(
        test_images, test_ids, transform=get_transforms(config, "test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    avg_probs = None

    for model_path in fold_models:
        # Load model
        model = create_model(config, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Predict (Returns Probs)
        probs, ids = predict(model, test_loader, device)

        if avg_probs is None:
            avg_probs = probs
        else:
            avg_probs += probs

    # Average over folds
    avg_probs /= len(fold_models)

    # Create Submission DataFrame
    sorted_breeds = sorted(data_dict["label_map"], key=data_dict["label_map"].get)
    sub_df = pd.DataFrame(avg_probs, columns=sorted_breeds)
    sub_df.insert(0, "id", [str(x) for x in ids])

    # Save
    sub_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
