import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BCEWithLogitsLoss

from library.config import Config
from library.dataset import get_dataloaders
from library.model import BirdResNet18
from library.engine import train_one_epoch, validate_one_epoch
from library.utils import seed_everything, get_logger, compute_metric, ensure_dir


def predict_loader(model, loader, device):
    """
    Runs inference on a dataloader and returns predictions, targets, and recording IDs.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for inputs, labels, rec_ids in loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            # Handle rec_ids whether they are tensors or lists
            if isinstance(rec_ids, torch.Tensor):
                all_ids.append(rec_ids.numpy())
            else:
                all_ids.append(np.array(rec_ids))

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_ids),
    )


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    ensure_dir(Config.WORKING_DIR)
    logger = get_logger(os.path.join(Config.WORKING_DIR, "train.log"))
    device = Config.DEVICE

    logger(f"Starting training on device: {device}")

    # Containers for OOF (Out-Of-Fold) results and Test Predictions
    oof_results = []
    test_preds_sum = None
    test_ids = None

    # 2. Cross-Validation Loop
    for fold in range(Config.NUM_FOLDS):
        logger(f"\n{'='*20} Fold {fold} {'='*20}")

        # Get DataLoaders
        # load_cached_data=True ensures we use pre-processed spectrograms if available
        train_loader, val_loader, test_loader = get_dataloaders(
            fold_idx=fold, load_cached_data=True
        )

        # Initialize Model
        model = BirdResNet18()
        model.to(device)

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
        )
        criterion = BCEWithLogitsLoss()

        # Training State
        best_auc = 0.0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}_best.pth")
        patience_counter = 0

        # Epoch Loop
        for epoch in range(Config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)
            scheduler.step()

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                logger(f"Early stopping at epoch {epoch+1}")
                break

        logger(f"Fold {fold} Best AUC: {best_auc:.4f}")

        # 3. Inference for this Fold
        # Load best weights
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # OOF Inference (Predict on the validation fold)
        val_probs, val_targets, val_ids = predict_loader(model, val_loader, device)

        # Store OOF results
        for i in range(len(val_ids)):
            oof_results.append(
                {"rec_id": val_ids[i], "probs": val_probs[i], "targets": val_targets[i]}
            )

        # Test Inference (Accumulate for Ensemble)
        t_probs, _, t_ids = predict_loader(model, test_loader, device)

        if test_preds_sum is None:
            test_preds_sum = t_probs
            test_ids = t_ids
        else:
            test_preds_sum += t_probs

    # ==========================================
    # Final Validation Assessment
    # ==========================================
    logger("\nComputing Final Metrics...")

    # Load the official hold-out validation metadata to identify the correct subset
    val_metadata_path = Config.VAL_METADATA_PATH
    val_df_meta = pd.read_csv(val_metadata_path)
    target_val_ids = set(val_df_meta["rec_id"].values)

    # Filter OOF results to match the hold-out validation set
    final_preds = []
    final_targets = []
    final_ids = []

    # Data for failure analysis
    bce_losses = []
    num_labels_list = []

    for item in oof_results:
        rid = item["rec_id"]
        if rid in target_val_ids:
            final_ids.append(rid)
            final_preds.append(item["probs"])
            final_targets.append(item["targets"])

            # Calculate BCE Loss for Failure Analysis
            # Clip probabilities to avoid log(0)
            p = np.clip(item["probs"], 1e-7, 1 - 1e-7)
            y = item["targets"]
            # BCE = -mean(y*log(p) + (1-y)*log(1-p))
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            bce_losses.append(loss)

            num_labels_list.append(np.sum(y))

    final_preds = np.array(final_preds)
    final_targets = np.array(final_targets)

    # Compute and Print Final Metric
    final_metric = compute_metric(final_targets, final_preds)
    print(f"Final Validation Metric: {final_metric}")
    logger(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # Failure Analysis
    # ==========================================
    logger("\nFailure Analysis:")
    if len(bce_losses) > 0:
        # Correlation between Error (BCE Loss) and Input Feature (Label Count)
        correlation = np.corrcoef(bce_losses, num_labels_list)[0, 1]
        print(f"Correlation between Error (BCE Loss) and Label Count: {correlation}")
        logger(f"Correlation between Error (BCE Loss) and Label Count: {correlation}")
    else:
        logger("No validation samples found for failure analysis.")

    # ==========================================
    # Submission Generation
    # ==========================================
    THRESHOLD = 0.9072993371210134
    if final_metric > THRESHOLD:
        logger(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Average probabilities across 5 folds
        avg_test_probs = test_preds_sum / Config.NUM_FOLDS

        submission_rows = []
        for i in range(len(test_ids)):
            rec_id = int(test_ids[i])
            probs = avg_test_probs[i]

            # Flatten predictions: Id = rec_id * 100 + species_idx
            for species_idx in range(Config.NUM_CLASSES):
                row_id = rec_id * 100 + species_idx
                prob = probs[species_idx]
                submission_rows.append({"Id": row_id, "Probability": prob})

        sub_df = pd.DataFrame(submission_rows)
        # Sort by Id for consistency
        sub_df = sub_df.sort_values("Id")

        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        logger(f"Submission saved to {sub_path}")
    else:
        logger(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
