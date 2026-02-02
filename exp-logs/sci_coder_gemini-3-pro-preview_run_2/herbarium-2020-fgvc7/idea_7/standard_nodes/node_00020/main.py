import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import importlib
import library.config

importlib.reload(library.config)
from library.config import Config
from library.utils import seed_everything
from library.taxonomy import TaxonomyMapper
from library.dataset import get_dataloaders, get_test_loader
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.engine import train_one_epoch, validate, generate_submission


def main():
    # 1. Configuration and Setup
    # We override epochs to fit within the 2-hour runtime limit while maintaining the strategy.
    # Phase 1: 1 epoch @ 224px (Fast initialization)
    # Phase 2: 2 epochs @ 300px (Fine-tuning)
    config = Config(epochs_p1=1, epochs_p2=2)
    seed_everything(config.SEED)

    print(f"Device: {config.DEVICE}")

    # 2. Taxonomy Mapping
    # Load or build the mapping from category_id to model indices
    mapper = TaxonomyMapper(config).load_or_build()

    # 3. Model Initialization
    model = HierarchicalEfficientNet(config, mapper)
    model = model.to(config.DEVICE)

    # 4. Loss and Optimizer
    loss_fn = HierarchicalLoss(config)

    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_P1, weight_decay=1e-4)

    # -------------------------------------------------------------------------
    # Phase 1: Coarse Training (224x224)
    # -------------------------------------------------------------------------
    print("\n=== Starting Phase 1 Training (224x224) ===")
    train_loader_p1, val_loader_p1, _ = get_dataloaders(config, phase="p1")

    for epoch in range(1, config.EPOCHS_P1 + 1):
        train_metrics = train_one_epoch(
            model, train_loader_p1, optimizer, loss_fn, config.DEVICE, epoch, config
        )
        # We don't strictly need to validate in P1 to save time, but we save a checkpoint
        torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    # -------------------------------------------------------------------------
    # Phase 2: Fine-Grained Training (300x300)
    # -------------------------------------------------------------------------
    print("\n=== Starting Phase 2 Training (300x300) ===")
    train_loader_p2, val_loader_p2, _ = get_dataloaders(config, phase="p2")

    # Update Learning Rate for Fine-tuning
    for param_group in optimizer.param_groups:
        param_group["lr"] = config.LR_P2

    best_f1 = 0.0

    for epoch in range(1, config.EPOCHS_P2 + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader_p2, optimizer, loss_fn, config.DEVICE, epoch, config
        )

        # Validate
        val_metrics = validate(model, val_loader_p2, loss_fn, config.DEVICE, config)
        current_f1 = val_metrics["macro_f1"]

        print(f"Phase 2 Epoch {epoch} Val F1: {current_f1:.6f}")

        # Save Best Model
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved with F1: {best_f1:.6f}")

    # -------------------------------------------------------------------------
    # Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Final Evaluation & Failure Analysis ===")

    # Load best model
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
    model.eval()

    # Run validation to get predictions and targets
    # We need to manually run inference loop to get arrays for analysis
    # reusing validate() logic but extracting data

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species_targets, _, _ in val_loader_p2:
            images = images.to(config.DEVICE, non_blocking=True)
            outputs = model(images)
            species_logits = outputs[0]
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(species_targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Final Metric
    from sklearn.metrics import f1_score

    final_f1 = f1_score(all_targets, all_preds, average="macro")

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis: Correlation with Class Frequency
    print("Performing Failure Analysis...")

    # Get per-class accuracy
    classes = np.unique(all_targets)
    class_accuracies = {}
    for cls in classes:
        mask = all_targets == cls
        if np.sum(mask) > 0:
            acc = np.mean(all_preds[mask] == all_targets[mask])
            class_accuracies[cls] = acc

    # Get training class counts
    train_df = pd.read_csv(config.TRAIN_CSV)
    # Map category_id in train_df to species_idx used by model
    train_df["species_idx"] = train_df["category_id"].map(mapper.species_to_idx)
    train_counts = train_df["species_idx"].value_counts().to_dict()

    # Align arrays for correlation
    accs = []
    counts = []
    for cls in class_accuracies:
        accs.append(class_accuracies[cls])
        counts.append(train_counts.get(cls, 0))

    # Calculate correlation
    if len(accs) > 1:
        corr, p_val = pearsonr(counts, accs)
        print(
            f"Correlation between Class Frequency and Accuracy: {corr:.4f} (p={p_val:.4f})"
        )

        # Log correlation with log-frequency as well (often more linear)
        log_counts = np.log1p(counts)
        log_corr, log_p_val = pearsonr(log_counts, accs)
        print(f"Correlation between Log(Class Frequency) and Accuracy: {log_corr:.4f}")
    else:
        print("Not enough classes in validation to compute correlation.")

    # -------------------------------------------------------------------------
    # Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.43008749389564027

    if final_f1 > threshold:
        print(
            f"\nMetric ({final_f1:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )

        # 1. Generate raw predictions (indices)
        test_loader = get_test_loader(config, img_size=config.IMG_SIZE_P2)
        generate_submission(model, test_loader, config.DEVICE, config)

        # 2. Map indices back to category_ids
        print("Mapping predictions to original Category IDs...")
        submission_df = pd.read_csv(config.SUBMISSION_PATH)

        # Use the idx_to_species dictionary from mapper
        # mapper.idx_to_species is {idx: category_id}
        submission_df["Predicted"] = submission_df["Predicted"].map(
            mapper.idx_to_species
        )

        # Save corrected submission
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(
            f"Final submission saved to {config.SUBMISSION_PATH} with correct Category IDs."
        )

    else:
        print(
            f"\nMetric ({final_f1:.6f}) <= Threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
