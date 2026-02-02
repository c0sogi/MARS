import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

# Import from provided library
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import HierarchicalResNet
from library.engine import train_model, validate, generate_submission

# --- Configuration ---
SEED = 42
# Constraints for 8-minute runtime
SAMPLE_LIMIT = 10000
BATCH_SIZE_PHASE_1 = 128
BATCH_SIZE_PHASE_2 = 64
EPOCHS_PHASE_1 = 1
EPOCHS_PHASE_2 = 1
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GENUS_WEIGHT = 0.5
SUBMISSION_THRESHOLD = 0.35062931397784886
CHECKPOINT_DIR = "./working/idea_4/"


def get_validation_predictions(model, dataloader, device):
    """
    Helper to get raw predictions and targets for failure analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species_ids, _ in dataloader:
            images = images.to(device)
            # Forward pass (inference mode)
            species_logits, _ = model(images, species_label=None)
            preds = torch.argmax(species_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_ids.numpy())

    return np.array(all_preds), np.array(all_targets)


def perform_failure_analysis(model, val_loader, device, train_loader_ref):
    """
    Analyzes correlation between error rate and class frequency.
    """
    print("\n--- Failure Analysis ---")

    # 1. Get Predictions
    preds, targets = get_validation_predictions(model, val_loader, device)

    # 2. Calculate Per-Class Accuracy/Error
    # Create a DataFrame for analysis
    df_res = pd.DataFrame({"target": targets, "pred": preds})
    df_res["correct"] = df_res["target"] == df_res["pred"]

    # Group by target class to get accuracy per class
    class_stats = df_res.groupby("target")["correct"].mean().reset_index()
    class_stats.rename(columns={"correct": "accuracy"}, inplace=True)
    class_stats["error_rate"] = 1.0 - class_stats["accuracy"]

    # 3. Get Class Frequencies from Training Data
    # We need to access the underlying dataframe of the train set
    train_df = train_loader_ref.dataset.df
    class_counts = train_df["category_id"].value_counts().reset_index()
    class_counts.columns = ["target", "frequency"]

    # 4. Merge
    analysis_df = pd.merge(class_stats, class_counts, on="target", how="left")
    analysis_df["frequency"] = analysis_df["frequency"].fillna(0)

    # 5. Calculate Correlation
    # Using log frequency as distribution is likely long-tailed
    if len(analysis_df) > 1:
        corr, p_val = stats.pearsonr(
            np.log1p(analysis_df["frequency"]), analysis_df["error_rate"]
        )
        print(f"Correlation (Log Class Frequency vs Error Rate): {corr:.4f}")
        print(f"P-value: {p_val:.4f}")

        if corr < 0:
            print("Observation: Rare classes tend to have higher error rates.")
        else:
            print(
                "Observation: No negative correlation found (unexpected for long-tail)."
            )
    else:
        print("Not enough classes in validation subset for correlation analysis.")


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Phase 1: Low Resolution Training (160x160)
    print("\n=== Phase 1: Training at 160x160 ===")
    train_loader_1, val_loader_1, num_species, num_genera = get_dataloaders(
        image_size=160,
        batch_size=BATCH_SIZE_PHASE_1,
        num_workers=2,
        load_cached_data=True,
        sample_limit=SAMPLE_LIMIT,
    )

    print(f"Num Species: {num_species}, Num Genera: {num_genera}")

    # Initialize Model
    model = HierarchicalResNet(
        num_species=num_species,
        num_genera=num_genera,
        backbone_name="resnet50",
        pretrained=True,
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    # Train Phase 1
    model = train_model(
        model=model,
        train_loader=train_loader_1,
        val_loader=val_loader_1,
        optimizer=optimizer,
        device=device,
        num_epochs=EPOCHS_PHASE_1,
        genus_weight=GENUS_WEIGHT,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # 3. Phase 2: High Resolution Fine-tuning (224x224)
    print("\n=== Phase 2: Fine-tuning at 224x224 ===")
    train_loader_2, val_loader_2, _, _ = get_dataloaders(
        image_size=224,
        batch_size=BATCH_SIZE_PHASE_2,
        num_workers=2,
        load_cached_data=True,
        sample_limit=SAMPLE_LIMIT,
    )

    # Reduce LR for fine-tuning
    for param_group in optimizer.param_groups:
        param_group["lr"] = LR * 0.1

    # Train Phase 2
    model = train_model(
        model=model,
        train_loader=train_loader_2,
        val_loader=val_loader_2,
        optimizer=optimizer,
        device=device,
        num_epochs=EPOCHS_PHASE_2,
        genus_weight=GENUS_WEIGHT,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # 4. Final Validation
    print("\n=== Final Evaluation ===")
    final_f1 = validate(model, val_loader_2, device)
    print(f"Final Validation Metric: {final_f1}")

    # 5. Failure Analysis
    perform_failure_analysis(model, val_loader_2, device, train_loader_2)

    # 6. Submission
    if final_f1 > SUBMISSION_THRESHOLD:
        print("\nMetric threshold passed. Generating submission...")
        test_loader = get_test_dataloader(
            image_size=256,  # Resize to 256 then crop to 224 (handled in transforms)
            batch_size=BATCH_SIZE_PHASE_2,
            num_workers=2,
        )
        generate_submission(model, test_loader, device, output_dir="./submission")
        print("Submission generated.")
    else:
        print(
            f"\nMetric {final_f1} did not beat threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
