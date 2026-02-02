import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader

# Import provided library functions and classes
from library.utils import set_seed, process_taxonomy
from library.dataset import HerbariumDataset, get_transforms
from library.model import CascadedEfficientNet
from library.losses import FocalLoss
from library.engine import train_model, validate, predict_test_set


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Taxonomy Mapping
    print("Processing taxonomy...")
    # Generate or load the mapping from species -> genus -> family
    taxonomy_map = process_taxonomy(
        os.path.join(INPUT_DIR, "nybg2020/train/metadata.json"),
        output_dir=WORKING_DIR,
        load_cached_data=True,
    )

    # Determine number of classes for each hierarchical level
    num_species = taxonomy_map["species_label"].max() + 1
    num_genera = taxonomy_map["genus_label"].max() + 1
    num_families = taxonomy_map["family_label"].max() + 1

    print(
        f"Taxonomy: {num_species} Species, {num_genera} Genera, {num_families} Families"
    )

    # 3. Data Preparation (Fast Baseline)
    # Load full training metadata
    train_full_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))

    # Sample a subset of images for training to ensure the script finishes quickly.
    # 50,000 samples provide a good balance between coverage and speed for a baseline.
    SAMPLE_SIZE = 50000
    if len(train_full_df) > SAMPLE_SIZE:
        train_subset_df = train_full_df.sample(
            n=SAMPLE_SIZE, random_state=42
        ).reset_index(drop=True)
    else:
        train_subset_df = train_full_df

    # Save temporary training CSV
    temp_train_csv = os.path.join(WORKING_DIR, "train_subset.csv")
    train_subset_df.to_csv(temp_train_csv, index=False)
    print(f"Created temporary training subset with {len(train_subset_df)} images.")

    # Validation CSV path
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    # 4. Model Initialization
    print("Initializing CascadedEfficientNet...")
    model = CascadedEfficientNet(
        num_families=num_families,
        num_genera=num_genera,
        num_species=num_species,
        backbone_name="efficientnet_b3",
        pretrained=True,
    ).to(device)

    # Define Losses
    # Focal Loss for the imbalanced species classification
    criterion_species = FocalLoss(gamma=2.0)
    # Standard Cross Entropy for auxiliary heads
    criterion_aux = nn.CrossEntropyLoss()

    # 5. Phase 1: Coarse Alignment (224x224)
    print("\n=== Phase 1: Coarse Alignment (224x224) ===")
    BATCH_SIZE_P1 = 128
    IMG_SIZE_P1 = 224
    LR_P1 = 1e-3
    EPOCHS_P1 = 1

    # Datasets & Loaders for Phase 1
    train_dataset_p1 = HerbariumDataset(
        csv_path=temp_train_csv,
        taxonomy_map=taxonomy_map,
        transform=get_transforms(IMG_SIZE_P1, mode="train"),
        input_dir=INPUT_DIR,
    )
    val_dataset_p1 = HerbariumDataset(
        csv_path=val_csv_path,
        taxonomy_map=taxonomy_map,
        transform=get_transforms(IMG_SIZE_P1, mode="val"),
        input_dir=INPUT_DIR,
    )

    train_loader_p1 = DataLoader(
        train_dataset_p1,
        batch_size=BATCH_SIZE_P1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader_p1 = DataLoader(
        val_dataset_p1,
        batch_size=BATCH_SIZE_P1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer & Scheduler for Phase 1
    optimizer_p1 = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=1e-4)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=EPOCHS_P1
    )

    save_path_p1 = os.path.join(WORKING_DIR, "model_phase1.pth")

    # Train Phase 1
    model = train_model(
        model=model,
        train_loader=train_loader_p1,
        val_loader=val_loader_p1,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        criterion_species=criterion_species,
        criterion_aux=criterion_aux,
        device=device,
        num_epochs=EPOCHS_P1,
        save_path=save_path_p1,
        loss_weights=(1.0, 0.5, 0.5),
    )

    # 6. Phase 2: Fine-Grained Refinement (300x300)
    print("\n=== Phase 2: Fine-Grained Refinement (300x300) ===")
    BATCH_SIZE_P2 = 64  # Reduced batch size for larger image resolution
    IMG_SIZE_P2 = 300
    LR_P2 = 1e-4
    EPOCHS_P2 = 1

    # Datasets & Loaders for Phase 2
    train_dataset_p2 = HerbariumDataset(
        csv_path=temp_train_csv,
        taxonomy_map=taxonomy_map,
        transform=get_transforms(IMG_SIZE_P2, mode="train"),
        input_dir=INPUT_DIR,
    )
    val_dataset_p2 = HerbariumDataset(
        csv_path=val_csv_path,
        taxonomy_map=taxonomy_map,
        transform=get_transforms(IMG_SIZE_P2, mode="val"),
        input_dir=INPUT_DIR,
    )

    train_loader_p2 = DataLoader(
        train_dataset_p2,
        batch_size=BATCH_SIZE_P2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader_p2 = DataLoader(
        val_dataset_p2,
        batch_size=BATCH_SIZE_P2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer & Scheduler for Phase 2
    optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=LR_P2, weight_decay=1e-4)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=EPOCHS_P2
    )

    save_path_p2 = os.path.join(WORKING_DIR, "model_phase2.pth")

    # Train Phase 2 (continues from Phase 1 weights)
    model = train_model(
        model=model,
        train_loader=train_loader_p2,
        val_loader=val_loader_p2,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        criterion_species=criterion_species,
        criterion_aux=criterion_aux,
        device=device,
        num_epochs=EPOCHS_P2,
        save_path=save_path_p2,
        loss_weights=(1.0, 0.5, 0.5),
    )

    # 7. Final Validation & Failure Analysis
    print("\n=== Final Evaluation ===")
    # Perform final validation to get the exact metric
    val_loss, val_f1 = validate(
        model, val_loader_p2, criterion_species, criterion_aux, device
    )

    print(f"Final Validation Metric: {val_f1}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    all_preds = []
    all_labels = []

    # Collect predictions and labels
    with torch.no_grad():
        for images, targets in val_loader_p2:
            images = images.to(device)
            species_label = targets[0].to(device)  # Target 0 is Species

            logits, _, _ = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(species_label.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    errors = (all_preds != all_labels).astype(int)

    # Load validation dataframe to get metadata for correlation
    val_df = pd.read_csv(val_csv_path)

    if len(val_df) != len(errors):
        print(
            "Warning: Validation dataframe length mismatch with predictions. Skipping detailed correlation."
        )
    else:
        # Correlation 1: Error vs Class Frequency (in training set)
        train_counts = train_full_df["category_id"].value_counts().to_dict()
        val_freqs = val_df["category_id"].map(train_counts).fillna(0).values

        if np.std(errors) > 0 and np.std(val_freqs) > 0:
            corr_freq = np.corrcoef(errors, val_freqs)[0, 1]
            print(f"Correlation between Error and Class Frequency: {corr_freq:.4f}")
        else:
            print(
                "Correlation between Error and Class Frequency: Undefined (zero variance)"
            )

        # Correlation 2: Error vs Region ID
        val_regions = val_df["region_id"].values
        if np.std(errors) > 0 and np.std(val_regions) > 0:
            corr_region = np.corrcoef(errors, val_regions)[0, 1]
            print(f"Correlation between Error and Region ID: {corr_region:.4f}")
        else:
            print("Correlation between Error and Region ID: Undefined (zero variance)")

    # 8. Submission
    THRESHOLD = 0.43008749389564027
    if val_f1 > THRESHOLD:
        print(
            f"\nValidation F1 ({val_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_csv_path = os.path.join(METADATA_DIR, "test.csv")
        # Initialize test dataset (no taxonomy map needed)
        test_dataset = HerbariumDataset(
            csv_path=test_csv_path,
            taxonomy_map=None,
            transform=get_transforms(IMG_SIZE_P2, mode="test"),
            input_dir=INPUT_DIR,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_P2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        submission_path = "./submission/submission.csv"
        predict_test_set(model, test_loader, device, output_path=submission_path)
    else:
        print(
            f"\nValidation F1 ({val_f1}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
