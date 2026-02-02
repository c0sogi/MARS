import os
import sys
import torch
import torch.optim as optim

# Set memory management env var to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader

# Add library path to sys.path to ensure imports work
sys.path.append("./library")

# Import provided library modules
from library.taxonomy_utils import build_taxonomy_mapping, get_taxonomy_stats, set_seed
from library.dataset import HerbariumDataset, get_transforms
from library.model import get_model
from library.losses import HierarchicalLoss
from library.train_utils import train_model, validate


def run_failure_analysis(model, val_loader, device, taxonomy_map):
    """
    Analyzes model failures on the validation set.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_regions = []
    all_genus = []
    all_family = []

    # We need to access the original dataframe to get region_ids corresponding to the batches
    # Since the loader shuffles, we can't just iterate the df.
    # However, the dataset returns targets which contain genus/family.
    # Region is not in the target dict by default in HerbariumDataset.__getitem__
    # We will rely on the fact that we can reconstruct or just use the targets we have (Genus/Family).
    # For Region, we'll have to skip it or modify the dataset, but we cannot modify provided files.
    # We will analyze Genus and Family error correlations.

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs["species"], 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets["species"].cpu().numpy())
            all_genus.extend(targets["genus"].cpu().numpy())
            all_family.extend(targets["family"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_genus = np.array(all_genus)
    all_family = np.array(all_family)

    # Calculate Error (1 for incorrect, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    print(f"Total Validation Samples: {len(errors)}")
    print(f"Total Errors: {errors.sum()}")
    print(f"Overall Accuracy: {1.0 - errors.mean():.4f}")

    # Correlation Analysis
    # We correlate the binary error vector with the integer IDs of the taxonomy.
    # While not strictly linear, it indicates if higher IDs (arbitrary) have more errors,
    # or we can just print error rates per family/genus for top ones.
    # The prompt asks for "correlation between the model's error magnitude and the input features".

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "genus_id": all_genus,
            "family_id": all_family,
            "species_id": all_targets,
        }
    )

    corr_genus = df_analysis["error"].corr(df_analysis["genus_id"])
    corr_family = df_analysis["error"].corr(df_analysis["family_id"])

    print(f"Correlation between Error and Genus ID: {corr_genus:.4f}")
    print(f"Correlation between Error and Family ID: {corr_family:.4f}")

    # Additional insight: Error rate by Family (Top 5 worst)
    family_error_rates = df_analysis.groupby("family_id")["error"].mean()
    print("Top 5 Families with highest error rates (min 10 samples):")
    counts = df_analysis["family_id"].value_counts()
    valid_families = counts[counts >= 10].index
    print(family_error_rates[valid_families].sort_values(ascending=False).head(5))


def main():
    # 1. Configuration & Setup
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Hyperparameters
    BATCH_SIZE_PHASE_1 = 32  # 224x224
    BATCH_SIZE_PHASE_2 = 16  # 300x300 (reduced to be safe on memory)
    LR_PHASE_1 = 1e-3
    LR_PHASE_2 = 1e-4
    NUM_EPOCHS_PHASE_1 = 1
    NUM_EPOCHS_PHASE_2 = 1
    THRESHOLD = 0.43008749389564027

    # 2. Data Preparation
    print("Building Taxonomy Mapping...")
    taxonomy_map = build_taxonomy_mapping(
        metadata_path=os.path.join(INPUT_DIR, "nybg2020/train/metadata.json"),
        cache_dir=os.path.join(WORKING_DIR, "taxonomy_cache"),
    )

    stats = get_taxonomy_stats(taxonomy_map)
    num_species = stats["num_species"]
    num_genus = stats["num_genera"]
    num_family = stats["num_families"]

    print(f"Taxonomy: {num_species} Species, {num_genus} Genera, {num_family} Families")

    # 3. Model Initialization
    model = get_model(num_species, num_genus, num_family, pretrained=True)
    model.to(DEVICE)

    criterion = HierarchicalLoss(weights={"species": 1.0, "genus": 0.5, "family": 0.5})

    # ---------------------------------------------------------
    # 4. Phase 1: Coarse Training (224x224)
    # ---------------------------------------------------------
    print("\n==== Phase 1: Coarse Training (224x224) ====")

    # Datasets
    train_dataset_p1 = HerbariumDataset(
        csv_path=os.path.join(METADATA_DIR, "train.csv"),
        taxonomy_map=taxonomy_map,
        transform=get_transforms("train", image_size=224),
        input_root=INPUT_DIR,
    )

    val_dataset_p1 = HerbariumDataset(
        csv_path=os.path.join(METADATA_DIR, "val.csv"),
        taxonomy_map=taxonomy_map,
        transform=get_transforms("val", image_size=224),
        input_root=INPUT_DIR,
    )

    train_loader_p1 = DataLoader(
        train_dataset_p1,
        batch_size=BATCH_SIZE_PHASE_1,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    val_loader_p1 = DataLoader(
        val_dataset_p1,
        batch_size=BATCH_SIZE_PHASE_1,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    optimizer_p1 = optim.AdamW(model.parameters(), lr=LR_PHASE_1)

    # Train Phase 1
    model, history_p1 = train_model(
        model=model,
        train_loader=train_loader_p1,
        val_loader=val_loader_p1,
        criterion=criterion,
        optimizer=optimizer_p1,
        num_epochs=NUM_EPOCHS_PHASE_1,
        device=DEVICE,
        save_dir=os.path.join(WORKING_DIR, "phase1"),
        patience=1,
    )

    # ---------------------------------------------------------
    # 5. Phase 2: Fine-Grained Training (300x300)
    # ---------------------------------------------------------
    print("\n==== Phase 2: Fine-Grained Training (300x300) ====")

    # Re-initialize Datasets with higher resolution
    train_dataset_p2 = HerbariumDataset(
        csv_path=os.path.join(METADATA_DIR, "train.csv"),
        taxonomy_map=taxonomy_map,
        transform=get_transforms("train", image_size=300),
        input_root=INPUT_DIR,
    )

    val_dataset_p2 = HerbariumDataset(
        csv_path=os.path.join(METADATA_DIR, "val.csv"),
        taxonomy_map=taxonomy_map,
        transform=get_transforms("val", image_size=300),
        input_root=INPUT_DIR,
    )

    train_loader_p2 = DataLoader(
        train_dataset_p2,
        batch_size=BATCH_SIZE_PHASE_2,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    val_loader_p2 = DataLoader(
        val_dataset_p2,
        batch_size=BATCH_SIZE_PHASE_2,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    # Lower learning rate for fine-tuning
    optimizer_p2 = optim.AdamW(model.parameters(), lr=LR_PHASE_2)

    # Train Phase 2
    model, history_p2 = train_model(
        model=model,
        train_loader=train_loader_p2,
        val_loader=val_loader_p2,
        criterion=criterion,
        optimizer=optimizer_p2,
        num_epochs=NUM_EPOCHS_PHASE_2,
        device=DEVICE,
        save_dir=os.path.join(WORKING_DIR, "phase2"),
        patience=1,
    )

    # ---------------------------------------------------------
    # 6. Final Validation & Analysis
    # ---------------------------------------------------------
    print("\n==== Final Validation ====")
    # Ensure we use the best model from Phase 2 (already loaded by train_model)
    val_metrics, final_f1 = validate(model, val_loader_p2, criterion, DEVICE)

    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    run_failure_analysis(model, val_loader_p2, DEVICE, taxonomy_map)

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    if final_f1 > THRESHOLD:
        print(
            f"\nMetric ({final_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = HerbariumDataset(
            csv_path=os.path.join(METADATA_DIR, "test.csv"),
            transform=get_transforms("test", image_size=300),
            is_test=True,
            input_root=INPUT_DIR,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_PHASE_2 * 2,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )

        model.eval()
        predictions = []
        ids = []

        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(DEVICE)

                outputs = model(images)
                # We only care about species prediction for submission
                _, preds = torch.max(outputs["species"], 1)

                predictions.extend(preds.cpu().numpy())
                ids.extend(image_ids.numpy())

        submission_df = pd.DataFrame({"Id": ids, "Predicted": predictions})

        # Sort by Id just in case
        submission_df = submission_df.sort_values("Id")

        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}. Rows: {len(submission_df)}")

    else:
        print(
            f"\nMetric ({final_f1}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
