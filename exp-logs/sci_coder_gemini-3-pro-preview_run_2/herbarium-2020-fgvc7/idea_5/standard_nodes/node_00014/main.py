import os
import random
import warnings
import numpy as np
import pandas as pd
import torch

# Import configuration and library modules
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    WORKING_DIR,
    SUBMISSION_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)
from library.taxonomy import TaxonomyManager
from library.dataset import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.engine import train_model, validate

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=SEED):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed()

    # 2. Data Preparation
    # Create a subset of training data for fast baseline execution (< 2 hours)
    full_train_df = pd.read_csv(TRAIN_CSV)

    # Use 50,000 samples (approx 8% of data) for training
    SUBSET_SIZE = 50000
    if len(full_train_df) > SUBSET_SIZE:
        train_subset_df = full_train_df.sample(n=SUBSET_SIZE, random_state=SEED)
    else:
        train_subset_df = full_train_df

    train_subset_path = os.path.join(WORKING_DIR, "train_subset.csv")
    train_subset_df.to_csv(train_subset_path, index=False)

    # Initialize Taxonomy Manager (loads from cache or raw metadata)
    # Force regeneration to ensure full coverage of species (Cite debug_lesson_4)
    taxonomy = TaxonomyManager(load_cached_data=False)

    # Get DataLoaders
    # We use the subset for training, but full validation/test sets
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv=train_subset_path,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # 3. Model Initialization
    model = HierarchicalEfficientNet(
        num_species_classes=taxonomy.get_num_species(),
        num_genus_classes=taxonomy.get_num_genus(),
        num_family_classes=taxonomy.get_num_family(),
    )
    model.to(DEVICE)

    # 4. Training
    # Limit to 3 epochs for fast baseline
    trained_model = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=3
    )

    # 5. Final Validation and Metric Calculation
    criterion = HierarchicalLoss()
    val_metrics = validate(trained_model, val_loader, criterion, DEVICE)
    final_metric = val_metrics["macro_f1"]

    # Required output
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    trained_model.eval()

    all_preds = []
    all_targets = []

    # Collect predictions on validation set
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            # Targets: (species, genus, family)
            species_targets = targets[0]

            outputs = trained_model(images)
            # Species head predictions
            preds = torch.argmax(outputs["species"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute binary error (1 if wrong, 0 if correct)
    errors = (all_preds != all_targets).astype(int)

    # Load validation metadata
    val_df = pd.read_csv(VAL_CSV)

    # Align lengths
    min_len = min(len(val_df), len(errors))
    val_df = val_df.iloc[:min_len].copy()
    errors = errors[:min_len]

    val_df["error"] = errors

    # Map IDs for correlation
    val_df["genus_id"] = val_df["category_id"].apply(taxonomy.get_genus_id)
    val_df["family_id"] = val_df["category_id"].apply(taxonomy.get_family_id)

    # Calculate correlations
    corr_region = val_df["region_id"].corr(val_df["error"])
    corr_genus = val_df["genus_id"].corr(val_df["error"])
    corr_family = val_df["family_id"].corr(val_df["error"])

    print("Correlation between model error and input features:")
    print(f"Region ID Correlation: {corr_region}")
    print(f"Genus ID Correlation: {corr_genus}")
    print(f"Family ID Correlation: {corr_family}")

    # 7. Submission Generation
    THRESHOLD = 0.43008749389564027

    if final_metric > THRESHOLD:
        test_preds = []
        test_ids = []

        trained_model.eval()
        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(DEVICE)

                outputs = trained_model(images)
                preds = torch.argmax(outputs["species"], dim=1)

                # Convert indices back to raw IDs
                pred_indices = preds.cpu().numpy()
                pred_raw_ids = [taxonomy.get_raw_id(idx) for idx in pred_indices]

                test_preds.extend(pred_raw_ids)
                test_ids.extend(image_ids.numpy())

        submission_df = pd.DataFrame({"Id": test_ids, "Predicted": test_preds})

        submission_df.sort_values("Id", inplace=True)

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
