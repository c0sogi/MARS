import os
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from library.utils import seed_everything, get_logger
from library.dataset import get_hierarchy_mappings, get_transforms, PlantDataset
from library.model import CascadingPlantModel
from library.engine import train_one_epoch, validate, generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 16
    IMAGE_SIZE = 128  # Small size for fast demonstration
    LR = 1e-3
    EPOCHS = 1
    SUBSET_SIZE_TRAIN = 200  # Limit samples for speed
    SUBSET_SIZE_VAL = 50
    SUBSET_SIZE_TEST = 50

    WORK_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    os.makedirs(WORK_DIR, exist_ok=True)

    seed_everything(SEED)
    logger = get_logger(os.path.join(WORK_DIR, "run.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------------------------------------------------------
    # 2. Hierarchy Mapping
    # ---------------------------------------------------------
    logger.info("Loading hierarchy mappings...")
    # This function parses the JSON to get family/genus/species relationships and counts
    hierarchy_df, num_families, num_genera, num_species = get_hierarchy_mappings(
        metadata_json_path=os.path.join(INPUT_DIR, "train_metadata.json"),
        load_cached_data=True,
        cache_dir=CACHE_DIR,
    )
    logger.info(
        f"Hierarchy Stats: Families={num_families}, Genera={num_genera}, Species={num_species}"
    )

    # ---------------------------------------------------------
    # 3. Data Loading (Train/Val)
    # ---------------------------------------------------------
    logger.info("Preparing datasets...")

    # Load metadata CSVs
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Subset for demonstration speed
    df_train = df_train.sample(
        n=min(len(df_train), SUBSET_SIZE_TRAIN), random_state=SEED
    ).reset_index(drop=True)
    df_val = df_val.sample(
        n=min(len(df_val), SUBSET_SIZE_VAL), random_state=SEED
    ).reset_index(drop=True)

    # Create Datasets
    # Note: We pass hierarchy_df to train/val datasets to get parent labels
    train_dataset = PlantDataset(
        df=df_train,
        root_dir=INPUT_DIR,
        hierarchy_df=hierarchy_df,
        transform=get_transforms(data="train", image_size=IMAGE_SIZE),
        is_test=False,
    )

    val_dataset = PlantDataset(
        df=df_val,
        root_dir=INPUT_DIR,
        hierarchy_df=hierarchy_df,
        transform=get_transforms(data="valid", image_size=IMAGE_SIZE),
        is_test=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    logger.info("Initializing model...")
    model = CascadingPlantModel(
        num_species=num_species,
        num_genera=num_genera,
        num_families=num_families,
        backbone_name="tf_efficientnetv2_b0",  # Small efficient backbone
        pretrained=True,
    )
    model = model.to(device)

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scheduler
        )

        # Validate
        val_f1, val_loss = validate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}"
        )

    # Save checkpoint
    checkpoint_path = os.path.join(WORK_DIR, "checkpoints", "best_model.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")

    # ---------------------------------------------------------
    # 6. Inference / Submission
    # ---------------------------------------------------------
    logger.info("Generating submission...")

    # Load Test Metadata
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Subset test data for speed
    df_test_subset = df_test.iloc[:SUBSET_SIZE_TEST].copy()

    # Create Test Dataset (is_test=True returns image and image_id)
    test_dataset = PlantDataset(
        df=df_test_subset,
        root_dir=INPUT_DIR,
        hierarchy_df=None,  # Not needed for test
        transform=get_transforms(data="test", image_size=IMAGE_SIZE),
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Generate Submission
    submission_path = os.path.join(WORK_DIR, "submission.csv")
    generate_submission(model, test_loader, device, save_path=submission_path)

    # ---------------------------------------------------------
    # 7. Verification
    # ---------------------------------------------------------
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        logger.info(f"Submission generated with shape: {sub_df.shape}")

        # Assertions to verify correctness
        assert (
            len(sub_df) == SUBSET_SIZE_TEST
        ), f"Expected {SUBSET_SIZE_TEST} predictions, got {len(sub_df)}"
        assert list(sub_df.columns) == [
            "Id",
            "Predicted",
        ], "Incorrect submission columns"
        assert not sub_df.isnull().values.any(), "Submission contains null values"

        logger.info("Verification successful. Demo completed.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
