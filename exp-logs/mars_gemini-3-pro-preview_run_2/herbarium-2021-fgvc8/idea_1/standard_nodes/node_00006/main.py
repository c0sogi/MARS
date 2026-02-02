import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from library.utils import set_seed, get_taxonomy_mappings, calculate_f1_score
from library.model import MultiTaskResNet
from library.dataset import HerbariumDataset
from library.trainer import Trainer


def main():
    # 1. Configuration
    SEED = 42
    BATCH_SIZE = 512
    EPOCHS = 5
    LR = 1e-3
    # Limits for fast baseline execution
    TRAIN_SAMPLES = 250000
    VAL_SAMPLES = 10000
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "./working/idea_1/best_model.pth"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Set seed for reproducibility
    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    # 2. Taxonomy Mappings
    print("Loading taxonomy mappings...")
    (
        species_to_family,
        species_to_order,
        species_to_idx,
        idx_to_species,
        num_families,
        num_orders,
        num_species,
    ) = get_taxonomy_mappings()

    # 3. Data Preparation
    print("Preparing data...")
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    # Subsample for fast baseline
    if len(df_train) > TRAIN_SAMPLES:
        df_train = df_train.sample(n=TRAIN_SAMPLES, random_state=SEED).reset_index(
            drop=True
        )
    if len(df_val) > VAL_SAMPLES:
        df_val = df_val.sample(n=VAL_SAMPLES, random_state=SEED).reset_index(drop=True)

    # Calculate class frequencies in the training subset for failure analysis later
    train_class_counts = df_train["category_id"].value_counts().to_dict()

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    taxonomy_maps = (species_to_idx, species_to_family, species_to_order)
    train_dataset = HerbariumDataset(
        df_train, "./input", taxonomy_maps, train_transform
    )
    val_dataset = HerbariumDataset(df_val, "./input", taxonomy_maps, val_test_transform)
    test_dataset = HerbariumDataset(
        df_test, "./input", transform=val_test_transform, is_test=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("Initializing model...")
    model = MultiTaskResNet(num_species, num_families, num_orders).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Scheduler (OneCycleLR) (Cite solution_lesson_node_00002)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    # 5. Training
    print("Starting training...")
    trainer = Trainer(
        model, criterion, optimizer, DEVICE, SAVE_PATH, scheduler=scheduler
    )
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=2)

    # 6. Validation Assessment
    print("Evaluating on validation set...")
    # Reload best model for validation metrics
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images, species_targets, _, _ = batch
            images = images.to(DEVICE)

            species_out, _, _ = model(images)
            preds = torch.argmax(species_out, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(species_targets.numpy())

    val_f1 = calculate_f1_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_f1}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Error: 1 if incorrect, 0 if correct
    errors = (all_preds != all_targets).astype(int)

    # Feature 1: Class Frequency (in training set)
    # Map target indices back to category_id
    val_category_ids = [idx_to_species[idx] for idx in all_targets]
    val_freqs = [train_class_counts.get(cat_id, 0) for cat_id in val_category_ids]

    # Feature 2: File Size
    # We need to access the file paths from the dataframe corresponding to the validation set
    # Since val_loader is not shuffled (shuffle=False), the order matches df_val
    file_sizes = []
    for rel_path in df_val["file_path"]:
        full_path = os.path.join("./input", rel_path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except OSError:
            file_sizes.append(0)

    # Calculate correlations
    if len(errors) > 0:
        # Avoid division by zero in correlation if variance is 0
        if np.std(errors) > 0 and np.std(val_freqs) > 0:
            corr_freq = np.corrcoef(errors, val_freqs)[0, 1]
            print(f"Correlation between Error and Class Frequency: {corr_freq}")
        else:
            print(
                "Correlation between Error and Class Frequency: Undefined (zero variance)"
            )

        if np.std(errors) > 0 and np.std(file_sizes) > 0:
            corr_size = np.corrcoef(errors, file_sizes)[0, 1]
            print(f"Correlation between Error and File Size: {corr_size}")
        else:
            print("Correlation between Error and File Size: Undefined (zero variance)")

    # 8. Submission
    if val_f1 > 0.2867658284090583:
        print("Generating submission...")
        trainer.predict(test_loader, idx_to_species, SUBMISSION_PATH)
    else:
        print(f"Validation F1 ({val_f1}) not high enough for submission.")
    print("Done.")


if __name__ == "__main__":
    main()
