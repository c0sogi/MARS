import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from library.utils import set_seed, calculate_f1_score, get_taxonomy_mappings


# ==========================================
# Model Architecture
# ==========================================
class MultiTaskResNet(nn.Module):
    def __init__(self, num_species, num_families, num_orders, pretrained=True):
        super(MultiTaskResNet, self).__init__()
        # Load ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)

        # Get the input dimension of the FC layer
        num_ftrs = self.backbone.fc.in_features

        # Replace the original FC layer with Identity to get features directly
        self.backbone.fc = nn.Identity()

        # Define the three heads
        self.species_head = nn.Linear(num_ftrs, num_species)
        self.family_head = nn.Linear(num_ftrs, num_families)
        self.order_head = nn.Linear(num_ftrs, num_orders)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Forward through heads
        species_out = self.species_head(features)
        family_out = self.family_head(features)
        order_out = self.order_head(features)

        return species_out, family_out, order_out


# ==========================================
# Dataset Class
# ==========================================
class HerbariumDataset(Dataset):
    def __init__(self, df, root_dir, taxonomy_maps=None, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            root_dir (str): Root directory for images (usually './input').
            taxonomy_maps (tuple): (species_to_idx, species_to_family, species_to_order).
                                   Required for training/validation.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is a test set (no labels).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        if not self.is_test and taxonomy_maps:
            self.species_to_idx, self.species_to_family, self.species_to_order = (
                taxonomy_maps
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen based on analysis)
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["image_id"]
        else:
            # Get raw category_id
            cat_id = row["category_id"]

            # Map to internal indices
            species_label = self.species_to_idx[cat_id]
            family_label = self.species_to_family[cat_id]
            order_label = self.species_to_order[cat_id]

            return image, species_label, family_label, order_label


# ==========================================
# Main Execution Function
# ==========================================
def train_and_predict(
    epochs=10,
    batch_size=128,
    learning_rate=1e-3,
    num_workers=4,
    device="cuda",
    seed=42,
    debug_limit=None,
):
    """
    Main function to train the MultiTask model and generate submission.

    Args:
        epochs (int): Max training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        num_workers (int): DataLoader workers.
        device (str): 'cuda' or 'cpu'.
        seed (int): Random seed.
        debug_limit (int, optional): If set, limits dataset size for debugging.
    """
    set_seed(seed)

    # Ensure working directories exist
    os.makedirs("./working/idea_1", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

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

    print(
        f"Taxonomy: {num_species} species, {num_families} families, {num_orders} orders."
    )

    # ---------------------------------------------------------
    # 1. Prepare Data
    # ---------------------------------------------------------
    print("Preparing datasets...")

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

    # Load Metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    if debug_limit:
        print(f"Debug mode: limiting train to {debug_limit} samples.")
        df_train = df_train.head(debug_limit)
        df_val = df_val.head(debug_limit)
        df_test = df_test.head(debug_limit)

    # Create Datasets
    taxonomy_maps = (species_to_idx, species_to_family, species_to_order)

    train_dataset = HerbariumDataset(
        df_train, "./input", taxonomy_maps, train_transform
    )
    val_dataset = HerbariumDataset(df_val, "./input", taxonomy_maps, val_test_transform)
    test_dataset = HerbariumDataset(
        df_test, "./input", transform=val_test_transform, is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 2. Initialize Model
    # ---------------------------------------------------------
    print("Initializing model...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = MultiTaskResNet(num_species, num_families, num_orders).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    print("Starting training...")

    best_f1 = -1.0
    patience = 3
    patience_counter = 0
    best_model_path = "./working/idea_1/best_model.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training Step
        for images, species_targets, family_targets, order_targets in train_loader:
            images = images.to(device)
            species_targets = species_targets.to(device)
            family_targets = family_targets.to(device)
            order_targets = order_targets.to(device)

            optimizer.zero_grad()

            species_out, family_out, order_out = model(images)

            loss_species = criterion(species_out, species_targets)
            loss_family = criterion(family_out, family_targets)
            loss_order = criterion(order_out, order_targets)

            # Weighted sum of losses
            total_loss = loss_species + 0.5 * loss_family + 0.5 * loss_order

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validation Step
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, species_targets, _, _ in val_loader:
                images = images.to(device)
                species_targets = species_targets.to(device)

                species_out, _, _ = model(images)
                preds = torch.argmax(species_out, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(species_targets.cpu().numpy())

        # Calculate Metric
        val_f1 = calculate_f1_score(all_targets, all_preds)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val F1: {val_f1}")

        # Early Stopping & Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  New best model saved with F1: {best_f1}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # ---------------------------------------------------------
    # 4. Inference
    # ---------------------------------------------------------
    print("Starting inference on test set...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    submission_rows = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            species_out, _, _ = model(images)
            preds = torch.argmax(species_out, dim=1).cpu().numpy()

            image_ids = image_ids.numpy()

            for img_id, pred_idx in zip(image_ids, preds):
                # Map internal index back to original category_id
                original_cat_id = idx_to_species[pred_idx]
                submission_rows.append({"Id": img_id, "Predicted": original_cat_id})

    # ---------------------------------------------------------
    # 5. Save Submission
    # ---------------------------------------------------------
    df_submission = pd.DataFrame(submission_rows)

    # Ensure correct sorting if needed (though Id is usually arbitrary, sorting by Id is good practice)
    df_submission = df_submission.sort_values("Id")

    output_path = "./submission/submission.csv"
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}. Total rows: {len(df_submission)}")
