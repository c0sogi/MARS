import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from library.utils import set_seed, calculate_f1_score, get_taxonomy_mappings
from library.model import MultiTaskResNet
from library.dataset import HerbariumDataset


class Trainer:
    def __init__(self, model, criterion, optimizer, device, save_path, scheduler=None):
        """
        Args:
            model: The MultiTaskResNet model.
            criterion: Loss function (CrossEntropyLoss).
            optimizer: Optimizer (AdamW).
            device: Torch device (cpu or cuda).
            save_path: Path to save the best model checkpoint.
            scheduler: Learning rate scheduler (optional).
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.scheduler = scheduler
        self.best_f1 = -1.0
        # Initialize GradScaler for AMP (Cite solution_lesson_node_00002)
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            # Unpack batch: images, species, family, order
            images, species_targets, family_targets, order_targets = batch

            images = images.to(self.device)
            species_targets = species_targets.to(self.device)
            family_targets = family_targets.to(self.device)
            order_targets = order_targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP (Cite solution_lesson_node_00002)
            with torch.cuda.amp.autocast():
                species_out, family_out, order_out = self.model(images)

                # Calculate losses
                loss_species = self.criterion(species_out, species_targets)
                loss_family = self.criterion(family_out, family_targets)
                loss_order = self.criterion(order_out, order_targets)

                # Weighted sum of losses (1.0 for species, 0.5 for auxiliaries)
                total_loss = loss_species + 0.5 * loss_family + 0.5 * loss_order

            # Backward pass with scaler
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            running_loss += total_loss.item() * images.size(0)
            n_samples += images.size(0)

        return running_loss / n_samples if n_samples > 0 else 0.0

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        Returns:
            float: Macro F1 score on species predictions.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Val loader returns images, species, family, order
                images, species_targets, _, _ = batch
                images = images.to(self.device)

                species_out, _, _ = self.model(images)

                # We only care about species prediction for the metric
                preds = torch.argmax(species_out, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(species_targets.numpy())

        return calculate_f1_score(all_targets, all_preds)

    def fit(self, train_loader, val_loader, epochs, patience):
        """
        Runs the full training loop with early stopping.
        """
        # Ensure save directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_f1 = self.validate(val_loader)

            # Print metrics with full precision
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss} - Val F1: {val_f1}")

            # Early Stopping and Checkpointing
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(self.model.state_dict(), self.save_path)
                patience_counter = 0
                print(f"New best model saved with F1: {self.best_f1}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader, idx_to_species, output_file):
        """
        Generates predictions for the test set and saves to CSV.
        """
        # Load best model weights
        if os.path.exists(self.save_path):
            print(f"Loading best model from {self.save_path}...")
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)
            )
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                # Test loader returns images, image_ids
                images, image_ids = batch
                images = images.to(self.device)

                species_out, _, _ = self.model(images)
                preds = torch.argmax(species_out, dim=1).cpu().numpy()
                image_ids = image_ids.numpy()

                for img_id, pred_idx in zip(image_ids, preds):
                    # Map internal index back to original category_id
                    original_cat_id = idx_to_species[pred_idx]
                    submission_rows.append({"Id": img_id, "Predicted": original_cat_id})

        df_submission = pd.DataFrame(submission_rows)
        # Ensure sorting by Id
        df_submission = df_submission.sort_values("Id")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_submission.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")


def train_model(
    epochs=10,
    batch_size=128,
    learning_rate=1e-3,
    num_workers=4,
    device="cuda",
    seed=42,
    debug_limit=None,
    patience=3,
):
    """
    Orchestrates the data setup, model initialization, training, and prediction.
    """
    set_seed(seed)

    # 1. Get Taxonomy Mappings
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

    # 2. Define Transforms
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

    # 3. Load Metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    if debug_limit:
        print(f"Debug mode: limiting datasets to {debug_limit} samples.")
        df_train = df_train.head(debug_limit)
        df_val = df_val.head(debug_limit)
        df_test = df_test.head(debug_limit)

    taxonomy_maps = (species_to_idx, species_to_family, species_to_order)

    # 4. Create Datasets and Loaders
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

    # 5. Initialize Model
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet(num_species, num_families, num_orders).to(device_obj)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device_obj,
        save_path="./working/idea_1/best_model.pth",
    )

    # 7. Start Training
    print("Starting training...")
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)

    # 8. Generate Submission
    print("Generating submission...")
    trainer.predict(test_loader, idx_to_species, "./submission/submission.csv")
