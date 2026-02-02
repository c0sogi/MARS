import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config, utils, model, dataset


class Trainer:
    """
    Handles the training, validation, and optimization of the HierarchicalResNet model.
    """

    def __init__(self, net, train_loader, val_loader, device=None):
        self.model = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device(config.DEVICE)

        # Move model to device
        self.model.to(self.device)

        # Loss function
        # Both heads use CrossEntropyLoss.
        # ArcFace output is scaled logits, suitable for CE Loss.
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS
        )

    def train_one_epoch(self):
        self.model.train()
        losses = utils.AverageMeter()

        for i, (images, species_labels, genus_labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            species_labels = species_labels.to(self.device)
            genus_labels = genus_labels.to(self.device)

            # Forward pass
            # Pass species_labels to enable ArcFace margin calculation
            species_logits, genus_logits = self.model(
                images, species_label=species_labels
            )

            # Calculate losses
            loss_species = self.criterion(species_logits, species_labels)
            loss_genus = self.criterion(genus_logits, genus_labels)

            # Multi-task loss
            loss = loss_species + (config.LAMBDA_GENUS * loss_genus)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, species_labels, _ in self.val_loader:
                images = images.to(self.device)

                # Inference: label=None returns cosine similarity scores
                species_logits, _ = self.model(images, species_label=None)

                # Prediction is the class with max similarity
                preds = torch.argmax(species_logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(species_labels.numpy())

        # Calculate Macro F1
        f1 = utils.calculate_metrics(all_targets, all_preds)
        return f1

    def fit(self, num_epochs=config.NUM_EPOCHS, patience=5):
        best_f1 = -1.0
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_f1 = self.validate()

            # Update scheduler
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Validation F1: {val_f1}")

            # Checkpoint and Early Stopping
            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            checkpoint_path = os.path.join(
                config.WORKING_DIR, f"checkpoint_epoch_{epoch+1}.pth"
            )
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_f1": best_f1,
                    "optimizer": self.optimizer.state_dict(),
                },
                is_best,
                filename=checkpoint_path,
                best_filename=config.MODEL_SAVE_PATH,
            )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation F1: {best_f1}")
        return best_f1


def generate_submission(net, test_loader, device=None):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    if device is None:
        device = torch.device(config.DEVICE)

    net.to(device)
    net.eval()

    ids = []
    predictions = []

    print("Generating submission...")
    with torch.no_grad():
        for images, image_ids, _ in test_loader:
            images = images.to(device)

            # Inference
            species_logits, _ = net(images, species_label=None)
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()

            ids.extend(image_ids.numpy())
            predictions.extend(preds)

    # Create DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def train():
    """
    Main entry point to setup data, model, and run training.
    """
    # Set seeds
    utils.seed_everything(config.SEED)

    # 1. Load Mappings to determine class counts
    # This also ensures the cache is built
    _, num_species, num_genus = config.get_mappings(load_cached=True)

    print(f"Initializing model with {num_species} species and {num_genus} genera.")

    # 2. Data Loaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    net = model.HierarchicalResNet(
        num_species=num_species,
        num_genus=num_genus,
        backbone_name=config.BACKBONE,
        pretrained=True,
    )

    # 4. Trainer Initialization
    trainer = Trainer(net, train_loader, val_loader)

    # 5. Run Training
    trainer.fit(num_epochs=config.NUM_EPOCHS)

    # 6. Load Best Model for Submission
    print(f"Loading best model from {config.MODEL_SAVE_PATH}...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    net.load_state_dict(checkpoint["state_dict"])

    # 7. Generate Submission
    generate_submission(net, test_loader)
