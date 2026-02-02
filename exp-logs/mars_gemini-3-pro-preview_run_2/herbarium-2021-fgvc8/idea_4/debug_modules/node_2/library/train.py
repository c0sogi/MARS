import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed) is handled in main scripts usually, but good practice
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Placeholders
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.num_families = 0
        self.num_orders = 0

    def initialize_model(self, num_families, num_orders, pretrained=True):
        """Initializes the HierarchicalConvNeXt model."""
        self.num_families = num_families
        self.num_orders = num_orders
        self.model = HierarchicalConvNeXt(
            num_families=num_families, num_orders=num_orders, pretrained=pretrained
        ).to(self.device)

    def save_checkpoint(self, path):
        """Saves the model state dict."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        """Loads the model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def train_epoch(self, loader, epoch_idx, total_epochs):
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (
            images,
            species_targets,
            family_targets,
            order_targets,
        ) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            species_targets = species_targets.to(self.device, non_blocking=True)
            family_targets = family_targets.to(self.device, non_blocking=True)
            order_targets = order_targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast():
                species_logits, family_logits, order_logits = self.model(images)

                loss_species = self.criterion(species_logits, species_targets)
                loss_family = self.criterion(family_logits, family_targets)
                loss_order = self.criterion(order_logits, order_targets)

                loss = loss_species + loss_family + loss_order

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch_idx+1}/{total_epochs}] Training Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self, loader):
        """
        Validates the model and returns the Macro F1 score for Species.
        Also calculates accuracy for monitoring.
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for images, species_targets, family_targets, order_targets in loader:
                images = images.to(self.device, non_blocking=True)
                species_targets = species_targets.to(self.device, non_blocking=True)
                family_targets = family_targets.to(self.device, non_blocking=True)
                order_targets = order_targets.to(self.device, non_blocking=True)

                with autocast():
                    species_logits, family_logits, order_logits = self.model(images)

                    loss_species = self.criterion(species_logits, species_targets)
                    loss_family = self.criterion(family_logits, family_targets)
                    loss_order = self.criterion(order_logits, order_targets)

                    loss = loss_species + loss_family + loss_order

                total_loss += loss.item()

                # Store predictions for F1 calculation
                preds = torch.argmax(species_logits, dim=1).cpu().numpy()
                targets = species_targets.cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_loss = total_loss / len(loader)

        # Calculate Macro F1
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        print(f"Validation Loss: {avg_loss:.6f}")
        print(f"Validation Species Macro F1: {macro_f1}")  # Full precision as requested

        return macro_f1

    def run_stage_1(self):
        """
        Stage 1: Representation Learning.
        Instance-balanced sampling, backbone unfrozen.
        """
        print("\n=== Starting Stage 1: Representation Learning ===")

        # 1. Get DataLoaders
        loaders, num_families, num_orders = get_dataloaders(stage=1)
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        # 2. Initialize Model
        self.initialize_model(num_families, num_orders, pretrained=Config.PRETRAINED)
        self.model.unfreeze_backbone()

        # 3. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.STAGE1_LR,
            weight_decay=Config.STAGE1_WEIGHT_DECAY,
        )

        total_steps = len(train_loader) * Config.STAGE1_EPOCHS
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.STAGE1_LR,
            total_steps=total_steps,
            pct_start=0.1,
        )

        # 4. Training Loop
        best_f1 = 0.0

        for epoch in range(Config.STAGE1_EPOCHS):
            self.train_epoch(train_loader, epoch, Config.STAGE1_EPOCHS)
            val_f1 = self.validate(val_loader)

            if val_f1 > best_f1:
                print(f"New best Stage 1 model found (F1: {val_f1}). Saving...")
                best_f1 = val_f1
                self.save_checkpoint(Config.MODEL_CHECKPOINT)

        print(f"Stage 1 completed. Best F1: {best_f1}")

    def run_stage_2(self):
        """
        Stage 2: Classifier Re-balancing.
        Class-balanced sampling, backbone frozen, fine-tune heads.
        """
        print("\n=== Starting Stage 2: Classifier Re-balancing ===")

        # 1. Get DataLoaders (Stage 2 uses WeightedRandomSampler)
        loaders, num_families, num_orders = get_dataloaders(stage=2)
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        # 2. Load Best Model from Stage 1
        if self.model is None:
            self.initialize_model(num_families, num_orders, pretrained=False)

        if os.path.exists(Config.MODEL_CHECKPOINT):
            print("Loading best model from Stage 1...")
            self.load_checkpoint(Config.MODEL_CHECKPOINT)
        else:
            print(
                "Warning: No checkpoint found from Stage 1. Starting from scratch (not recommended)."
            )

        # 3. Freeze Backbone
        self.model.freeze_backbone()
        print("Backbone frozen for Stage 2.")

        # 4. Optimizer (Re-init for heads only usually, but AdamW handles filtered params)
        # We filter parameters that require grad
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=Config.STAGE2_LR,
            weight_decay=Config.STAGE1_WEIGHT_DECAY,
        )

        # No OneCycleLR for fine-tuning, maybe simple StepLR or constant
        self.scheduler = None

        # 5. Training Loop
        # We track if we improve upon Stage 1's best score
        # But typically we save the best of Stage 2 regardless to ensure we use the re-balanced weights
        best_f1 = 0.0

        # Initial validation to set baseline
        print("Validating baseline before Stage 2 fine-tuning...")
        best_f1 = self.validate(val_loader)

        for epoch in range(Config.STAGE2_EPOCHS):
            self.train_epoch(train_loader, epoch, Config.STAGE2_EPOCHS)
            val_f1 = self.validate(val_loader)

            if val_f1 > best_f1:
                print(f"New best Stage 2 model found (F1: {val_f1}). Saving...")
                best_f1 = val_f1
                self.save_checkpoint(Config.MODEL_CHECKPOINT)
            else:
                print(f"Stage 2 F1 ({val_f1}) did not improve over best ({best_f1}).")

        print(f"Stage 2 completed. Best F1: {best_f1}")

    def predict_and_submit(self):
        """
        Generates predictions for the test set and saves submission file.
        """
        print("\n=== Generating Predictions ===")

        # 1. Load Data
        loaders, num_families, num_orders = get_dataloaders(
            stage=1
        )  # Stage arg doesn't affect test loader
        test_loader = loaders["test"]

        # 2. Load Best Model
        if self.model is None:
            self.initialize_model(num_families, num_orders, pretrained=False)

        if os.path.exists(Config.MODEL_CHECKPOINT):
            self.load_checkpoint(Config.MODEL_CHECKPOINT)
        else:
            print("Error: No checkpoint found for inference.")
            return

        self.model.eval()

        ids = []
        predictions = []

        print("Running inference on test set...")
        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(self.device, non_blocking=True)

                with autocast():
                    species_logits, _, _ = self.model(images)

                preds = torch.argmax(species_logits, dim=1).cpu().numpy()

                ids.extend(image_ids.numpy())
                predictions.extend(preds)

        # 3. Create Submission DataFrame
        df_sub = pd.DataFrame({"Id": ids, "Predicted": predictions})

        # Sort by Id just in case
        df_sub.sort_values(by="Id", inplace=True)

        # 4. Save
        save_path = Config.SUBMISSION_FILE
        df_sub.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(f"Total predictions: {len(df_sub)}")


def run_training():
    """
    Main entry point to execute the training pipeline.
    """
    set_seed(Config.SEED)
    Config.setup()

    trainer = Trainer()

    # Run Stage 1
    trainer.run_stage_1()

    # Run Stage 2
    trainer.run_stage_2()

    # Generate Submission
    trainer.predict_and_submit()
