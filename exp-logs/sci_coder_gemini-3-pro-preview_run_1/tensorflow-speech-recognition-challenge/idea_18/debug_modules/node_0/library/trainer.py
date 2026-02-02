import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.optim.swa_utils import AveragedModel

from library.config import Config
from library.utils import (
    set_seed,
    update_bn,
    compute_accuracy,
    map_to_competition_label,
)
from library.dataset import get_dataloaders
from library.model import DilatedEfficientNet


class Trainer:
    """
    Trainer class for the Speech Command Recognition task.
    Manages the training loop, SWA schedule, validation, and submission generation.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(Config.SEED)

        # Initialize Model
        # We train on Fine-Grained classes (31 classes)
        self.model = DilatedEfficientNet(num_classes=Config.NUM_CLASSES)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LR,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        # Phase 1 lasts for 35 epochs (Config.SWA_START_EPOCH - 1).
        # We set T_max to 35 so the LR decays to MIN_LR exactly at the end of epoch 35.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.SWA_START_EPOCH - 1,
            eta_min=Config.MIN_LR,
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # SWA Model
        # Wraps the model to compute running averages of parameters
        self.swa_model = AveragedModel(self.model)
        self.swa_start_epoch = Config.SWA_START_EPOCH

    def train_one_epoch(self, loader):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size

            # Calculate accuracy on fine-grained labels
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == targets).sum().item()
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        return epoch_loss, epoch_acc

    def validate(self, loader, model_to_validate=None):
        """Runs validation on the given loader."""
        if model_to_validate is None:
            model_to_validate = self.model

        model_to_validate.eval()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model_to_validate(inputs)
                loss = self.criterion(outputs, targets)

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size

                _, predicted = torch.max(outputs, 1)
                running_correct += (predicted == targets).sum().item()
                total_samples += batch_size

        val_loss = running_loss / total_samples
        val_acc = running_correct / total_samples
        return val_loss, val_acc

    def train(self, load_cached_data=True):
        """
        Main training loop implementing the Seamless SWA strategy.
        """
        print(f"Starting training on device: {self.device}")

        # Load Data
        train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
            load_cached_data=load_cached_data
        )

        best_acc = 0.0
        best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
        swa_model_path = os.path.join(Config.WORK_DIR, "swa_model.pth")

        start_time = time.time()

        for epoch in range(1, Config.EPOCHS + 1):
            epoch_start = time.time()

            # 1. Train Step
            train_loss, train_acc = self.train_one_epoch(train_loader)

            # 2. Validation Step (Standard Model)
            val_loss, val_acc = self.validate(val_loader)

            # 3. SWA Update (Phase 2)
            # If we are in the SWA phase, update the averaged model
            if epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                is_swa_phase = True
            else:
                is_swa_phase = False

            # 4. Scheduler Step
            # We only step the scheduler during Phase 1.
            # Once Phase 1 ends, the LR stays at MIN_LR (the floor) for the SWA phase.
            current_lr = self.optimizer.param_groups[0]["lr"]
            if epoch < self.swa_start_epoch:
                self.scheduler.step()

            # 5. Checkpoint Best Model
            # We track the best standard model for safety/comparison
            saved_msg = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                saved_msg = "[Saved Best]"

            epoch_duration = time.time() - epoch_start
            phase_str = "SWA" if is_swa_phase else "Std"

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{Config.EPOCHS} [{phase_str}] | "
                f"Time: {epoch_duration:.1f}s | "
                f"LR: {current_lr:.8f} | "
                f"Train Loss: {train_loss:.8f} | Train Acc: {train_acc:.8f} | "
                f"Val Loss: {val_loss:.8f} | Val Acc: {val_acc:.8f} {saved_msg}"
            )

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.2f} minutes.")

        # 6. Finalize SWA Model
        print("Finalizing SWA model (updating BN statistics)...")
        # Update BN statistics using the training data
        update_bn(train_loader, self.swa_model, self.device)

        # Save SWA Model
        torch.save(self.swa_model.state_dict(), swa_model_path)
        print(f"SWA model saved to {swa_model_path}")

        # Validate SWA Model
        swa_val_loss, swa_val_acc = self.validate(
            val_loader, model_to_validate=self.swa_model
        )
        print(
            f"SWA Model Validation | Loss: {swa_val_loss:.8f} | Acc: {swa_val_acc:.8f}"
        )

        # 7. Generate Submission
        self.generate_submission(test_loader, class_to_idx)

    def generate_submission(self, test_loader, class_to_idx):
        """
        Generates predictions for the test set using the SWA model,
        maps them to the competition labels, and saves the submission file.
        """
        print("Generating submission using SWA model...")

        # Invert class mapping: Index -> Fine-Grained Label
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        self.swa_model.eval()

        # Get filenames from the dataset dataframe (order is preserved in loader)
        test_df = test_loader.dataset.df

        all_logits = []

        # Inference Loop
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.swa_model(inputs)
                all_logits.append(outputs.cpu())

        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)

        # Get predictions (indices)
        _, pred_indices = torch.max(all_logits, 1)

        submission_data = []

        for i, idx in enumerate(pred_indices):
            idx = idx.item()
            fine_label = idx_to_class[idx]

            # Map Fine-Grained Label -> Competition Label (12 classes)
            comp_label = map_to_competition_label(fine_label)

            # Get filename
            full_rel_path = test_df.iloc[i]["filepath"]
            fname = os.path.basename(full_rel_path)

            submission_data.append({"fname": fname, "label": comp_label})

        # Create DataFrame and Save
        sub_df = pd.DataFrame(submission_data)
        submission_path = os.path.join(Config.WORK_DIR, "submission.csv")
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
