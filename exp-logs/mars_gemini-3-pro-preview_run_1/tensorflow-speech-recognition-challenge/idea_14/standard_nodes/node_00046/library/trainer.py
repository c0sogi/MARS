import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, DistillationLoss
from library.model import DilatedEfficientNet


class Trainer:
    """
    Manages the training lifecycle for the Self-Distillation (Born-Again Networks) strategy.
    Handles Teacher training, Student distillation, validation, and submission generation.
    """

    def __init__(
        self, train_loader, val_loader, test_loader, label_encoder, config=Config
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.le = label_encoder
        self.config = config
        self.device = config.device

        # Ensure working directory exists for checkpoints
        os.makedirs(self.config.working_dir, exist_ok=True)

        # Loss functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_distill = DistillationLoss(
            distillation_weight=config.distillation_lambda
        )

    def _calculate_accuracy(self, outputs, targets):
        """Helper to calculate simple accuracy."""
        _, preds = torch.max(outputs, 1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)

    def _calculate_mixup_accuracy(self, outputs, y_a, y_b, lam):
        """Helper to calculate weighted accuracy for Mixup."""
        _, preds = torch.max(outputs, 1)
        correct_a = (preds == y_a).sum().item()
        correct_b = (preds == y_b).sum().item()
        return lam * correct_a + (1 - lam) * correct_b

    def train_teacher_epoch(self, model, optimizer):
        """
        Trains the Teacher model for one epoch using Cross-Entropy and Mixup.
        """
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply Mixup
            inputs, y_a, y_b, lam = mixup_data(
                inputs, targets, self.config.mixup_alpha, self.device
            )

            optimizer.zero_grad()
            outputs = model(inputs)

            # Standard Mixup Loss
            loss = mixup_criterion(self.criterion_ce, outputs, y_a, y_b, lam)

            loss.backward()
            optimizer.step()

            # Track metrics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += self._calculate_mixup_accuracy(outputs, y_a, y_b, lam)
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def train_student_epoch(self, student, teacher, optimizer):
        """
        Trains the Student model for one epoch using Distillation Loss.
        The Teacher is frozen and provides soft targets.
        """
        student.train()
        teacher.eval()

        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply Mixup to inputs
            # We use the same mixed inputs for both Student and Teacher to align feature space
            inputs_mixed, y_a, y_b, lam = mixup_data(
                inputs, targets, self.config.mixup_alpha, self.device
            )

            # Get Teacher Logits (No Gradient)
            with torch.no_grad():
                teacher_logits = teacher(inputs_mixed)

            optimizer.zero_grad()
            student_logits = student(inputs_mixed)

            # Calculate Distillation Loss (Weighted CE + KL)
            loss = self.criterion_distill(
                student_logits, teacher_logits, targets, mixup_args=(y_a, y_b, lam)
            )

            loss.backward()
            optimizer.step()

            # Track metrics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += self._calculate_mixup_accuracy(
                student_logits, y_a, y_b, lam
            )
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def validate(self, model):
        """
        Evaluates the model on the validation set.
        """
        model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = self.criterion_ce(outputs, targets)

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += (
                    self._calculate_accuracy(outputs, targets) * batch_size
                )
                total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def generate_submission(self, model_path):
        """
        Generates predictions for the test set and saves the submission CSV.
        Handles mapping from fine-grained classes to the 12 competition labels.
        """
        print(f"Generating submission using model at {model_path}...")

        # Load Model
        model = DilatedEfficientNet(config=self.config)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        predictions = []
        filenames = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)

                # Retrieve filenames for the current batch
                start_idx = i * self.config.batch_size
                end_idx = min(
                    (i + 1) * self.config.batch_size, len(self.test_loader.dataset)
                )

                # Access the dataframe in the dataset to get filenames
                batch_fnames = (
                    self.test_loader.dataset.df.iloc[start_idx:end_idx]["filepath"]
                    .apply(os.path.basename)
                    .tolist()
                )

                # Decode integer labels back to string labels (e.g., 'bed', 'yes', 'silence')
                decoded_labels = self.le.inverse_transform(preds.cpu().numpy())

                predictions.extend(decoded_labels)
                filenames.extend(batch_fnames)

        # Post-processing: Map fine-grained classes to competition targets
        final_labels = []
        target_set = set(self.config.target_labels)

        for label in predictions:
            if label in target_set:
                final_labels.append(label)
            elif label == self.config.silence_label:
                final_labels.append(label)
            else:
                # Map all auxiliary classes (bed, bird, etc.) to 'unknown'
                final_labels.append(self.config.unknown_label)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"fname": filenames, "label": final_labels})

        # Save to disk
        os.makedirs(self.config.submission_dir, exist_ok=True)
        df_sub.to_csv(self.config.submission_path, index=False)
        print(f"Submission saved to {self.config.submission_path}")

    def run(self):
        """
        Executes the full training pipeline:
        1. Train Teacher -> Save Best
        2. Train Student (Distilled from Teacher) -> Save Best
        3. Generate Submission using Best Student
        """
        set_seed(self.config.seed)

        # ==========================================
        # Stage 1: Train Teacher
        # ==========================================
        print("Starting Stage 1: Teacher Training")
        teacher_model = DilatedEfficientNet(config=self.config).to(self.device)
        optimizer = optim.AdamW(
            teacher_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs_teacher, eta_min=self.config.min_lr
        )

        best_teacher_acc = 0.0
        best_teacher_path = os.path.join(self.config.checkpoint_dir, "teacher_best.pth")

        for epoch in range(self.config.epochs_teacher):
            train_loss, train_acc = self.train_teacher_epoch(teacher_model, optimizer)
            val_loss, val_acc = self.validate(teacher_model)
            scheduler.step()

            print(
                f"Teacher Epoch {epoch+1}/{self.config.epochs_teacher} - "
                f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
                f"Val Loss: {val_loss}, Val Acc: {val_acc}"
            )

            if val_acc > best_teacher_acc:
                best_teacher_acc = val_acc
                torch.save(teacher_model.state_dict(), best_teacher_path)

        print(f"Stage 1 Complete. Best Teacher Acc: {best_teacher_acc}")

        # ==========================================
        # Stage 2: Train Student (Distillation)
        # ==========================================
        print("Starting Stage 2: Student Distillation")

        # Load Best Teacher
        teacher_model.load_state_dict(
            torch.load(best_teacher_path, map_location=self.device)
        )
        teacher_model.eval()
        # Freeze Teacher parameters to save memory/compute
        for param in teacher_model.parameters():
            param.requires_grad = False

        # Initialize Fresh Student
        student_model = DilatedEfficientNet(config=self.config).to(self.device)
        optimizer = optim.AdamW(
            student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs_student, eta_min=self.config.min_lr
        )

        best_student_acc = 0.0
        best_student_path = os.path.join(self.config.checkpoint_dir, "student_best.pth")

        for epoch in range(self.config.epochs_student):
            train_loss, train_acc = self.train_student_epoch(
                student_model, teacher_model, optimizer
            )
            val_loss, val_acc = self.validate(student_model)
            scheduler.step()

            print(
                f"Student Epoch {epoch+1}/{self.config.epochs_student} - "
                f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
                f"Val Loss: {val_loss}, Val Acc: {val_acc}"
            )

            if val_acc > best_student_acc:
                best_student_acc = val_acc
                torch.save(student_model.state_dict(), best_student_path)

        print(f"Stage 2 Complete. Best Student Acc: {best_student_acc}")

        # ==========================================
        # Generate Submission
        # ==========================================
        self.generate_submission(best_student_path)
