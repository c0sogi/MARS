import os
import glob
import torch
import torch.nn as nn
import torchaudio
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
import pandas as pd
from library.config import TRAIN_CONFIG, PATH_CONFIG, LABEL_CONFIG, AUDIO_CONFIG
from library.model import DilatedEfficientNetB2
from library.transforms import LogMelSpectrogram, BackgroundNoiseAugment, SpecAugment


class Trainer:
    def __init__(self, train_loader, val_loader, test_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device(TRAIN_CONFIG.device)
        self.epochs = TRAIN_CONFIG.epochs
        self.swa_start_epoch = TRAIN_CONFIG.swa_start_epoch

        # Initialize Model
        self.model = DilatedEfficientNetB2(num_classes=LABEL_CONFIG.num_classes).to(
            self.device
        )

        # Initialize SWA Model
        self.swa_model = AveragedModel(self.model).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CONFIG.lr,
            weight_decay=TRAIN_CONFIG.weight_decay,
        )

        # Scheduler: Cosine Annealing for the convergence phase
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.swa_start_epoch, eta_min=TRAIN_CONFIG.min_lr
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # GPU Transforms
        self.log_mel = LogMelSpectrogram(AUDIO_CONFIG).to(self.device)

        # Load Background Noise for Augmentation (Cite solution_lesson_node_00055)
        noise_files = glob.glob(
            os.path.join(
                PATH_CONFIG.input_root, "train", "audio", "_background_noise_", "*.wav"
            )
        )
        noise_tensors = []
        for f in noise_files:
            try:
                w, sr = torchaudio.load(f)
                if sr != AUDIO_CONFIG.sample_rate:
                    w = torchaudio.transforms.Resample(sr, AUDIO_CONFIG.sample_rate)(w)
                if w.shape[0] > 1:
                    w = w.mean(dim=0, keepdim=True)
                noise_tensors.append(w.squeeze(0))
            except Exception:
                continue

        if noise_tensors:
            self.noise_bank = torch.cat(noise_tensors, dim=0).to(self.device)
        else:
            self.noise_bank = torch.zeros(16000).to(self.device)

        self.wave_aug = BackgroundNoiseAugment(self.noise_bank, p=0.5).to(self.device)
        self.spec_aug = SpecAugment(p=0.5).to(self.device)

        # Tracking
        self.best_acc = 0.0

        # Ensure checkpoint directory exists
        os.makedirs(TRAIN_CONFIG.checkpoint_dir, exist_ok=True)

    def mixup_data(self, x, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        self.wave_aug.train()
        self.spec_aug.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for waveforms, labels in self.train_loader:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            # 1. Waveform Augmentation
            waveforms = self.wave_aug(waveforms)

            # 2. Convert to Spectrogram
            specs = self.log_mel(waveforms)

            # 3. Spectrogram Augmentation
            specs = self.spec_aug(specs)

            # 4. Mixup
            mixed_specs, y_a, y_b, lam = self.mixup_data(
                specs, labels, TRAIN_CONFIG.mixup_alpha
            )

            # 5. Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(mixed_specs)

            # 6. Loss Calculation
            loss = self.mixup_criterion(self.criterion, outputs, y_a, y_b, lam)

            # 7. Backward Pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * waveforms.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            # Approximate accuracy for mixup (weighted sum of correct predictions)
            correct += (
                lam * predicted.eq(y_a).sum().float()
                + (1 - lam) * predicted.eq(y_b).sum().float()
            ).item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Scheduler Management
        current_lr = self.optimizer.param_groups[0]["lr"]

        if epoch_idx < self.swa_start_epoch:
            self.scheduler.step()
        else:
            # SWA Phase: Maintain constant low LR
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = TRAIN_CONFIG.swa_lr

        return epoch_loss, epoch_acc, current_lr

    def validate(self, loader, model_to_eval=None):
        if model_to_eval is None:
            model_to_eval = self.model

        model_to_eval.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for waveforms, labels in loader:
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)

                # Preprocessing (No Augmentation)
                specs = self.log_mel(waveforms)

                # Forward
                outputs = model_to_eval(specs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * waveforms.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / total, correct / total

    def fit(self):
        print(f"Starting training on {self.device} for {self.epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, lr = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(self.val_loader)

            print(
                f"Epoch {epoch}/{self.epochs} | LR: {lr} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Save Best Model (Standard)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), TRAIN_CONFIG.best_model_path)
                print(f"New best model saved with Val Acc: {val_acc}")

            # SWA Collection Phase
            if epoch > self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                print(f"SWA: Updated parameters at epoch {epoch}")

        print("Training finished. Updating SWA Batch Norm statistics...")

        # Update BN for SWA
        # We wrap the transform and model because update_bn expects the model to handle the input from the loader
        bn_model = nn.Sequential(self.log_mel, self.swa_model)
        bn_model.train()

        # Run one pass over train loader to calibrate BN stats
        update_bn(self.train_loader, bn_model, device=self.device)

        # Save SWA Model
        torch.save(self.swa_model.state_dict(), TRAIN_CONFIG.swa_model_path)
        print(f"SWA model saved to {TRAIN_CONFIG.swa_model_path}")

        # Validate SWA Model
        swa_val_loss, swa_val_acc = self.validate(
            self.val_loader, model_to_eval=self.swa_model
        )
        print(f"Final SWA Val Acc: {swa_val_acc}")

        # Generate Submission
        if self.test_loader:
            self.generate_submission()

    def generate_submission(self):
        print("Generating submission with SWA model...")
        self.swa_model.eval()
        all_preds = []

        # Inference
        with torch.no_grad():
            for waveforms, _ in self.test_loader:
                waveforms = waveforms.to(self.device)
                specs = self.log_mel(waveforms)
                outputs = self.swa_model(specs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted_ids = torch.max(probs, dim=1)
                all_preds.extend(predicted_ids.cpu().numpy())

        # Match predictions with filenames
        # Handle Subset vs Standard Dataset
        dataset = self.test_loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            indices = dataset.indices
            full_df = dataset.dataset.df
            sub_df = full_df.iloc[indices].reset_index(drop=True)
        else:
            sub_df = dataset.df

        submission_records = []
        for idx, row in sub_df.iterrows():
            fname = os.path.basename(row["filepath"])
            pred_id = all_preds[idx]

            # Map ID -> Fine Grained Label -> Submission Label
            fine_label = LABEL_CONFIG.id2label[pred_id]
            sub_label = LABEL_CONFIG.map_to_submission_label(fine_label)

            submission_records.append({"fname": fname, "label": sub_label})

        # Save CSV
        df_sub = pd.DataFrame(submission_records)
        os.makedirs(os.path.dirname(PATH_CONFIG.submission_path), exist_ok=True)
        df_sub.to_csv(PATH_CONFIG.submission_path, index=False)
        print(f"Submission saved to {PATH_CONFIG.submission_path}")
