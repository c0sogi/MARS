import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import from provided library files
from library.config import (
    PathConfig,
    AudioConfig,
    MelConfig,
    ModelConfig,
    TrainConfig,
    IDX_TO_LABEL,
)
from library.utils import (
    set_seed,
    MetricMonitor,
    save_checkpoint,
    load_checkpoint,
)
from library.transforms import (
    GPUNoiseInjector,
    MultiResMelSpectrogram,
    GPUSpecAugment,
)
from library.dataset import get_dataloaders
from library.model import ResNeStCRNN


class Trainer:
    """
    Manages the GPU-native training pipeline for the Multi-Resolution ResNeSt-CRNN.
    """

    def __init__(
        self,
        path_config: PathConfig,
        audio_config: AudioConfig,
        mel_config: MelConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ):
        self.path_config = path_config
        self.audio_config = audio_config
        self.mel_config = mel_config
        self.model_config = model_config
        self.train_config = train_config

        self.device = torch.device(train_config.device)

        # Ensure reproducibility
        set_seed(train_config.seed)

        # --- Initialize GPU-Native Transforms ---
        # 1. Raw Audio Augmentation
        self.noise_injector = GPUNoiseInjector(
            path_config, audio_config, train_config
        ).to(self.device)

        # 2. Feature Extraction
        self.mel_spectrogram = MultiResMelSpectrogram(mel_config, audio_config).to(
            self.device
        )

        # 3. Spectrogram Augmentation
        self.spec_augment = GPUSpecAugment(train_config).to(self.device)

        # --- Initialize Model ---
        self.model = ResNeStCRNN(model_config).to(self.device)

        # --- Optimization ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_config.num_epochs, eta_min=1e-6
        )

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        self.noise_injector.train()
        self.spec_augment.train()

        metric_monitor = MetricMonitor()

        for batch_idx, (waveforms, targets) in enumerate(train_loader):
            waveforms = waveforms.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # --- GPU Pipeline ---
            # 1. Dynamic Noise Injection (Raw Audio)
            aug_waveforms = self.noise_injector(waveforms)

            # 2. Multi-Resolution Feature Extraction
            # Result: (B, 3, F, T)
            specs = self.mel_spectrogram(aug_waveforms)

            # 3. SpecAugment (Spectrogram)
            aug_specs = self.spec_augment(specs)

            # 4. Model Forward Pass
            logits = self.model(aug_specs)

            loss = self.criterion(logits, targets)

            # --- Backprop ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- Metrics ---
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == targets).float().mean()

            metric_monitor.update("loss", loss.item(), waveforms.size(0))
            metric_monitor.update("acc", accuracy.item(), waveforms.size(0))

        return (
            metric_monitor.metrics["loss"]["avg"],
            metric_monitor.metrics["acc"]["avg"],
        )

    def validate(self, val_loader):
        """Runs validation loop."""
        self.model.eval()
        # Transforms in eval mode (no noise, no masking)
        self.noise_injector.eval()
        self.spec_augment.eval()

        metric_monitor = MetricMonitor()

        with torch.no_grad():
            for waveforms, targets in val_loader:
                waveforms = waveforms.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # --- GPU Pipeline (Inference Mode) ---
                # 1. No Noise Injection

                # 2. Feature Extraction
                specs = self.mel_spectrogram(waveforms)

                # 3. No SpecAugment

                # 4. Model Forward Pass
                logits = self.model(specs)

                loss = self.criterion(logits, targets)

                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == targets).float().mean()

                metric_monitor.update("loss", loss.item(), waveforms.size(0))
                metric_monitor.update("acc", accuracy.item(), waveforms.size(0))

        return (
            metric_monitor.metrics["loss"]["avg"],
            metric_monitor.metrics["acc"]["avg"],
        )

    def fit(self, load_cached_data=True):
        """
        Main training loop with Early Stopping.
        """
        # Load Data
        train_loader, val_loader, _ = get_dataloaders(
            self.path_config,
            self.audio_config,
            self.train_config,
            load_cached_data=load_cached_data,
        )

        best_acc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.path_config.working_dir, "best_model.pth")

        print(f"Starting training on {self.device}...")

        for epoch in range(1, self.train_config.num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch}/{self.train_config.num_epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
            )

            # Early Stopping & Checkpointing
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "best_acc": best_acc,
                    },
                    is_best=True,
                    checkpoint_dir=self.path_config.working_dir,
                )
                print(f"  -> New best model saved! (Acc: {best_acc:.6f})")
            else:
                patience_counter += 1
                print(
                    f"  -> Patience: {patience_counter}/{self.train_config.early_stopping_patience}"
                )

            if patience_counter >= self.train_config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set using the best model.
        """
        # Load Data
        _, _, test_loader = get_dataloaders(
            self.path_config,
            self.audio_config,
            self.train_config,
            load_cached_data=load_cached_data,
        )

        # Load Best Model
        best_model_path = os.path.join(self.path_config.working_dir, "best_model.pth")
        print(f"Loading best model from {best_model_path} for inference...")

        checkpoint = load_checkpoint(
            self.model, best_model_path, device=self.train_config.device
        )
        if checkpoint is None:
            print("Error: No checkpoint found. Cannot predict.")
            return

        self.model.eval()
        self.mel_spectrogram.eval()

        all_preds = []

        with torch.no_grad():
            for waveforms, _ in test_loader:
                waveforms = waveforms.to(self.device, non_blocking=True)

                # Inference Pipeline
                specs = self.mel_spectrogram(waveforms)
                logits = self.model(specs)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())

        # Map indices to labels
        pred_labels = [IDX_TO_LABEL[idx] for idx in all_preds]

        # Match with filenames
        # We read the test csv again to ensure alignment with the dataloader
        # (Dataloader preserves order of the CSV)
        df_test = pd.read_csv(self.path_config.test_csv)

        # The fname in submission should be the filename, not the full path
        # e.g., clip_000044442.wav
        fnames = df_test["filepath"].apply(lambda x: os.path.basename(x)).tolist()

        submission_df = pd.DataFrame({"fname": fnames, "label": pred_labels})

        # Save Submission
        print(f"Saving submission to {self.path_config.submission_path}...")
        submission_df.to_csv(self.path_config.submission_path, index=False)
        print("Submission saved successfully.")
