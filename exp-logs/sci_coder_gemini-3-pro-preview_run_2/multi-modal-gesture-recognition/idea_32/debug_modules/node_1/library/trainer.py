import os
import torch
import torch.optim as optim
import numpy as np
import library.config as config
import library.utils as utils
from library.losses import CombinedSegmentationLoss


class Trainer:
    """
    Trainer class for BMGCN model.
    Manages training, validation, inference, and submission generation.
    """

    def __init__(self, model, device, train_config=config.TRAIN_CONFIG):
        self.model = model.to(device)
        self.device = device
        self.config = train_config

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Loss Function: CombinedSegmentationLoss (Weighted CE + BCE + Smoothness)
        self.criterion = CombinedSegmentationLoss(train_config).to(self.device)

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            features = batch["features"].to(self.device)
            cls_labels = batch["cls_labels"].to(self.device)
            bnd_labels = batch["bnd_labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, mask)

            targets = {"cls_labels": cls_labels, "bnd_labels": bnd_labels, "mask": mask}

            # Compute loss
            loss, _ = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            if self.config.get("gradient_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, dataloader):
        """
        Runs validation on the provided dataloader.
        Returns average loss and accuracy (on Stage 3 predictions).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                cls_labels = batch["cls_labels"].to(self.device)
                bnd_labels = batch["bnd_labels"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(features, mask)

                targets = {
                    "cls_labels": cls_labels,
                    "bnd_labels": bnd_labels,
                    "mask": mask,
                }

                loss, _ = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Calculate Accuracy on Stage 3 output
                # outputs['stage3']['cls_probs'] shape: (B, T, C)
                probs = outputs["stage3"]["cls_probs"]
                preds = torch.argmax(probs, dim=-1)  # (B, T)

                # Masked accuracy calculation
                matched = (preds == cls_labels) & mask
                correct += matched.sum().item()
                total_samples += mask.sum().item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy

    def fit(self, train_loader, val_loader, epochs, patience, checkpoint_path):
        """
        Executes the full training loop with early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            utils.print_metric("Epoch", epoch)
            utils.print_metric("Train Loss", train_loss)
            utils.print_metric("Val Loss", val_loss)
            utils.print_metric("Val Acc", val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                utils.save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def predict(self, dataloader):
        """
        Runs inference on the dataloader and generates formatted predictions.
        Applies median filtering and sequence decoding.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                sample_ids = batch["sample_ids"]
                lengths = batch["lengths"]

                outputs = self.model(features, mask)
                # Use Stage 3 probabilities for final prediction
                probs = outputs["stage3"]["cls_probs"].cpu().numpy()

                for i, sample_id in enumerate(sample_ids):
                    length = lengths[i]
                    sample_probs = probs[i, :length, :]  # (T, C)

                    # 1. Argmax to get labels
                    pred_labels = np.argmax(sample_probs, axis=-1)

                    # 2. Median Filter Smoothing
                    pred_labels = self._median_filter(pred_labels)

                    # 3. Decode (Remove background 0 and duplicates)
                    decoded_sequence = self._decode_sequence(pred_labels)

                    results.append((sample_id, decoded_sequence))
        return results

    def _median_filter(self, labels):
        """
        Applies a median filter to smooth the label sequence.
        """
        k = config.INFERENCE_CONFIG["median_window"]
        if k <= 1 or len(labels) < k:
            return labels

        # Edge padding to preserve boundary information
        pad_width = k // 2
        padded = np.pad(labels, pad_width, mode=config.INFERENCE_CONFIG["pad_mode"])

        # Rolling window median
        filtered = np.zeros_like(labels)
        for i in range(len(labels)):
            window = padded[i : i + k]
            filtered[i] = np.median(window)

        return filtered.astype(int)

    def _decode_sequence(self, labels):
        """
        Collapses repeated labels and removes background class (0).
        """
        unique_seq = []
        prev = -1
        for l in labels:
            if l != prev:
                if l != 0:  # 0 is background
                    unique_seq.append(l)
                prev = l
        return unique_seq

    def generate_submission_file(self, dataloader, output_path):
        """
        Generates predictions and saves them to a CSV file in the required format.
        Format: SessionID,Label1,Label2,...
        """
        results = self.predict(dataloader)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            for sample_id, sequence in results:
                # Convert list of ints to comma-separated string
                seq_str = ",".join(map(str, sequence))
                # Write line: SessionID,label1,label2...
                if seq_str:
                    line = f"{sample_id},{seq_str}\n"
                else:
                    line = f"{sample_id},\n"
                f.write(line)

        print(f"Submission saved to {output_path}")
