import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import library.config as config
import library.utils as utils

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class MultiTaskEncoder(nn.Module):
    """
    Stage 1: Bi-LSTM Encoder.
    Processes the input sequence and produces initial class and boundary predictions.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(MultiTaskEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output is 2 * hidden_dim due to bidirectional
        lstm_out_dim = hidden_dim * 2

        # Heads
        self.cls_head = nn.Linear(lstm_out_dim, num_classes)
        self.bnd_head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x, mask=None):
        # x: (B, T, InputDim)
        # mask: (B, T) - used for packing if desired, but simple masking output is sufficient here

        self.lstm.flatten_parameters()
        feat, _ = self.lstm(x)  # (B, T, 2*Hidden)

        cls_logits = self.cls_head(feat)  # (B, T, NumClasses)
        bnd_logits = self.bnd_head(feat)  # (B, T, 1)

        cls_probs = F.softmax(cls_logits, dim=-1)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_logits, cls_probs, bnd_logits, bnd_probs


class BoundaryModulatedBlock(nn.Module):
    """
    Refinement Block with Boundary Modulation.
    Gate = sigmoid(Wg * X + Wmod * B)
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(BoundaryModulatedBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv_filter = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.conv_gate = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.conv_mod = nn.Conv1d(
            1, channels, 1
        )  # Projects boundary prob to channel space
        self.conv_out = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, boundary):
        # x: (B, C, T)
        # boundary: (B, 1, T)

        f = torch.tanh(self.conv_filter(x))

        g_base = self.conv_gate(x)
        g_mod = self.conv_mod(boundary)
        g = torch.sigmoid(g_base + g_mod)

        z = f * g
        out = self.conv_out(z)
        out = self.dropout(out)

        return x + out


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: CNN-based refinement using boundary information.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers, num_classes, kernel_size, dropout
    ):
        super(RefinementStage, self).__init__()

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.layers.append(
                BoundaryModulatedBlock(hidden_dim, kernel_size, dilation, dropout)
            )

        self.cls_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x, boundary_signal):
        # x: (B, InputDim, T) - InputDim is usually NumClasses + 1
        # boundary_signal: (B, 1, T)

        feat = self.input_proj(x)

        for layer in self.layers:
            feat = layer(feat, boundary_signal)

        cls_logits = self.cls_head(feat)
        bnd_logits = self.bnd_head(feat)

        # Permute back to (B, T, C) for consistency
        cls_logits = cls_logits.transpose(1, 2)
        bnd_logits = bnd_logits.transpose(1, 2)

        cls_probs = F.softmax(cls_logits, dim=-1)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_logits, cls_probs, bnd_logits, bnd_probs


class BMGCN(nn.Module):
    """
    Boundary-Modulated Gated-Cascaded Network.
    Stage 1 (LSTM) -> Stage 2 (Refinement) -> Stage 3 (Sharpening)
    """

    def __init__(self):
        super(BMGCN, self).__init__()

        # Config alias
        c = config.MODEL_CONFIG

        # Stage 1
        self.stage1 = MultiTaskEncoder(
            input_dim=c["input_dim"],
            hidden_dim=c["hidden_dim"],
            num_layers=c["lstm_layers"],
            num_classes=c["num_classes"],
            dropout=c["dropout"],
        )

        # Stage 2
        # Input: ClsProbs (21) + BndProb (1) = 22
        refine_input_dim = c["num_classes"] + 1
        self.stage2 = RefinementStage(
            input_dim=refine_input_dim,
            hidden_dim=c["dilation_channels"],
            num_layers=c["refinement_layers"],
            num_classes=c["num_classes"],
            kernel_size=c["kernel_size"],
            dropout=c["dropout"],
        )

        # Stage 3
        self.stage3 = RefinementStage(
            input_dim=refine_input_dim,
            hidden_dim=c["dilation_channels"],
            num_layers=c["refinement_layers"],
            num_classes=c["num_classes"],
            kernel_size=c["kernel_size"],
            dropout=c["dropout"],
        )

    def forward(self, x, mask):
        # x: (B, T, InputDim)
        # mask: (B, T)

        outputs = {}

        # --- Stage 1 ---
        s1_cls_logits, s1_cls_probs, s1_bnd_logits, s1_bnd_probs = self.stage1(x)

        # Apply Mask
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        s1_cls_probs = s1_cls_probs * mask_expanded
        s1_bnd_probs = s1_bnd_probs * mask_expanded

        outputs["stage1"] = {
            "cls_logits": s1_cls_logits,
            "cls_probs": s1_cls_probs,
            "bnd_logits": s1_bnd_logits,
            "bnd_probs": s1_bnd_probs,
        }

        # --- Stage 2 ---
        # Prepare Input: Concat (B, T, C) -> (B, C, T)
        s2_in = torch.cat([s1_cls_probs, s1_bnd_probs], dim=2).transpose(1, 2)
        s2_bnd_sig = s1_bnd_probs.transpose(1, 2)  # (B, 1, T)

        s2_cls_logits, s2_cls_probs, s2_bnd_logits, s2_bnd_probs = self.stage2(
            s2_in, s2_bnd_sig
        )

        # Apply Mask
        s2_cls_probs = s2_cls_probs * mask_expanded
        s2_bnd_probs = s2_bnd_probs * mask_expanded

        outputs["stage2"] = {
            "cls_logits": s2_cls_logits,
            "cls_probs": s2_cls_probs,
            "bnd_logits": s2_bnd_logits,
            "bnd_probs": s2_bnd_probs,
        }

        # --- Stage 3 ---
        s3_in = torch.cat([s2_cls_probs, s2_bnd_probs], dim=2).transpose(1, 2)
        s3_bnd_sig = s2_bnd_probs.transpose(1, 2)

        s3_cls_logits, s3_cls_probs, s3_bnd_logits, s3_bnd_probs = self.stage3(
            s3_in, s3_bnd_sig
        )

        # Apply Mask
        s3_cls_probs = s3_cls_probs * mask_expanded
        s3_bnd_probs = s3_bnd_probs * mask_expanded

        outputs["stage3"] = {
            "cls_logits": s3_cls_logits,
            "cls_probs": s3_cls_probs,
            "bnd_logits": s3_bnd_logits,
            "bnd_probs": s3_bnd_probs,
        }

        return outputs


# -----------------------------------------------------------------------------
# Trainer & Inference Logic
# -----------------------------------------------------------------------------


class BMGCNTrainer:
    def __init__(self, model, device, train_config=config.TRAIN_CONFIG):
        self.model = model.to(device)
        self.device = device
        self.config = train_config

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Loss Functions
        class_weights = torch.tensor(self.config["class_weights"]).float().to(device)
        self.criterion_cls = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.criterion_bnd = nn.BCEWithLogitsLoss(reduction="none")

    def compute_loss(self, outputs, targets):
        # targets: {cls_labels: (B, T), bnd_labels: (B, T), mask: (B, T)}
        cls_target = targets["cls_labels"]
        bnd_target = targets["bnd_labels"].unsqueeze(-1)  # (B, T, 1)
        mask = targets["mask"]

        total_loss = 0.0
        stats = {}

        # Iterate over stages
        for stage_name in ["stage1", "stage2", "stage3"]:
            out = outputs[stage_name]

            # 1. Classification Loss (Weighted CE)
            # Flatten for CE: (B*T, C) vs (B*T)
            # Apply mask manually
            loss_cls_raw = self.criterion_cls(
                out["cls_logits"].view(-1, config.NUM_CLASSES), cls_target.view(-1)
            )
            loss_cls_raw = loss_cls_raw.view(mask.shape)
            loss_cls = (loss_cls_raw * mask).sum() / (mask.sum() + 1e-6)

            # 2. Boundary Loss (BCE)
            loss_bnd_raw = self.criterion_bnd(out["bnd_logits"], bnd_target)
            loss_bnd = (loss_bnd_raw.squeeze(-1) * mask).sum() / (mask.sum() + 1e-6)

            # 3. Smoothness Loss (T-MSE on Probs)
            # MSE(P_t, P_{t-1})
            probs = out["cls_probs"]  # (B, T, C)
            diff = probs[:, 1:, :] - probs[:, :-1, :]
            loss_smooth_raw = torch.mean(diff**2, dim=-1)  # (B, T-1)
            mask_smooth = mask[:, 1:] * mask[:, :-1]
            loss_smooth = (loss_smooth_raw * mask_smooth).sum() / (
                mask_smooth.sum() + 1e-6
            )

            # Weighted Sum
            stage_loss = (
                self.config["lambda_cls"] * loss_cls
                + self.config["lambda_bnd"] * loss_bnd
                + self.config["lambda_smooth"] * loss_smooth
            )

            total_loss += stage_loss
            stats[f"{stage_name}_loss"] = stage_loss.item()
            stats[f"{stage_name}_cls"] = loss_cls.item()

        return total_loss, stats

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            cls_labels = batch["cls_labels"].to(self.device)
            bnd_labels = batch["bnd_labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features, mask)

            targets = {"cls_labels": cls_labels, "bnd_labels": bnd_labels, "mask": mask}
            loss, _ = self.compute_loss(outputs, targets)

            loss.backward()

            if self.config["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )

            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

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
                loss, _ = self.compute_loss(outputs, targets)
                val_loss += loss.item()

                # Accuracy on Stage 3
                preds = torch.argmax(outputs["stage3"]["cls_probs"], dim=-1)
                correct += ((preds == cls_labels) * mask).sum().item()
                total += mask.sum().item()

        return val_loss / len(dataloader), correct / (total + 1e-6)

    def predict(self, dataloader):
        """
        Runs inference and generates formatted predictions.
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
                # Use Stage 3 probabilities
                probs = outputs["stage3"]["cls_probs"].cpu().numpy()

                for i, sample_id in enumerate(sample_ids):
                    length = lengths[i]
                    sample_probs = probs[i, :length, :]  # (T, C)

                    # 1. Argmax
                    pred_labels = np.argmax(sample_probs, axis=-1)

                    # 2. Median Filter Smoothing
                    pred_labels = self._median_filter(pred_labels)

                    # 3. Decode (Remove background 0 and duplicates)
                    decoded_sequence = self._decode_sequence(pred_labels)

                    results.append((sample_id, decoded_sequence))
        return results

    def _median_filter(self, labels):
        k = config.INFERENCE_CONFIG["median_window"]
        if k <= 1 or len(labels) < k:
            return labels

        # Edge padding
        pad_width = k // 2
        padded = np.pad(labels, pad_width, mode=config.INFERENCE_CONFIG["pad_mode"])

        # Rolling window median
        # Efficient stride trick or simple loop
        filtered = np.zeros_like(labels)
        for i in range(len(labels)):
            window = padded[i : i + k]
            filtered[i] = np.median(window)

        return filtered.astype(int)

    def _decode_sequence(self, labels):
        # Collapse repeats and remove background (0)
        unique_seq = []
        prev = -1
        for l in labels:
            if l != prev:
                if l != 0:  # 0 is background
                    unique_seq.append(l)
                prev = l
        return unique_seq

    def fit(self, train_loader, val_loader, epochs, patience, checkpoint_path):
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
