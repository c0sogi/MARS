import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import os

from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    INPUT_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BACKGROUND_CLASS_WEIGHT,
    SMOOTHING_THRESHOLD,
    SMOOTHING_LOSS_WEIGHT,
    NUM_CLASSES,
    SEED,
)
from library.dataset import GestureDataset
from library.model import KC_IRN
import library.inference as inference
import library.data_utils as data_utils

# Ensure deterministic behavior
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class CustomLoss(nn.Module):
    """
    Cascaded Loss Function:
    1. Weighted NLL Loss for each stage (Deep Supervision).
    2. Log-Space Smoothing Loss (Truncated MSE) for temporal consistency.
    """

    def __init__(self, device):
        super(CustomLoss, self).__init__()

        # Class Weights: Down-weight background (index 0)
        weights = torch.ones(NUM_CLASSES, device=device)
        weights[0] = BACKGROUND_CLASS_WEIGHT
        self.nll_loss = nn.NLLLoss(weight=weights)

        self.smoothing_threshold = SMOOTHING_THRESHOLD
        self.smoothing_weight = SMOOTHING_LOSS_WEIGHT

    def smoothing_loss(self, log_probs):
        """
        Calculates Truncated MSE between adjacent frames in log-space.
        Input: (Batch, Time, Classes)
        """
        # Diff between t and t-1
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared error
        sq_diff = diff**2

        # Truncate (clamp) the error
        # We clamp the squared error to threshold^2
        truncated_sq_diff = torch.clamp(sq_diff, max=self.smoothing_threshold**2)

        return torch.mean(truncated_sq_diff)

    def forward(self, outputs, targets):
        """
        outputs: tuple (out_1, out_2, out_3) each of shape (B, T, C)
        targets: (B, T)
        """
        total_loss = 0.0

        # Flatten targets for NLLLoss: (B*T)
        targets_flat = targets.view(-1)

        for stage_out in outputs:
            # 1. Classification Loss
            # Reshape (B, T, C) -> (B*T, C)
            stage_out_flat = stage_out.reshape(-1, NUM_CLASSES)
            cls_loss = self.nll_loss(stage_out_flat, targets_flat)

            # 2. Smoothing Loss
            smooth_loss = self.smoothing_loss(stage_out)

            # Sum
            total_loss += cls_loss + (self.smoothing_weight * smooth_loss)

        return total_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: returns tuple (out1, out2, out3)
        outputs = model(inputs)

        # Calculate loss (Deep Supervision handled inside criterion)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy on the final stage (out3)
        final_output = outputs[-1]  # (B, T, C)
        _, predicted = torch.max(final_output, 2)

        # Mask out padding if necessary?
        # Dataset produces fixed windows, so we evaluate on the whole window.
        # However, accuracy on background class dominates.
        # We calculate global accuracy here.
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.numel()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc


def validate_full_sequences(model, device):
    """
    Performs validation on full sequences and computes Levenshtein Error Rate.
    Cite Lesson 00075: Model Selection based on Sequence Metric.
    """
    model.eval()
    df_val = pd.read_csv(VAL_METADATA_PATH)

    total_distance = 0
    total_gestures = 0

    for _, row in df_val.iterrows():
        data_path = os.path.join(INPUT_DIR, row["data_path"])
        audio_path = os.path.join(INPUT_DIR, row["audio_path"])

        # Ground Truth
        gt_labels = []
        if pd.notna(row["labels"]):
            try:
                label_list = json.loads(row["labels"])
                label_list.sort(key=lambda x: x["begin"])
                gt_labels = [int(l["id"]) for l in label_list]
            except:
                pass

        # Inference
        skeleton = data_utils.load_robust_mat(data_path, load_cached_data=True)
        if skeleton is None or skeleton.shape[0] == 0:
            total_distance += len(gt_labels)
            total_gestures += len(gt_labels)
            continue

        num_frames = skeleton.shape[0]
        audio = data_utils.extract_audio_mfcc(
            audio_path, num_frames, load_cached_data=True
        )

        kinematics = data_utils.compute_kinematics(skeleton)
        T, J, D = kinematics.shape
        kinematics_flat = kinematics.reshape(T, J * D)

        features = np.concatenate([kinematics_flat, audio], axis=-1)

        avg_probs = inference.predict_sliding_window(model, features, device)
        frame_preds = np.argmax(avg_probs, axis=1)
        predicted_gestures = inference.post_process_predictions(frame_preds)

        dist = inference.calculate_levenshtein(predicted_gestures, gt_labels)
        total_distance += dist
        total_gestures += len(gt_labels)

    error_rate = total_distance / total_gestures if total_gestures > 0 else 1.0
    return error_rate


def train_model(limit_data=None, load_cached_data=True):
    """
    Main training function.
    Args:
        limit_data (int): Optional limit for debugging.
        load_cached_data (bool): Whether to use cached features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Datasets
    print("Loading Training Data...")
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH,
        is_train=True,
        load_cached_data=load_cached_data,
        limit=limit_data,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = KC_IRN().to(device)

    # 3. Setup Optimizer and Loss
    # Cite Lesson 00055: Use Weight Decay
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(device)

    # 4. Training Loop
    best_val_metric = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation on full sequences (Cite Lesson 00075)
        val_metric = validate_full_sequences(model, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Error Rate: {val_metric:.4f}"
        )

        # Early Stopping based on Levenshtein Error Rate
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  Validation metric improved. Saved model.")
        else:
            patience_counter += 1
            print(
                f"  EarlyStopping counter: {patience_counter} out of {EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Error Rate: {best_val_metric}")

    return best_model_path
