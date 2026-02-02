import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from library.config import (
    INPUT_DIM,
    NUM_CLASSES,
    HIDDEN_DIM,
    DROPOUT_ENCODER,
    DROPOUT_TCN,
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DILATIONS,
    BACKGROUND_CLASS_ID,
    BACKGROUND_WEIGHT,
    SMOOTHING_LOSS_WEIGHT,
    SMOOTHING_THRESHOLD,
    WINDOW_SIZE,
    INFERENCE_STRIDE,
    MIN_DURATION,
    SUBMISSION_DIR,
    WORKING_DIR,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    SEED,
)
from library.utils import (
    log_space_smoothing_loss,
    run_length_encoding,
    compute_levenshtein_score,
)
from library.data_loader import (
    get_dataloaders,
    get_test_loader,
    KinematicAugmentor,
    AudioProcessor,
)

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ==========================================
# 1. Model Components
# ==========================================


class AffineScalingLayer(nn.Module):
    """
    Learnable Affine Scaling: Y = X * W + B
    Aligns feature magnitudes without enforcing unit variance.
    """

    def __init__(self, num_features):
        super(AffineScalingLayer, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, num_features))
        self.shift = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        # x: (Batch, Time, Features)
        return x * self.scale + self.shift


class HighCapacityEncoder(nn.Module):
    """
    Stage 1: Bi-GRU Encoder with Affine Scaling and Dropout.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(HighCapacityEncoder, self).__init__()
        self.scaler = AffineScalingLayer(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.scaler(x)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class TCNBlock(nn.Module):
    """
    Dilated Convolutional Block with Gated Activation and Residual Connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()

        # Centered padding: (k-1)*d // 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels * 2,  # For Gating (Tanh + Sigmoid)
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

        # Residual connection handling
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        # x: (Batch, Channels, Time)
        res = x if self.downsample is None else self.downsample(x)

        out = self.conv(x)
        # Gated Activation: Split channels
        out1, out2 = out.chunk(2, dim=1)
        out = torch.tanh(out1) * torch.sigmoid(out2)

        out = self.dropout(out)
        return out + res


class TCNStage(nn.Module):
    """
    Refinement Stage: Stack of Dilated TCN Blocks.
    Input: Class Probabilities from previous stage.
    """

    def __init__(self, num_classes, internal_channels, kernel_size, dilations, dropout):
        super(TCNStage, self).__init__()

        layers = []
        # Input Projection
        layers.append(nn.Conv1d(num_classes, internal_channels, 1))

        # Stacked Dilated Blocks
        for d in dilations:
            layers.append(
                TCNBlock(internal_channels, internal_channels, kernel_size, d, dropout)
            )

        # Output Projection
        layers.append(nn.Conv1d(internal_channels, num_classes, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) -> Permute to (Batch, NumClasses, Time) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.net(x)
        # Permute back to (Batch, Time, NumClasses)
        return out.permute(0, 2, 1)


class ASH_KN(nn.Module):
    """
    Adaptive-Scale High-Capacity Kinematic Network.
    Three-Stage Cascade: Encoder -> Refinement 1 -> Refinement 2.
    """

    def __init__(self):
        super(ASH_KN, self).__init__()

        # Stage 1: Encoder
        self.stage1 = HighCapacityEncoder(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT_ENCODER,
        )

        # Stage 2: Monotonic Non-Causal Refinement
        self.stage2 = TCNStage(
            num_classes=NUM_CLASSES,
            internal_channels=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dilations=TCN_DILATIONS,
            dropout=DROPOUT_TCN,
        )

        # Stage 3: Independent Iterative Refinement
        self.stage3 = TCNStage(
            num_classes=NUM_CLASSES,
            internal_channels=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dilations=TCN_DILATIONS,
            dropout=DROPOUT_TCN,
        )

    def forward(self, x):
        # Stage 1
        logits1 = self.stage1(x)
        probs1 = torch.softmax(logits1, dim=2)

        # Stage 2 (Input: Probs from Stage 1)
        logits2 = self.stage2(probs1)
        probs2 = torch.softmax(logits2, dim=2)

        # Stage 3 (Input: Probs from Stage 2)
        logits3 = self.stage3(probs2)

        return logits1, logits2, logits3


# ==========================================
# 2. Training Logic
# ==========================================


def calculate_loss(logits1, logits2, logits3, targets, device):
    # Class Weights: Down-weight background
    weights = torch.ones(NUM_CLASSES).to(device)
    weights[BACKGROUND_CLASS_ID] = BACKGROUND_WEIGHT

    ce_loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Flatten for CE Loss: (Batch * Time, Classes) vs (Batch * Time)
    loss1 = ce_loss_fn(logits1.reshape(-1, NUM_CLASSES), targets.reshape(-1))
    loss2 = ce_loss_fn(logits2.reshape(-1, NUM_CLASSES), targets.reshape(-1))
    loss3 = ce_loss_fn(logits3.reshape(-1, NUM_CLASSES), targets.reshape(-1))

    # Smoothing Loss (Log-Space) for refinement stages
    log_probs2 = F.log_softmax(logits2, dim=2)
    log_probs3 = F.log_softmax(logits3, dim=2)

    smooth2 = log_space_smoothing_loss(log_probs2, threshold=SMOOTHING_THRESHOLD)
    smooth3 = log_space_smoothing_loss(log_probs3, threshold=SMOOTHING_THRESHOLD)

    total_loss = (loss1 + loss2 + loss3) + SMOOTHING_LOSS_WEIGHT * (smooth2 + smooth3)

    return total_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        l1, l2, l3 = model(inputs)
        loss = calculate_loss(l1, l2, l3, targets, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            l1, l2, l3 = model(inputs)
            loss = calculate_loss(l1, l2, l3, targets, device)
            total_loss += loss.item()

    return total_loss / len(loader)


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Initialize Model
    model = ASH_KN().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
    return best_model_path


# ==========================================
# 3. Inference Logic
# ==========================================


def predict_sequence(model, skeleton, audio, device):
    """
    Runs sliding window inference on a single sequence.
    """
    model.eval()
    num_frames = skeleton.shape[0]

    # Prepare Augmentor (No augmentation for test)
    augmentor = KinematicAugmentor(augment=False)

    # Pre-calculate features for the whole sequence
    # Note: We process the whole sequence at once to get features,
    # but run model on windows to match training distribution.

    # However, KinematicAugmentor expects (T, J, 3)
    # We can process the whole sequence to get features (T, InputDim)
    kinematic_feats = augmentor(skeleton)  # (T, J*9)
    # Audio is already (T, MFCC)

    # Concatenate
    full_features = np.concatenate([kinematic_feats, audio], axis=1)  # (T, InputDim)

    # Prepare for accumulation
    # Shape: (T, NumClasses)
    accumulated_probs = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    count_matrix = np.zeros((num_frames, 1), dtype=np.float32)

    # Sliding Window
    step = INFERENCE_STRIDE
    win_size = WINDOW_SIZE

    windows = []
    indices = []

    # If sequence is shorter than window, pad it
    if num_frames < win_size:
        pad_len = win_size - num_frames
        padded_feats = np.pad(full_features, ((0, pad_len), (0, 0)), mode="constant")
        windows.append(padded_feats)
        indices.append((0, num_frames))  # Valid range
    else:
        for start in range(0, num_frames - win_size + 1, step):
            end = start + win_size
            windows.append(full_features[start:end])
            indices.append((start, end))

        # Handle last frame
        if (num_frames - win_size) % step != 0:
            start = num_frames - win_size
            end = num_frames
            windows.append(full_features[start:end])
            indices.append((start, end))

    # Batch processing
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_wins = windows[i : i + batch_size]
            batch_idxs = indices[i : i + batch_size]

            # To Tensor
            input_tensor = torch.FloatTensor(np.array(batch_wins)).to(device)

            # Forward
            _, _, logits3 = model(input_tensor)
            probs3 = torch.softmax(logits3, dim=2).cpu().numpy()

            # Accumulate
            for j, (start, end) in enumerate(batch_idxs):
                # If padded (short seq), only take valid part
                valid_len = end - start
                # For short seq, end is num_frames, start is 0.
                # But window is padded to win_size.
                if num_frames < win_size:
                    accumulated_probs[0:num_frames] += probs3[j, 0:num_frames]
                    count_matrix[0:num_frames] += 1
                else:
                    accumulated_probs[start:end] += probs3[j]
                    count_matrix[start:end] += 1

    # Average
    avg_probs = accumulated_probs / np.maximum(count_matrix, 1.0)

    # Argmax
    predictions = np.argmax(avg_probs, axis=1)

    return predictions


def generate_submission(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = ASH_KN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get Test Data (Raw dict)
    _, test_data = get_test_loader(batch_size=1)  # Loader not used, just data dict

    results = []

    print("Generating predictions...")
    for sid in tqdm(sorted(test_data.keys())):
        sample = test_data[sid]
        skel = sample["skeleton"]
        audio = sample["audio"]

        if skel is None or len(skel) == 0:
            # Fallback for empty data
            pred_seq = []
        else:
            # Predict frame-wise
            frame_preds = predict_sequence(model, skel, audio, device)

            # Decode (RLE)
            pred_seq = run_length_encoding(frame_preds, min_duration=MIN_DURATION)

        # Format string: "2,12,3"
        pred_str = ",".join(map(str, pred_seq))
        results.append(f"{sid},{pred_str}")

    # Save
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(sub_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {sub_path}")


def run_pipeline():
    # 1. Train
    best_model_path = run_training()

    # 2. Generate Submission
    generate_submission(best_model_path)


if __name__ == "__main__":
    # This block is strictly forbidden by instructions,
    # but the functions above are available for import.
    pass
