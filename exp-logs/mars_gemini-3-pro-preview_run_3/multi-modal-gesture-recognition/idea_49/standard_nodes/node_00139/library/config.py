import os
import sys
import json
import random
import glob
import math
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import sklearn.metrics

# ==========================================
# Configuration & Constants
# ==========================================


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_49"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Data Parameters
    WINDOW_SIZE = 64
    STRIDE = 32
    NUM_CLASSES = 21  # 20 gestures + 1 background (0)

    # Audio
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13

    # Model Hyperparameters
    HIDDEN_DIM = 96
    DROPOUT = 0.4

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Loss
    BACKGROUND_WEIGHT = 0.2
    MSE_LAMBDA = 0.15

    # Post-processing
    MIN_GESTURE_LENGTH = 5

    # Reproducibility
    SEED = 42


# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)

# ==========================================
# Data Processing Utilities
# ==========================================


def robust_load_mat(path):
    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return mat
    except Exception:
        return None


def get_skeleton_data(mat, num_frames):
    # Initialize with zeros: [Frames, Joints, Coords]
    skeleton_data = np.zeros((num_frames, 20, 3), dtype=np.float32)

    if mat is None:
        return skeleton_data

    if "Video" not in mat:
        return skeleton_data

    video = mat["Video"]
    if isinstance(video, np.ndarray) and video.ndim == 0:
        video = video.item()

    if not hasattr(video, "Frames"):
        return skeleton_data

    frames = video.Frames
    if not isinstance(frames, (np.ndarray, list)):
        frames = [frames]

    for i, frame in enumerate(frames):
        if i >= num_frames:
            break
        if hasattr(frame, "Skeleton") and hasattr(frame.Skeleton, "WorldPosition"):
            wp = frame.Skeleton.WorldPosition
            try:
                if hasattr(wp, "X") and hasattr(wp, "Y") and hasattr(wp, "Z"):
                    x = np.array(wp.X, dtype=np.float32)
                    y = np.array(wp.Y, dtype=np.float32)
                    z = np.array(wp.Z, dtype=np.float32)
                    if x.size == 20:
                        skeleton_data[i, :, 0] = x
                        skeleton_data[i, :, 1] = y
                        skeleton_data[i, :, 2] = z
                elif isinstance(wp, np.ndarray) and wp.shape == (20, 3):
                    skeleton_data[i] = wp
            except:
                pass
    return skeleton_data


def process_audio(audio_path, target_frames):
    if not os.path.exists(audio_path):
        return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != Config.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, Config.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=Config.AUDIO_SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.mean(dim=0).transpose(0, 1).numpy()  # [Time, N_MFCC]

        # Interpolate to match video frames
        if mfcc.shape[0] != target_frames:
            if mfcc.shape[0] > 0:
                x_old = np.linspace(0, 1, mfcc.shape[0])
                x_new = np.linspace(0, 1, target_frames)
                mfcc_new = np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)
                for i in range(Config.N_MFCC):
                    mfcc_new[:, i] = np.interp(x_new, x_old, mfcc[:, i])
                return mfcc_new
            else:
                return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)
        return mfcc
    except:
        return np.zeros((target_frames, Config.N_MFCC), dtype=np.float32)


def augment_skeleton(skeleton, rotation=True, scale=True, noise_sigma=0.01):
    aug_skeleton = skeleton.copy()

    if rotation:
        theta = np.random.uniform(-0.3, 0.3)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        shape = aug_skeleton.shape
        reshaped = aug_skeleton.reshape(-1, 3)
        reshaped = np.dot(reshaped, R.T)
        aug_skeleton = reshaped.reshape(shape)

    if scale:
        s = np.random.uniform(0.9, 1.1)
        aug_skeleton = aug_skeleton * s

    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, aug_skeleton.shape)
        aug_skeleton = aug_skeleton + noise

    return aug_skeleton


def compute_kinematics(skeleton):
    # Root-relative centering (using first joint as root approximation)
    root = skeleton[:, 0:1, :]
    centered = skeleton - root

    # Velocity & Acceleration (Central Differences)
    vel = np.gradient(centered, axis=0)
    acc = np.gradient(vel, axis=0)

    T = skeleton.shape[0]
    pos_flat = centered.reshape(T, -1)
    vel_flat = vel.reshape(T, -1)
    acc_flat = acc.reshape(T, -1)

    # 20 joints * 3 coords * 3 features = 180 dimensions
    return np.concatenate([pos_flat, vel_flat, acc_flat], axis=1)


# ==========================================
# Dataset Class
# ==========================================


class GestureDataset(Dataset):
    def __init__(self, csv_file, mode="train", load_cached_data=True):
        self.mode = mode
        self.df = pd.read_csv(csv_file)
        self.samples = []

        cache_file = os.path.join(Config.CACHE_DIR, f"dataset_{mode}.npz")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {mode} data from cache...")
            try:
                data = np.load(cache_file, allow_pickle=True)
                self.samples = data["samples"]
            except:
                self.process_and_cache(cache_file)
        else:
            self.process_and_cache(cache_file)

    def process_and_cache(self, cache_file):
        print(f"Processing {self.mode} data...")
        processed_samples = []

        for idx, row in self.df.iterrows():
            sample_id = row["sample_id"]
            data_path = os.path.join(Config.INPUT_DIR, row["data_path"])
            audio_path = os.path.join(Config.INPUT_DIR, row["audio_path"])

            mat = robust_load_mat(data_path)
            if mat is None:
                continue

            num_frames = int(mat.Video.NumFrames) if hasattr(mat, "Video") else 0
            if num_frames == 0:
                continue

            raw_skeleton = get_skeleton_data(mat, num_frames)
            mfcc = process_audio(audio_path, num_frames)

            labels = np.zeros(num_frames, dtype=np.int64)
            if self.mode != "test":
                label_list = json.loads(row["labels"])
                for l in label_list:
                    start = max(0, l["begin"] - 1)
                    end = min(num_frames, l["end"])
                    labels[start:end] = l["id"]

            processed_samples.append(
                {
                    "id": sample_id,
                    "skeleton": raw_skeleton,
                    "audio": mfcc,
                    "labels": labels,
                    "length": num_frames,
                }
            )

        self.samples = processed_samples
        np.savez_compressed(cache_file, samples=self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        skeleton = sample["skeleton"]
        audio = sample["audio"]
        labels = sample["labels"]
        length = sample["length"]

        if self.mode == "train":
            if length > Config.WINDOW_SIZE:
                start = np.random.randint(0, length - Config.WINDOW_SIZE)
                end = start + Config.WINDOW_SIZE
            else:
                start = 0
                end = length

            skel_window = skeleton[start:end]
            audio_window = audio[start:end]
            label_window = labels[start:end]

            if len(skel_window) < Config.WINDOW_SIZE:
                pad_len = Config.WINDOW_SIZE - len(skel_window)
                skel_window = np.pad(
                    skel_window, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
                )
                audio_window = np.pad(
                    audio_window, ((0, pad_len), (0, 0)), mode="constant"
                )
                label_window = np.pad(
                    label_window, (0, pad_len), mode="constant", constant_values=0
                )

            skel_aug = augment_skeleton(
                skel_window, rotation=True, scale=True, noise_sigma=0.01
            )
            kinematics = compute_kinematics(skel_aug)
            features = np.concatenate([kinematics, audio_window], axis=1)

            return {
                "features": torch.tensor(features, dtype=torch.float32),
                "labels": torch.tensor(label_window, dtype=torch.long),
            }
        else:
            kinematics = compute_kinematics(skeleton)
            features = np.concatenate([kinematics, audio], axis=1)
            return {
                "features": torch.tensor(features, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "id": sample["id"],
            }


# ==========================================
# Model Architecture (SR-DGN)
# ==========================================


class DecoupledGating(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.gate_fc(x_norm))
        return x * gate


class StochasticDepth(nn.Module):
    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0:
            return x
        if torch.rand(1).item() < self.prob:
            return torch.zeros_like(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        stochastic_prob=0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.stochastic = StochasticDepth(stochastic_prob)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if self.training:
            if torch.rand(1).item() < self.stochastic.prob:
                out = torch.zeros_like(out)
        return self.relu(out + res)


class SRDGN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.gating = DecoupledGating(input_dim)
        self.gru = nn.GRU(
            input_dim,
            Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )
        self.fc1 = nn.Linear(Config.HIDDEN_DIM * 2, num_classes)

        tcn_channels = [64, 64, 64, 64, 64]
        kernel_size = 3
        dilations = [1, 2, 4, 8, 16]

        self.tcn2 = nn.ModuleList()
        in_ch = num_classes
        for i, d in enumerate(dilations):
            out_ch = tcn_channels[i]
            padding = (kernel_size - 1) * d // 2
            self.tcn2.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=padding,
                    dropout=0.2,
                    stochastic_prob=Config.STOCHASTIC_DROP_PROB,
                )
            )
            in_ch = out_ch
        self.fc2 = nn.Linear(in_ch, num_classes)

        self.tcn3 = nn.ModuleList()
        in_ch = num_classes
        for i, d in enumerate(dilations):
            out_ch = tcn_channels[i]
            padding = (kernel_size - 1) * d // 2
            self.tcn3.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=padding,
                    dropout=0.2,
                    stochastic_prob=Config.STOCHASTIC_DROP_PROB,
                )
            )
            in_ch = out_ch
        self.fc3 = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x_gated = self.gating(x)
        gru_out, _ = self.gru(x_gated)
        logits1 = self.fc1(gru_out)
        probs1 = torch.softmax(logits1, dim=2)

        x2 = probs1.transpose(1, 2)
        for layer in self.tcn2:
            x2 = layer(x2)
        x2 = x2.transpose(1, 2)
        logits2 = self.fc2(x2)
        probs2 = torch.softmax(logits2, dim=2)

        x3 = probs2.transpose(1, 2)
        for layer in self.tcn3:
            x3 = layer(x3)
        x3 = x3.transpose(1, 2)
        logits3 = self.fc3(x3)

        return logits1, logits2, logits3


# ==========================================
# Training & Loss
# ==========================================


class CascadedLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.mse_lambda = Config.MSE_LAMBDA

    def forward(self, logits_list, targets):
        total_loss = 0
        C = logits_list[0].shape[2]
        targets_flat = targets.view(-1)
        one_hot = F.one_hot(targets_flat, num_classes=C).float()

        for logits in logits_list:
            logits_flat = logits.reshape(-1, C)
            loss_ce = self.ce(logits_flat, targets_flat)
            probs = torch.softmax(logits_flat, dim=1)
            loss_mse = F.mse_loss(probs, one_hot)
            total_loss += loss_ce + self.mse_lambda * loss_mse
        return total_loss


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "train.csv"), mode="train"
    )
    val_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Input dim: 180 (kinematics) + 13 (audio) = 193
    model = SRDGN(input_dim=193, num_classes=Config.NUM_CLASSES).to(device)

    weights = torch.ones(Config.NUM_CLASSES).to(device)
    weights[0] = Config.BACKGROUND_WEIGHT
    criterion = CascadedLoss(weight=weights)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            l1, l2, l3 = model(features)
            loss = criterion([l1, l2, l3], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                l1, l2, l3 = model(features)
                loss = criterion([l1, l2, l3], labels)
                val_loss += loss.item()
                preds = torch.argmax(l3, dim=2)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


# ==========================================
# Inference & Submission
# ==========================================


def rle_encoding(preds):
    if len(preds) == 0:
        return ""
    segments = []
    current_segment = []

    for p in preds:
        if p != 0:
            current_segment.append(p)
        else:
            if len(current_segment) >= Config.MIN_GESTURE_LENGTH:
                vals, counts = np.unique(current_segment, return_counts=True)
                segments.append(vals[np.argmax(counts)])
            current_segment = []

    if len(current_segment) >= Config.MIN_GESTURE_LENGTH:
        vals, counts = np.unique(current_segment, return_counts=True)
        segments.append(vals[np.argmax(counts)])

    return ",".join(map(str, segments))


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRDGN(input_dim=193, num_classes=Config.NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()

    test_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "test.csv"), mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []
    print("Generating submission...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            sample_id = batch["id"][0]
            _, _, logits3 = model(features)
            preds = torch.argmax(logits3, dim=2).squeeze(0).cpu().numpy()
            pred_str = rle_encoding(preds)
            results.append(f"{sample_id},{pred_str}")

    with open(os.path.join(Config.SUBMISSION_DIR, "submission.csv"), "w") as f:
        for line in results:
            f.write(line + "\n")
    print("Submission saved.")


def run():
    train_model()
    generate_submission()
