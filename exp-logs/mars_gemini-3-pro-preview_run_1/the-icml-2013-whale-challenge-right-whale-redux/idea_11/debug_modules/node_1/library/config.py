import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, ResNet18_Weights


# ==========================================
# Configuration
# ==========================================
class Config:
    # Paths
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Audio Params
    SAMPLE_RATE = 2000
    DURATION = 2.0
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 20  # Results in ~200 frames for 2s (2000*2/20)
    F_MIN = 20
    F_MAX = 1000

    # Model Params
    HIDDEN_SIZE = 256  # BiGRU hidden size (output will be 512)
    PROJECTION_DIM = 512

    # Training Params
    SEED = 42
    BATCH_SIZE = 32
    N_EPOCHS = 25
    LEARNING_RATE = 1e-3
    POS_WEIGHT = 9.0
    MIXUP_ALPHA = 0.4
    PATIENCE = 5  # Early stopping patience

    # Augmentation
    TIME_MASK_PARAM = 20  # Max 20 frames (~200ms)
    FREQ_MASK_PARAM = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# Model Architecture
# ==========================================


class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x: (Batch, Time, Feats)
        weights = self.attention(x)
        out = torch.sum(x * weights, dim=1)
        return out


class HierarchicalResNetCRNN(nn.Module):
    def __init__(self):
        super(HierarchicalResNetCRNN, self).__init__()

        # Load Pretrained ResNet18
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer for 1 channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize with average of pretrained weights
        with torch.no_grad():
            self.conv1.weight.data = backbone.conv1.weight.data.mean(
                dim=1, keepdim=True
            )

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Layers with Coordinate Attention Injection
        self.layer1 = self._inject_ca(backbone.layer1, 64)
        self.layer2 = self._inject_ca(backbone.layer2, 128)
        self.layer3 = self._inject_ca(backbone.layer3, 256)
        self.layer4 = self._inject_ca(backbone.layer4, 512)

        # Modify strides for Time Preservation in Layer 3 and 4
        # Original L3 stride is 2. We want (2, 1) -> Freq downsample, Time preserve
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)

        # Original L4 stride is 2. We want (2, 1)
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

        # Hierarchical Aggregation Projection
        # L2: 128 ch, L3: 256 ch, L4: 512 ch -> Total 896
        self.projection = nn.Conv1d(
            128 + 256 + 512, Config.PROJECTION_DIM, kernel_size=1
        )

        # Temporal Modeling
        self.gru = nn.GRU(
            input_size=Config.PROJECTION_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Attention Pooling & Classifier
        self.attn_pool = AttentionPooling(Config.HIDDEN_SIZE * 2)
        self.classifier = nn.Linear(Config.HIDDEN_SIZE * 2, 1)

    def _inject_ca(self, layer, channels):
        # Wraps each block in the layer with Coordinate Attention
        new_layers = []
        for block in layer:
            new_layers.append(block)
            new_layers.append(CoordinateAttention(channels))
        return nn.Sequential(*new_layers)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, F/4, T/4)

        l1 = self.layer1(x)  # (B, 64, F/4, T/4)
        l2 = self.layer2(
            l1
        )  # (B, 128, F/8, T/8) -> wait, standard resnet L2 stride is 2.
        # Let's trace standard resnet:
        # Conv1: /2
        # Maxpool: /2 -> Total /4
        # L1: stride 1 -> /4
        # L2: stride 2 -> /8
        # L3: stride (2,1) -> F/16, T/8
        # L4: stride (2,1) -> F/32, T/8

        l3 = self.layer3(l2)  # (B, 256, F/16, T/8)
        l4 = self.layer4(l3)  # (B, 512, F/32, T/8)

        # Feature Aggregation
        # Global Average Pooling over Frequency dimension
        # l2: (B, 128, F2, T) -> (B, 128, T)
        f2 = torch.mean(l2, dim=2)
        f3 = torch.mean(l3, dim=2)
        f4 = torch.mean(l4, dim=2)

        # Ensure time dimensions match (should be identical due to stride settings, but safety check)
        min_t = min(f2.shape[2], f3.shape[2], f4.shape[2])
        f2 = f2[:, :, :min_t]
        f3 = f3[:, :, :min_t]
        f4 = f4[:, :, :min_t]

        # Concatenate
        combined = torch.cat([f2, f3, f4], dim=1)  # (B, 896, T)

        # Project
        projected = self.projection(combined)  # (B, 512, T)

        # Prepare for GRU (B, T, C)
        gru_in = projected.permute(0, 2, 1)

        gru_out, _ = self.gru(gru_in)  # (B, T, 512)

        # Attention Pooling
        pooled = self.attn_pool(gru_out)  # (B, 512)

        logits = self.classifier(pooled)
        return logits


# ==========================================
# Data Processing
# ==========================================


def load_and_process_data(metadata_path, cache_name, load_cached_data=True):
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    data_path = os.path.join(Config.CACHE_DIR, f"{cache_name}_data.npy")
    labels_path = os.path.join(Config.CACHE_DIR, f"{cache_name}_labels.npy")
    ids_path = os.path.join(Config.CACHE_DIR, f"{cache_name}_ids.npy")

    if load_cached_data and os.path.exists(data_path):
        print(f"Loading cached {cache_name} data...")
        data = np.load(data_path)
        ids = np.load(ids_path, allow_pickle=True)
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
            return data, labels, ids
        return data, None, ids

    print(f"Processing {cache_name} data from scratch...")
    df = pd.read_csv(metadata_path)

    # Pre-allocate arrays
    n_samples = len(df)
    # Calculate shape based on params
    # n_fft=1024, hop=20, sr=2000, dur=2.0 -> 2000*2 = 4000 samples.
    # n_frames = (4000 - 1024) // 20 + 1 ... approx 200.
    # Let's compute one to get exact shape
    dummy_waveform = torch.zeros(int(Config.SAMPLE_RATE * Config.DURATION))
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
    )
    dummy_spec = mel_transform(dummy_waveform)
    n_freq, n_time = dummy_spec.shape

    data_arr = np.zeros((n_samples, n_freq, n_time), dtype=np.float32)
    labels_arr = np.zeros(n_samples, dtype=np.float32)
    ids_arr = np.array(df["clip"].values)

    mel_transform = mel_transform  # Reuse
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    for i, row in df.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        try:
            waveform, sr = torchaudio.load(filepath)

            # Resample if needed (though analysis said all 2000Hz)
            if sr != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or Trim to fixed duration
            target_len = int(Config.SAMPLE_RATE * Config.DURATION)
            current_len = waveform.shape[1]
            if current_len < target_len:
                pad_amt = target_len - current_len
                waveform = F.pad(waveform, (0, pad_amt))
            elif current_len > target_len:
                waveform = waveform[:, :target_len]

            spec = mel_transform(waveform)
            spec = amp_to_db(spec)

            data_arr[i] = spec.squeeze().numpy()
            if "label" in row:
                labels_arr[i] = row["label"]

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    np.save(data_path, data_arr)
    np.save(ids_path, ids_arr)
    if "label" in df.columns:
        np.save(labels_path, labels_arr)
        return data_arr, labels_arr, ids_arr

    return data_arr, None, ids_arr


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, transform=None, is_train=False):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.ids = None  # Not needed for training loop
        self.transform = transform
        self.is_train = is_train

        self.spec_aug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.FREQ_MASK_PARAM
            ),
            torchaudio.transforms.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Add channel dimension: (F, T) -> (1, F, T)
        spec = self.data[idx].unsqueeze(0)
        label = self.labels[idx] if self.labels is not None else torch.tensor(0.0)

        if self.is_train:
            # SpecAugment
            spec = self.spec_aug(spec)

        return spec, label


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ==========================================
# Training & Evaluation
# ==========================================


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, Config.MIXUP_ALPHA
        )
        inputs, targets_a, targets_b = map(
            torch.autograd.Variable, (inputs, targets_a, targets_b)
        )

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()

        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            # Handle single item batch edge case
            if np.ndim(probs) == 0:
                probs = [probs]
            all_preds.extend(probs)

    return all_preds


# ==========================================
# Main Execution
# ==========================================


def main():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading and processing data...")
    train_data, train_labels, _ = load_and_process_data(
        os.path.join(Config.METADATA_DIR, "train.csv"), "train"
    )
    val_data, val_labels, _ = load_and_process_data(
        os.path.join(Config.METADATA_DIR, "val.csv"), "val"
    )
    test_data, _, test_ids = load_and_process_data(
        os.path.join(Config.METADATA_DIR, "test.csv"), "test"
    )

    # 2. Datasets & Loaders
    train_dataset = WhaleDataset(train_data, train_labels, is_train=True)
    val_dataset = WhaleDataset(val_data, val_labels, is_train=False)
    test_dataset = WhaleDataset(test_data, labels=None, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Setup
    model = HierarchicalResNetCRNN().to(device)

    # Handle Class Imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.N_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.N_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print("New best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # 6. Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df = pd.DataFrame({"clip": test_ids, "probability": predictions})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head().to_string())


if __name__ == "__main__":
    main()
