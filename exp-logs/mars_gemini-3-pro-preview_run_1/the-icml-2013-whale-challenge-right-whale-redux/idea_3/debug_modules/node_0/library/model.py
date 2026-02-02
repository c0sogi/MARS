import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import pandas as pd
import numpy as np
import os
import time

from library.config import Config
from library.dataset import get_dataloaders
from library.utils import calculate_auc

# ==========================================
# Model Architecture
# ==========================================


class TimePreservingResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18 with random weights (domain is different from ImageNet)
        # We explicitly set weights=None to initialize from scratch
        self.resnet = resnet18(weights=None)

        # Modify first conv layer: 1 input channel instead of 3
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify strides in layer2, layer3, layer4 to preserve time dimension
        # Stride (2, 1) means downsample Frequency (H) but keep Time (W)
        # We modify the first block of each layer where downsampling occurs

        # Layer 2
        self.resnet.layer2[0].conv1.stride = (2, 1)
        if self.resnet.layer2[0].downsample is not None:
            self.resnet.layer2[0].downsample[0].stride = (2, 1)

        # Layer 3
        self.resnet.layer3[0].conv1.stride = (2, 1)
        if self.resnet.layer3[0].downsample is not None:
            self.resnet.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4
        self.resnet.layer4[0].conv1.stride = (2, 1)
        if self.resnet.layer4[0].downsample is not None:
            self.resnet.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x):
        # Input: (B, 1, F, T)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, Features)
        # scores: (B, T, 1)
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)

        # Weighted sum: (B, Features)
        context = torch.sum(x * weights, dim=1)
        return context


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TimePreservingResNet()

        # ResNet18 ends with 512 channels.
        # Frequency dimension is downsampled by 32x total (2*2*2*2*2).
        # Input F=128 -> 128 / 32 = 4.
        self.backbone_out_dim = Config.BACKBONE_OUT_CHANNELS  # 512

        self.gru = nn.GRU(
            input_size=self.backbone_out_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        rnn_out_dim = Config.RNN_HIDDEN_DIM * 2

        self.attention = AttentionPooling(rnn_out_dim, Config.ATTENTION_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, Config.ATTENTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.ATTENTION_DIM, Config.NUM_CLASSES),
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.backbone(x)  # (B, 512, F_small, T_small)

        # Pool frequency dimension to 1 (Adaptive pool handles any size > 0)
        x = F.adaptive_avg_pool2d(x, (1, None))  # (B, 512, 1, T_small)

        # Prepare for RNN
        x = x.squeeze(2)  # (B, 512, T_small)
        x = x.permute(0, 2, 1)  # (B, T_small, 512)

        # RNN
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # (B, T_small, 2*Hidden)

        # Attention Pooling
        x = self.attention(x)  # (B, 2*Hidden)

        # Classifier
        logits = self.classifier(x)  # (B, 1)
        return logits


# ==========================================
# Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = calculate_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = calculate_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def generate_predictions(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())

    return all_preds


def run_pipeline():
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Model
    print("Initializing Model...")
    model = CRNN().to(device)

    # 3. Optimization
    # Handle class imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
        verbose=True,
    )

    # 4. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Scheduler step (maximize AUC)
        scheduler.step(val_auc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved (AUC: {best_val_auc:.6f})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Submission
    print("Generating Submission...")
    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    predictions = generate_predictions(model, test_loader, device)

    # Load test csv to get clip IDs
    df_test = pd.read_csv(Config.TEST_CSV)

    # Verify lengths match
    if len(df_test) != len(predictions):
        print(
            f"Error: Mismatch in predictions length. DF: {len(df_test)}, Preds: {len(predictions)}"
        )
        # Truncate or pad if absolutely necessary to save, though this indicates a bug
        min_len = min(len(df_test), len(predictions))
        df_test = df_test.iloc[:min_len]
        predictions = predictions[:min_len]

    df_test["probability"] = predictions

    # Save
    # Ensure columns are clip, probability
    submission_df = df_test[["clip", "probability"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute the pipeline
run_pipeline()
