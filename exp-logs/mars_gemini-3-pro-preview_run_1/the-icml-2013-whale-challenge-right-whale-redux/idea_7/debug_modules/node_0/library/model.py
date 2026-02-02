import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import calculate_auc


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted sum of the input sequence based on learned attention scores.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        scores = self.attention_weights(x)  # (Batch, Time, 1)
        weights = torch.softmax(scores, dim=1)
        # Weighted sum along the time dimension
        output = torch.sum(x * weights, dim=1)  # (Batch, Features)
        return output


class MultiResResNet34CRNN(nn.Module):
    """
    Multi-Resolution Time-Preserving ResNet-34 CRNN.

    Architecture:
    1. Input: 3-Channel Multi-Resolution Spectrogram (128 Mel bins).
    2. Backbone: ResNet-34 with modified strides in Layer 3 and 4 to preserve Time.
    3. Neck: Frequency Pooling.
    4. Head: Bi-Directional GRU + Attention Pooling + Classifier.
    """

    def __init__(self, pretrained=True):
        super(MultiResResNet34CRNN, self).__init__()

        # Load Pretrained ResNet34
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = models.resnet34(weights=weights)

        # ==========================================
        # Time-Preserving Stride Modification
        # ==========================================
        # Standard ResNet downsamples by 32x (2^5).
        # We want to preserve time in the deeper layers.
        # We change stride from (2, 2) to (2, 1) in Layer 3 and Layer 4.

        # Layer 3 modification
        self.base_model.layer3[0].conv1.stride = (2, 1)
        if self.base_model.layer3[0].downsample is not None:
            self.base_model.layer3[0].downsample[0].stride = (2, 1)

        # Layer 4 modification
        self.base_model.layer4[0].conv1.stride = (2, 1)
        if self.base_model.layer4[0].downsample is not None:
            self.base_model.layer4[0].downsample[0].stride = (2, 1)

        # Remove original classification head
        del self.base_model.avgpool
        del self.base_model.fc

        # ==========================================
        # CRNN Head
        # ==========================================
        # ResNet34 layer4 outputs 512 channels.
        self.rnn_input_dim = 512
        self.rnn_hidden_dim = 256

        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Input to attention is hidden_dim * 2 (bidirectional)
        self.attention = AttentionPooling(self.rnn_hidden_dim * 2)
        self.classifier = nn.Linear(self.rnn_hidden_dim * 2, 1)

    def forward(self, x):
        # Input x: (Batch, 3, 128, Time)

        # Pass through ResNet Backbone
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        # Output x: (Batch, 512, F_reduced, T_preserved)

        # Pool out the Frequency dimension (F_reduced -> 1)
        x = torch.mean(x, dim=2)  # (Batch, 512, T_preserved)

        # Permute for RNN: (Batch, Time, Features)
        x = x.permute(0, 2, 1)

        # Pass through Bi-GRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # (Batch, Time, Hidden*2)

        # Attention Pooling
        x = self.attention(x)  # (Batch, Hidden*2)

        # Classifier
        logits = self.classifier(x)  # (Batch, 1)

        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets_a, targets_b, lam) in enumerate(loader):
        inputs = inputs.to(device)
        targets_a = targets_a.to(device).view(-1, 1)
        targets_b = targets_b.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Mixup Loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(loader)
    auc = calculate_auc(all_targets, all_preds)

    return avg_loss, auc


def run_training(train_loader, val_loader):
    """
    Main training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    model = MultiResResNet34CRNN(pretrained=True).to(device)

    # Weighted Loss for Class Imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_auc = 0.0
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step(val_auc)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f} | "
            f"Time: {duration:.2f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc:.6f}")

    print(f"Training complete. Best Validation AUC: {best_auc:.6f}")
    return save_path


def generate_predictions(model_path, test_loader):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structure (no pretrained weights needed as we load state_dict)
    model = MultiResResNet34CRNN(pretrained=False).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy().flatten())

    # Map predictions to clip IDs using metadata
    # The loader iterates sequentially, so order is preserved.
    df_test = pd.read_csv(Config.TEST_CSV)

    if len(all_preds) != len(df_test):
        print(
            f"Warning: Count mismatch. Preds: {len(all_preds)}, Files: {len(df_test)}"
        )
        # Truncate or pad if necessary, though this shouldn't happen with correct loaders
        if len(all_preds) > len(df_test):
            all_preds = all_preds[: len(df_test)]
        else:
            all_preds.extend([0.0] * (len(df_test) - len(all_preds)))

    df_test["probability"] = all_preds

    # Save submission
    submission_df = df_test[["clip", "probability"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
