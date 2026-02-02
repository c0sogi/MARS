import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config


# Set seeds for reproducibility
def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CompositeCNN(nn.Module):
    """
    Composite-Input CNN for Ship vs Iceberg classification.
    """

    def __init__(self, config=None):
        super(CompositeCNN, self).__init__()
        self.config = config if config else Config()

        # Architecture Hyperparameters
        filters = self.config.CONV_FILTERS  # [64, 128, 128]
        dense_units = self.config.DENSE_UNITS  # 512
        dropout_rate = self.config.DROPOUT_RATE  # 0.5
        input_channels = self.config.IMG_CHANNELS  # 3

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=filters[0],
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(
            in_channels=filters[0], out_channels=filters[1], kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(
            in_channels=filters[1], out_channels=filters[2], kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate Flatten Size
        # Input: 75x75
        # Pool 1 (2x2): 37x37
        # Pool 2 (2x2): 18x18
        # Pool 3 (2x2): 9x9
        # Final: 128 * 9 * 9 = 10368
        self.flatten_dim = filters[2] * 9 * 9

        # Dense Layers
        # Input dimension is Flattened Image Features + 1 (Incidence Angle)
        self.fc1 = nn.Linear(self.flatten_dim + 1, dense_units)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_img, x_angle):
        # Block 1
        x = self.conv1(x_img)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Concatenate with Incidence Angle
        # Ensure angle is (Batch, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.unsqueeze(1)

        x = torch.cat((x, x_angle), dim=1)

        # Dense Layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        out = self.sigmoid(x)

        return out


def train_model(train_loader, val_loader, config):
    """
    Trains the CompositeCNN model with Early Stopping.
    """
    set_seeds(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = CompositeCNN(config).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.NUM_EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        for imgs, angles, labels in train_loader:
            imgs = imgs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)  # Match output shape (Batch, 1)

            optimizer.zero_grad()
            outputs = model(imgs, angles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, angles, labels in val_loader:
                imgs = imgs.to(device)
                angles = angles.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(imgs, angles)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            # Save checkpoint
            torch.save(best_model_state, config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def generate_submission(model, test_loader, config):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for imgs, angles in test_loader:
            imgs = imgs.to(device)
            angles = angles.to(device)

            outputs = model(imgs, angles)
            # Flatten predictions to list
            preds = outputs.cpu().numpy().flatten().tolist()
            predictions.extend(preds)

    # Load Test IDs
    # Assuming test_ids are cached or loaded from metadata
    if os.path.exists(config.CACHE_TEST_IDS):
        test_ids = np.load(config.CACHE_TEST_IDS, allow_pickle=True)
    else:
        # Fallback to metadata if cache not found (though data_loader should have created it)
        df_test = pd.read_csv(config.TEST_META_PATH)
        test_ids = df_test["id"].values

    # Ensure lengths match
    if len(test_ids) != len(predictions):
        print(
            f"Warning: Number of test IDs ({len(test_ids)}) does not match predictions ({len(predictions)})."
        )
        # In case of drop_last or other loader issues, truncate or pad?
        # Usually loaders are configured to not drop last for test.
        # We will proceed assuming they match or taking the minimum.
        min_len = min(len(test_ids), len(predictions))
        test_ids = test_ids[:min_len]
        predictions = predictions[:min_len]

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Save
    df_sub.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
