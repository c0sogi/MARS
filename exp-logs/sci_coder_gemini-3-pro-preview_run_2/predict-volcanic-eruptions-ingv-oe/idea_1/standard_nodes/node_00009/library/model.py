import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import random
from library.config import Config
from library.feature_extractor import extract_features


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# ---------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------
class VolcanoHybrid(nn.Module):
    """
    Hybrid CNN-MLP Architecture.
    Branch 1: ResNet18 for Log-Mel Spectrograms (Cite solution_lesson_node_00006).
    Branch 2: MLP for Statistical Features.
    """

    def __init__(self, tabular_input_dim, hidden_layers=None, dropout_rate=0.3):
        super(VolcanoHybrid, self).__init__()

        # --- CNN Branch ---
        # Using ResNet18, modifying first layer for 10 channels
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.conv1 = nn.Conv2d(
            10, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove classification head, keep feature vector (512 dim)
        self.cnn.fc = nn.Identity()

        # --- MLP Branch ---
        if hidden_layers is None:
            hidden_layers = [256, 128]

        mlp_layers = []
        in_dim = tabular_input_dim
        for h_dim in hidden_layers:
            mlp_layers.append(nn.Linear(in_dim, h_dim))
            mlp_layers.append(nn.BatchNorm1d(h_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # --- Fusion Head ---
        # ResNet18 output is 512, MLP output is last hidden layer
        fusion_dim = 512 + hidden_layers[-1]
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x is a list/tuple: [spectrogram, tabular]
        spec, tab = x

        # CNN Forward
        cnn_feat = self.cnn(spec)

        # MLP Forward
        mlp_feat = self.mlp(tab)

        # Fusion
        combined = torch.cat((cnn_feat, mlp_feat), dim=1)
        return self.head(combined)


# Note: prepare_data and train_model logic is moved to library/trainer.py and library/data_loader.py
# to avoid circular imports and duplication. This file now mainly holds the Model Class.


# ---------------------------------------------------------
# Inference & Submission
# ---------------------------------------------------------
def generate_submission(model, feature_cols, device, debug_size=None):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating submission...")
    df_test = extract_features(
        Config.TEST_METADATA_PATH, Config.TEST_FEATURES_CACHE, debug_size=debug_size
    )

    X_test = df_test[feature_cols].values.astype(np.float32)
    segment_ids = df_test["segment_id"].values

    # Load Scaler
    mean_path = os.path.join(Config.WORKING_DIR, "scaler_mean.npy")
    scale_path = os.path.join(Config.WORKING_DIR, "scaler_scale.npy")

    if os.path.exists(mean_path) and os.path.exists(scale_path):
        mean = np.load(mean_path)
        scale = np.load(scale_path)
        # Manual standardization
        X_test = (X_test - mean) / scale
    else:
        print("Warning: Scaler files not found. Test data will not be scaled.")

    # Prediction Loop
    model.eval()
    predictions = []

    test_tensor = torch.tensor(X_test)
    dataset = TensorDataset(test_tensor)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].to(device)
            outputs = model(X_batch).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            predictions.extend(outputs.cpu().numpy())

    # Save Submission
    df_sub = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# ---------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------
def run_pipeline():
    """
    Main entry point to execute the full training and submission pipeline.
    """
    Config.setup()
    device = torch.device(Config.DEVICE)

    # 1. Prepare Data
    train_loader, val_loader, input_dim, feature_cols = prepare_data(
        debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # 2. Train Model
    model = train_model(
        train_loader,
        val_loader,
        input_dim,
        device,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
        model_save_path=Config.MODEL_SAVE_PATH,
    )

    # 3. Generate Submission
    generate_submission(
        model, feature_cols, device, debug_size=Config.DEBUG_SAMPLE_SIZE
    )
