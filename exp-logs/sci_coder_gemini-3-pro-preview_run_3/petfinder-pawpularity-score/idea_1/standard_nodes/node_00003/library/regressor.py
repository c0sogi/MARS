import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import compute_rmse, save_submission, set_seed


class PawpularityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x)


class NeuralRegressor:
    """
    A wrapper around a PyTorch MLP that handles feature concatenation,
    scaling, and prediction clipping for the Pawpularity score.
    Cite solution_lesson_node_00002: Replaces linear head with MLP.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.scaler = StandardScaler()
        self.model = None

    def _prepare_features(self, img_features, meta_features):
        """
        Concatenates image embeddings and metadata features.
        """
        if not isinstance(img_features, np.ndarray):
            img_features = np.array(img_features)
        if not isinstance(meta_features, np.ndarray):
            meta_features = np.array(meta_features)
        return np.hstack([img_features, meta_features])

    def fit(self, img_features, meta_features, targets):
        """
        Fits the MLP model to the training data.
        """
        # 1. Prepare and Scale Data
        X = self._prepare_features(img_features, meta_features)
        X_scaled = self.scaler.fit_transform(X)

        # Convert to Tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=True, drop_last=True
        )

        # 2. Initialize Model
        input_dim = X.shape[1]
        self.model = PawpularityMLP(
            input_dim, Config.MLP_HIDDEN_DIM, Config.MLP_DROPOUT
        ).to(self.device)

        # 3. Setup Training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LR,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )

        # 4. Training Loop
        self.model.train()
        print(f"Training MLP for {Config.MLP_EPOCHS} epochs...")

        for epoch in range(Config.MLP_EPOCHS):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            # Calculate average RMSE for the epoch
            epoch_rmse = np.sqrt(epoch_loss / len(dataset))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{Config.MLP_EPOCHS} - Loss: {epoch_rmse:.4f}")

    def predict(self, img_features, meta_features):
        """
        Generates predictions for the given features.
        """
        self.model.eval()
        X = self._prepare_features(img_features, meta_features)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()

        # Clip predictions to the valid range of the dataset
        return np.clip(preds, 1.0, 100.0)
