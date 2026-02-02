import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config


class TaxiFareMLP(nn.Module):
    """
    Multi-Layer Perceptron for Taxi Fare Prediction.
    Combines learned embeddings for categorical features with dense layers for continuous features.
    """

    def __init__(
        self, embedding_configs, continuous_input_dim, hidden_dims, dropout_rate
    ):
        """
        Args:
            embedding_configs (list of tuples): List of (num_embeddings, embedding_dim) for each categorical feature.
            continuous_input_dim (int): Number of continuous features.
            hidden_dims (list of int): Dimensions of hidden layers.
            dropout_rate (float): Dropout probability.
        """
        super(TaxiFareMLP, self).__init__()

        # 1. Embedding Layers
        # We create a list of embedding layers corresponding to the categorical inputs
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, embed_dim)
                for num_embeddings, embed_dim in embedding_configs
            ]
        )

        # Calculate total dimension after concatenating embeddings and continuous features
        total_embed_dim = sum(e_dim for _, e_dim in embedding_configs)
        input_dim = total_embed_dim + continuous_input_dim

        # 2. MLP Layers
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim

        # Output Layer (Regression)
        layers.append(nn.Linear(curr_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat (torch.Tensor): LongTensor of shape (batch_size, num_cat_features).
            x_cont (torch.Tensor): FloatTensor of shape (batch_size, num_cont_features).
        """
        # Process Embeddings
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] contains indices for the i-th categorical feature
            embedded.append(emb_layer(x_cat[:, i]))

        # Concatenate all embeddings: (batch_size, total_embed_dim)
        x_emb = torch.cat(embedded, dim=1)

        # Concatenate with continuous features: (batch_size, total_embed_dim + cont_dim)
        x = torch.cat([x_emb, x_cont], dim=1)

        # Pass through MLP
        output = self.mlp(x)

        # Squeeze to shape (batch_size,) to match target shape
        return output.squeeze(1)


class NNPredictor:
    """
    Wrapper for training, evaluating, and predicting with the TaxiFareMLP.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = None
        self.config_state = None  # To store input dims for saving/loading

        # Hyperparameters
        self.params = Config.NN_PARAMS
        self.patience = self.params["patience"]
        self.epochs = self.params["epochs"]
        self.lr = self.params["learning_rate"]
        self.wd = self.params["weight_decay"]

    def _get_embedding_configs(self):
        """
        Maps the fixed categorical columns to their (vocab_size, embed_dim).
        Order matches data_loader.py: [pickup_cluster, dropoff_cluster, hour, day_of_week, year]
        """
        emb_dims = self.params["embedding_dims"]
        # Vocab sizes:
        # Clusters: Config.N_CLUSTERS
        # Hour: 24
        # Day of Week: 7
        # Year: We use 2020 as a safe upper bound (data covers ~2009-2015)

        configs = [
            (Config.N_CLUSTERS, emb_dims["cluster"]),  # pickup_cluster
            (Config.N_CLUSTERS, emb_dims["cluster"]),  # dropoff_cluster
            (24, emb_dims["hour"]),  # hour
            (7, emb_dims["dow"]),  # day_of_week
            (2020, emb_dims["year"]),  # year
        ]
        return configs

    def fit(self, train_loader, val_loader):
        """
        Trains the neural network with early stopping.
        """
        # 1. Infer Input Dimensions from a batch
        # We assume loaders are not empty
        sample_cat, sample_cont, _ = next(iter(train_loader))
        continuous_input_dim = sample_cont.shape[1]

        embedding_configs = self._get_embedding_configs()

        # Store config for saving
        self.config_state = {
            "embedding_configs": embedding_configs,
            "continuous_input_dim": continuous_input_dim,
            "hidden_dims": self.params["hidden_dims"],
            "dropout": self.params["dropout"],
        }

        # 2. Initialize Model
        self.model = TaxiFareMLP(
            embedding_configs=embedding_configs,
            continuous_input_dim=continuous_input_dim,
            hidden_dims=self.params["hidden_dims"],
            dropout_rate=self.params["dropout"],
        ).to(self.device)

        # 3. Setup Optimization
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        criterion = nn.MSELoss()

        # 4. Training Loop
        best_val_rmse = float("inf")
        patience_counter = 0
        best_model_state = None

        print(f"Starting training on {self.device} for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            # --- Training ---
            self.model.train()
            train_loss_sum = 0.0
            train_count = 0

            for x_cat, x_cont, y in train_loader:
                x_cat = x_cat.to(self.device)
                x_cont = x_cont.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(x_cat, x_cont)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * y.size(0)
                train_count += y.size(0)

            avg_train_loss = train_loss_sum / train_count
            train_rmse = np.sqrt(avg_train_loss)

            # --- Validation ---
            self.model.eval()
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for x_cat, x_cont, y in val_loader:
                    x_cat = x_cat.to(self.device)
                    x_cont = x_cont.to(self.device)
                    y = y.to(self.device)

                    preds = self.model(x_cat, x_cont)
                    loss = criterion(preds, y)

                    val_loss_sum += loss.item() * y.size(0)
                    val_count += y.size(0)

            avg_val_loss = val_loss_sum / val_count
            val_rmse = np.sqrt(avg_val_loss)

            print(
                f"Epoch {epoch+1}/{self.epochs} - Train RMSE: {train_rmse} - Validation RMSE: {val_rmse}"
            )

            # --- Early Stopping ---
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                # Save best state in memory (move to CPU to save GPU RAM)
                best_model_state = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Best Validation RMSE: {best_val_rmse}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        Args:
            test_loader (DataLoader): DataLoader yielding (x_cat, x_cont).
        Returns:
            np.ndarray: Predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                # Handle cases where loader might return (x_cat, x_cont) or (x_cat, x_cont, y)
                if len(batch) >= 2:
                    x_cat, x_cont = batch[0], batch[1]
                else:
                    raise ValueError("Unexpected batch structure in test_loader")

                x_cat = x_cat.to(self.device)
                x_cont = x_cont.to(self.device)

                preds = self.model(x_cat, x_cont)
                predictions.append(preds.cpu().numpy())

        return np.concatenate(predictions)

    def save(self, path):
        """
        Saves the model state and configuration.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config_state": self.config_state,
        }
        torch.save(checkpoint, path)
        print(f"NN model saved to {path}")

    def load(self, path):
        """
        Loads the model state and configuration.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.config_state = checkpoint["config_state"]

        # Re-initialize model architecture
        self.model = TaxiFareMLP(
            embedding_configs=self.config_state["embedding_configs"],
            continuous_input_dim=self.config_state["continuous_input_dim"],
            hidden_dims=self.config_state["hidden_dims"],
            dropout_rate=self.config_state["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"NN model loaded from {path}")
