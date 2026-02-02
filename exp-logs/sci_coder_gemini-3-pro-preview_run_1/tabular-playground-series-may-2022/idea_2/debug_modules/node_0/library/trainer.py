import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.utils import seed_everything, get_device
from library.dataset import get_datasets
from library.model import (
    ManufacturingNet,
    train_one_epoch,
    validate,
    generate_submission,
)


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction processes.
    Encapsulates the model, optimizer, scheduler, and early stopping logic.
    """

    def __init__(self, model, optimizer, criterion, scheduler, device, patience=5):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.best_auc = -float("inf")
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """
        Executes one training epoch.
        Returns:
            train_loss (float): Average loss for the epoch.
            train_auc (float): ROC AUC score for the epoch.
        """
        return train_one_epoch(
            self.model,
            train_loader,
            self.criterion,
            self.optimizer,
            self.device,
            self.scheduler,
        )

    def validate(self, val_loader):
        """
        Executes validation on the validation set.
        Returns:
            val_loss (float): Average loss for the validation set.
            val_auc (float): ROC AUC score for the validation set.
        """
        return validate(self.model, val_loader, self.criterion, self.device)

    def fit(self, train_loader, val_loader, epochs):
        """
        Runs the training loop for a specified number of epochs with Early Stopping.
        """
        print("Starting training...")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_auc = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Print metrics with full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Train AUC: {train_auc} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping Logic
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore the best model state
        if self.best_model_state is not None:
            print(f"Loading best model with Val AUC: {self.best_auc}")
            self.model.load_state_dict(self.best_model_state)

    def predict(self, test_loader, output_path="./submission/submission.csv"):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        generate_submission(self.model, test_loader, self.device, output_path)


def run_experiment(
    epochs=20,
    batch_size=1024,
    learning_rate=1e-3,
    embedding_dim=32,
    hidden_units=[512, 256, 128],
    dropout_rate=0.2,
    patience=5,
    sample_size=None,
    load_cached_data=True,
    output_path="./submission/submission.csv",
):
    """
    Orchestrates the entire experiment: data loading, model setup, training, and prediction.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Initial learning rate.
        embedding_dim (int): Dimension of character embeddings.
        hidden_units (list): List of hidden layer sizes for the MLP.
        dropout_rate (float): Dropout probability.
        patience (int): Patience for early stopping.
        sample_size (int, optional): Number of samples to use for debugging.
        load_cached_data (bool): Whether to attempt loading preprocessed data from cache.
        output_path (str): Path to save the submission file.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading datasets...")
    # get_datasets handles the caching logic internally via process_data
    train_ds, val_ds, test_ds = get_datasets(
        load_cached_data=load_cached_data, sample_size=sample_size
    )

    # 3. Create DataLoaders
    # num_workers=4 is appropriate for the given 12 vCPUs
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 4. Determine Model Configuration
    # Inspect dataset tensors directly to determine shapes
    num_numerical = train_ds.X_num.shape[1]
    seq_len = train_ds.X_seq.shape[1]

    # Calculate vocab size based on max index in all splits
    max_idx_train = train_ds.X_seq.max()
    max_idx_val = val_ds.X_seq.max()
    max_idx_test = test_ds.X_seq.max()
    vocab_size = int(max(max_idx_train, max_idx_val, max_idx_test)) + 1

    print(
        f"Model Configuration: Num Features={num_numerical}, Seq Len={seq_len}, Vocab Size={vocab_size}"
    )

    # 5. Initialize Model
    model = ManufacturingNet(
        num_numerical_features=num_numerical,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    ).to(device)

    # 6. Setup Optimization
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        patience=patience,
    )

    # 8. Run Training
    trainer.fit(train_loader, val_loader, epochs)

    # 9. Generate Submission
    trainer.predict(test_loader, output_path=output_path)
