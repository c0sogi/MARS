import torch
import torch.optim as optim
import os
from library.config import Config
from library.utils import seed_everything, load_glove_embeddings
from library.data import get_vocab, get_dataloaders
from library.model import CQCRNN, train_one_epoch, validate, generate_submission


class Trainer:
    """
    Encapsulates the training and validation lifecycle.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float("inf")

    def fit(
        self, num_epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE
    ):
        """
        Runs the training loop with early stopping.
        """
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(num_epochs):
            # Run one training epoch
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.device
            )

            # Run validation
            val_loss, val_acc = validate(self.model, self.val_loader, self.device)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Long Acc: {val_acc}"
            )

            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Ensure directory exists before saving
                os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(
                    f"Validation loss improved. Saved model to {Config.MODEL_SAVE_PATH}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break


def train_model(
    epochs=Config.NUM_EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    load_cached_data=True,
):
    """
    Sets up the environment, loads data, initializes the model, and runs the trainer.
    Allows overriding key hyperparameters.
    """
    # Override Config values with arguments
    Config.BATCH_SIZE = batch_size
    Config.LEARNING_RATE = learning_rate
    Config.NUM_EPOCHS = epochs
    Config.EARLY_STOPPING_PATIENCE = patience

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data Preparation
    # Get vocabulary (cached or built from scratch)
    vocab = get_vocab(load_cached_data=load_cached_data)

    # Get DataLoaders (features cached or computed)
    train_loader, val_loader = get_dataloaders(vocab, load_cached_data=load_cached_data)

    # Get Embeddings (cached or computed)
    embeddings = load_glove_embeddings(vocab.stoi, load_cached_data=load_cached_data)

    # 2. Model Initialization
    model = CQCRNN(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        embedding_matrix=embeddings,
    ).to(device)

    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Start Training
    trainer = Trainer(model, train_loader, val_loader, optimizer, device)
    trainer.fit(num_epochs=epochs, patience=patience)


def run_inference():
    """
    Executes the submission generation pipeline defined in the model library.
    This loads the best saved model and generates 'submission.csv'.
    """
    generate_submission()
