import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.model import DSTResNet
from library.data_loader import get_dataloaders


class Trainer:
    def __init__(
        self,
        model,
        device,
        criterion,
        optimizer,
        scheduler,
        patience=5,
        checkpoint_dir="./working/idea_3",
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(dataloader.dataset)

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

        return running_loss / len(dataloader.dataset)

    def fit(self, train_loader, val_loader, epochs):
        print(f"Training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} - Train MAE: {train_loss} - Val MAE: {val_loss}"
            )

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Best Validation MAE: {self.best_val_loss}")


def train_model(
    epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    patience=5,
    window_size=11,
    load_cached_data=True,
):
    """
    Main function to setup and run the training pipeline.
    """
    # 1. Prepare Data
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=batch_size,
        window_size=window_size,
        load_cached_data=load_cached_data,
    )

    # 2. Setup Model
    # Based on library.model.process_data:
    # Dynamic features: dLat, dLon, dAlt, MeanCn0, MeanUnc, SatCount (6)
    # Static features: WlsLat, WlsLon (2)
    dynamic_features = 6
    static_features = 2

    model = DSTResNet(
        dynamic_features=dynamic_features,
        static_features=static_features,
        window_size=window_size,
        hidden_dim=128,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Setup Training Components
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Run Training
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=patience,
    )

    trainer.fit(train_loader, val_loader, epochs)
