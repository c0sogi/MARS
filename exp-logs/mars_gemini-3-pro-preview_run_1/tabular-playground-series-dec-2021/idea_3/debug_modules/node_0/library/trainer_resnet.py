import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_factory import DataFactory
from library.arch_resnet import TabularResNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)

    return val_loss, val_acc


def predict(model, loader, device):
    """
    Generates soft probability predictions for the test set.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for inputs in loader:
            # inputs is a list containing one tensor if from TensorDataset
            x = inputs[0].to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def train_neural_network(load_cached_data=True):
    """
    Main function to train the Tabular ResNet.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        np.ndarray: Probability predictions for the test set.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Training Neural Network on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_ds, val_ds, test_ds, input_dim = DataFactory.create_nn_datasets(
        load_cached_data=load_cached_data
    )

    batch_size = Config.NN_PARAMS["batch_size"]
    num_workers = Config.NN_PARAMS["num_workers"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = TabularResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        hidden_dims=Config.NN_PARAMS["hidden_dims"],
        dropout_rate=Config.NN_PARAMS["dropout_rate"],
        use_batch_norm=Config.NN_PARAMS["use_batch_norm"],
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.NN_PARAMS["learning_rate"],
        weight_decay=Config.NN_PARAMS["weight_decay"],
    )

    # Scheduler: Reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    epochs = Config.NN_PARAMS["epochs"]
    patience = Config.NN_PARAMS["patience"]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Validation Loss: {best_loss}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(best_model_wts)

    print("Generating predictions on Test set...")
    test_probs = predict(model, test_loader, device)

    return test_probs
