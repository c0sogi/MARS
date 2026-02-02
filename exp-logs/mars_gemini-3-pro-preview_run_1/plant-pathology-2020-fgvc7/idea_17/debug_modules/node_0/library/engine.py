import torch
import numpy as np
from library.config import Config
from library.model import train_one_epoch, validate


def check_initial_loss(model, loader, criterion, device):
    """
    Performs a sanity check on the initial loss of the model.
    Ensures that the model is not starting with a loss significantly higher than random guessing.
    """
    model.eval()
    with torch.no_grad():
        # Fetch the first batch
        try:
            batch = next(iter(loader))
            inputs, targets = batch
        except StopIteration:
            print("Error: Loader is empty.")
            return

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets).item()

        print(
            f"Initial Loss Check: {loss:.8f} (Threshold: {Config.INITIAL_LOSS_THRESHOLD})"
        )

        if loss > Config.INITIAL_LOSS_THRESHOLD:
            raise AssertionError(
                f"Initial loss {loss:.8f} is too high (expected < {Config.INITIAL_LOSS_THRESHOLD}). "
                "Check model initialization and weight loading."
            )


def train_fn(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    Wraps the library.model.train_one_epoch function.

    Returns:
        tuple: (epoch_loss, epoch_auc)
    """
    return train_one_epoch(model, loader, criterion, optimizer, device)


def eval_fn(model, loader, criterion, device, use_tta=False):
    """
    Evaluates the model on the validation set.
    Wraps the library.model.validate function.

    Args:
        model: The PyTorch model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Computation device.
        use_tta (bool): Whether to use Test Time Augmentation.

    Returns:
        tuple: (epoch_loss, epoch_auc, predictions, targets)
    """
    return validate(model, loader, criterion, device, use_tta=use_tta)


def inference_fn(model, loader, device, use_tta=False):
    """
    Generates predictions for the test set.
    Handles the specific structure of the test loader (image, image_id) and implements TTA.

    Args:
        model: The PyTorch model.
        loader: Test DataLoader.
        device: Computation device.
        use_tta (bool): Whether to use Test Time Augmentation (Horizontal + Vertical Flip).

    Returns:
        np.ndarray: Predicted probabilities of shape (N_samples, N_classes).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Test loader yields (image, image_id)
            # We only need the image for prediction
            inputs = batch[0].to(device)

            # Standard forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            if use_tta:
                # Horizontal Flip
                inputs_h = torch.flip(inputs, dims=[3])
                outputs_h = model(inputs_h)
                probs_h = torch.softmax(outputs_h, dim=1)

                # Vertical Flip
                inputs_v = torch.flip(inputs, dims=[2])
                outputs_v = model(inputs_v)
                probs_v = torch.softmax(outputs_v, dim=1)

                # Average predictions
                probs = (probs + probs_h + probs_v) / 3.0

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)
