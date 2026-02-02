import torch
from library.config import CFG
from library.utils import fbeta_score


def train_one_epoch(model, optimizer, dataloader, device, criterion, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (torch.utils.data.DataLoader): The training dataloader.
        device (torch.device): The device to run on.
        criterion (torch.nn.Module): The loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    running_loss = 0.0
    dataset_size = 0

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Note: Scheduler step is handled in the main loop for ReduceLROnPlateau

    epoch_loss = running_loss / dataset_size
    print(f"Train Loss: {epoch_loss}")

    return epoch_loss


def valid_one_epoch(model, dataloader, device, criterion):
    """
    Validates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        device (torch.device): The device to run on.
        criterion (torch.nn.Module): The loss function.

    Returns:
        float: Average loss for the epoch.
        float: Average F0.5 score for the epoch.
    """
    model.eval()

    running_loss = 0.0
    running_score = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * batch_size

            # Apply sigmoid to convert logits to probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            # Calculate batch F0.5 score
            score = fbeta_score(probs, masks, beta=CFG.beta, threshold=CFG.threshold)
            running_score += score * batch_size

            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_score = running_score / dataset_size

    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation F0.5 Score: {epoch_score}")

    return epoch_loss, epoch_score
