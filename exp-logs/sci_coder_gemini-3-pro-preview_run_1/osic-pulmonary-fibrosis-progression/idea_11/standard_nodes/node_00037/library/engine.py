import torch
from library.utils import AverageMeter, laplace_log_likelihood


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader providing the training data.
        optimizer (Optimizer): The optimizer for weight updates.
        criterion (nn.Module): The loss function.
        device (str or torch.device): The device to move tensors to.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, data in enumerate(dataloader):
        # 1. Move data to device
        axial = data["axial"].to(device)
        coronal = data["coronal"].to(device)
        tabular = data["tabular"].to(device)
        time_delta = data["time_delta"].to(device)
        baseline_fvc = data["baseline_fvc"].to(device)
        target = data["target"].to(device)

        batch_size = axial.size(0)

        # 2. Zero gradients
        optimizer.zero_grad()

        # 3. Forward pass
        # Model expects: axial, coronal, tabular, time_delta, baseline_fvc
        pred_fvc, pred_sigma = model(axial, coronal, tabular, time_delta, baseline_fvc)

        # 4. Calculate Loss
        loss = criterion(pred_fvc, pred_sigma, target)

        # 5. Backward pass and optimization
        loss.backward()
        optimizer.step()

        # 6. Update metrics
        loss_meter.update(loss.item(), batch_size)

    return loss_meter.avg


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation or test set.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader providing the data.
        criterion (nn.Module): The loss function.
        device (str or torch.device): The device to move tensors to.

    Returns:
        tuple: (average_loss, average_metric)
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # 1. Move data to device
            axial = data["axial"].to(device)
            coronal = data["coronal"].to(device)
            tabular = data["tabular"].to(device)
            time_delta = data["time_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            target = data["target"].to(device)

            batch_size = axial.size(0)

            # 2. Forward pass
            pred_fvc, pred_sigma = model(
                axial, coronal, tabular, time_delta, baseline_fvc
            )

            # 3. Calculate Loss
            loss = criterion(pred_fvc, pred_sigma, target)
            loss_meter.update(loss.item(), batch_size)

            # 4. Calculate Metric (Laplace Log Likelihood)
            # The utility function handles numpy conversion internally
            score = laplace_log_likelihood(target, pred_fvc, pred_sigma)
            metric_meter.update(score, batch_size)

    return loss_meter.avg, metric_meter.avg
