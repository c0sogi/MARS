import torch
from torch.utils.data import Dataset, DataLoader
from library import config, features


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    """

    def __init__(self, x_cont, u_out, ids, y=None):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.long)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        item = {
            "x_cont": self.x_cont[idx],
            "u_out": self.u_out[idx],
            "ids": self.ids[idx],
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


def get_data_loaders(load_cached_data=True, batch_size=config.BATCH_SIZE):
    """
    Loads data using the features library and returns PyTorch DataLoaders.
    """
    data = features.prepare_datasets(load_cached_data=load_cached_data)

    train_dataset = VentilatorDataset(
        x_cont=data["train_x_cont"],
        u_out=data["train_uout"],
        ids=data["train_ids"],
        y=data["train_y"],
    )

    val_dataset = VentilatorDataset(
        x_cont=data["val_x_cont"],
        u_out=data["val_uout"],
        ids=data["val_ids"],
        y=data["val_y"],
    )

    test_dataset = VentilatorDataset(
        x_cont=data["test_x_cont"],
        u_out=data["test_uout"],
        ids=data["test_ids"],
        y=None,
    )

    # Initialize DataLoaders
    # Train loader: Shuffle is True, drop_last is True to maintain consistent batch shapes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test loaders: Shuffle is False, drop_last is False to process all data
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
