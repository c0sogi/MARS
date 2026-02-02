import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import GASEConfig
from library.data_utils import process_and_cache_data
from library.graph_utils import MoleculeGraphBuilder
from library.mpnn_model import InteractionMPNN, MoleculeDataset, collate_mpnn


class MPNNRunner:
    """
    Manages the training lifecycle and embedding extraction of the MPNN.
    Encapsulates data loading, model initialization, training loop, and inference.
    """

    def __init__(self):
        self.device = torch.device(GASEConfig.DEVICE)
        self.seed = GASEConfig.RANDOM_SEED
        self._set_seed()

    def _set_seed(self):
        """Sets random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def train(
        self,
        load_cached_data=True,
        epochs=GASEConfig.MPNN_EPOCHS,
        batch_size=GASEConfig.MPNN_BATCH_SIZE,
    ):
        """
        Trains the MPNN model using the configured hyperparameters.

        Args:
            load_cached_data (bool): Whether to load processed data from cache.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        print(f"Using device: {self.device}")

        # 1. Prepare Data
        print("Initializing MoleculeGraphBuilder...")
        builder = MoleculeGraphBuilder()

        # Load Graph Data
        train_graph = builder.process_data("train", load_cached_data=load_cached_data)
        val_graph = builder.process_data("val", load_cached_data=load_cached_data)

        # Load Metadata (Targets + Distances)
        df_train, df_val, _ = process_and_cache_data(load_cached_data=load_cached_data)

        # Create Datasets
        train_dataset = MoleculeDataset("train", train_graph, df_train)
        val_dataset = MoleculeDataset("val", val_graph, df_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_mpnn,
            num_workers=GASEConfig.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_mpnn,
            num_workers=GASEConfig.NUM_WORKERS,
            pin_memory=True,
        )

        # 2. Model Setup
        model = InteractionMPNN().to(self.device)
        optimizer = AdamW(
            model.parameters(),
            lr=GASEConfig.MPNN_LEARNING_RATE,
            weight_decay=GASEConfig.MPNN_WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.L1Loss()  # MAE

        # 3. Training Loop
        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting MPNN training...")
        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for batch in train_loader:
                x, edge_index, edge_attr, pair_idx, targets, pair_attr = [
                    b.to(self.device) for b in batch
                ]

                optimizer.zero_grad()
                preds, _ = model(x, edge_index, edge_attr, pair_idx, pair_attr)

                loss = criterion(preds, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                train_loss_sum += loss.item() * targets.size(0)
                train_count += targets.size(0)

            avg_train_loss = train_loss_sum / train_count

            # Validation
            model.eval()
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for batch in val_loader:
                    x, edge_index, edge_attr, pair_idx, targets, pair_attr = [
                        b.to(self.device) for b in batch
                    ]
                    preds, _ = model(x, edge_index, edge_attr, pair_idx, pair_attr)
                    loss = criterion(preds, targets)
                    val_loss_sum += loss.item() * targets.size(0)
                    val_count += targets.size(0)

            avg_val_loss = val_loss_sum / val_count
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Train MAE: {avg_train_loss} | Val MAE: {avg_val_loss}"
            )

            # Checkpoint & Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), GASEConfig.MPNN_MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= GASEConfig.MPNN_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        print(f"Training complete. Best Val MAE: {best_val_loss}")

        # Clean up
        del train_loader, val_loader, train_dataset, val_dataset, model
        gc.collect()
        torch.cuda.empty_cache()

    def extract_embeddings(
        self, load_cached_data=True, batch_size=GASEConfig.MPNN_BATCH_SIZE
    ):
        """
        Generates embeddings for Train, Val, and Test sets using the trained MPNN.
        Saves them to .npy files specified in Config.

        Args:
            load_cached_data (bool): Whether to load processed data from cache.
            batch_size (int): Batch size for inference.
        """
        if not os.path.exists(GASEConfig.MPNN_MODEL_PATH):
            raise FileNotFoundError(
                f"MPNN model not found at {GASEConfig.MPNN_MODEL_PATH}. Run training first."
            )

        print("Loading best MPNN model for embedding extraction...")
        model = InteractionMPNN().to(self.device)
        model.load_state_dict(
            torch.load(GASEConfig.MPNN_MODEL_PATH, map_location=self.device)
        )
        model.eval()

        builder = MoleculeGraphBuilder()

        # Load all metadata
        df_train, df_val, df_test = process_and_cache_data(
            load_cached_data=load_cached_data
        )

        splits = [
            ("train", df_train, GASEConfig.EMBEDDINGS_TRAIN_PATH),
            ("val", df_val, GASEConfig.EMBEDDINGS_VAL_PATH),
            ("test", df_test, GASEConfig.EMBEDDINGS_TEST_PATH),
        ]

        for split_name, df_meta, save_path in splits:
            print(f"Generating embeddings for {split_name}...")

            graph_data = builder.process_data(
                split_name, load_cached_data=load_cached_data
            )
            dataset = MoleculeDataset(split_name, graph_data, df_meta)

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_mpnn,
                num_workers=GASEConfig.NUM_WORKERS,
                pin_memory=True,
            )

            embeddings_list = []

            with torch.no_grad():
                for batch in loader:
                    x, edge_index, edge_attr, pair_idx, _, pair_attr = [
                        b.to(self.device) for b in batch
                    ]

                    # We only need embeddings, ignore predictions
                    _, embeddings = model(x, edge_index, edge_attr, pair_idx, pair_attr)
                    embeddings_list.append(embeddings.cpu().numpy())

            # Concatenate
            full_embeddings = np.concatenate(embeddings_list, axis=0)

            # Save
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, full_embeddings)
            print(f"Saved embeddings to {save_path}. Shape: {full_embeddings.shape}")

            # Cleanup
            del dataset, loader, graph_data, embeddings_list, full_embeddings
            gc.collect()

        # Final cleanup
        del model
        torch.cuda.empty_cache()
