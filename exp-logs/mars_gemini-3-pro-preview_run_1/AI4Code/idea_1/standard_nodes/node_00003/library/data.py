import os
import json
import io
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from typing import List, Dict, Any

from library.config import Config
from library.utils import set_seed


class EmbeddingManager:
    """
    Manages the extraction, embedding, and caching of notebook data.
    Uses a frozen Sentence Transformer to convert text to vectors.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.backbone_name)
        self.model.max_seq_length = config.max_length
        self.model.to(config.device)
        self.model.eval()

    def _preprocess_text(self, text: str) -> str:
        """
        Basic preprocessing and truncation of cell text.
        """
        if not isinstance(text, str):
            return ""
        text = text.strip()
        # Heuristic truncation to avoid processing massive strings before tokenization
        # Assuming approx 4 chars per token, allow a buffer
        limit = self.config.max_length * 5
        return text[:limit]

    def _serialize_numpy(self, arr: np.ndarray) -> bytes:
        """
        Serialize a numpy array to bytes for storage in Parquet.
        """
        with io.BytesIO() as f:
            np.save(f, arr)
            return f.getvalue()

    def process_data(self, split: str, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Main method to load, process, and cache data for a given split.

        Args:
            split: 'train', 'val', or 'test'.
            load_cached_data: If True, attempts to load from disk first.

        Returns:
            pd.DataFrame containing the processed features.
        """
        # Determine paths
        if split == "train":
            meta_path = self.config.train_metadata_path
            cache_path = self.config.train_cache_path
        elif split == "val":
            meta_path = self.config.val_metadata_path
            cache_path = self.config.val_cache_path
        elif split == "test":
            meta_path = self.config.test_metadata_path
            cache_path = self.config.test_cache_path
        else:
            raise ValueError(f"Unknown split: {split}")

        # Try to load cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Computing features for {split} set...")

        # Load metadata
        df_meta = pd.read_csv(meta_path)

        # Debug sampling
        if self.config.debug:
            print(f"Debug mode: sampling {self.config.debug_sample_size} notebooks.")
            df_meta = df_meta.iloc[: self.config.debug_sample_size]

        data_rows = []

        # Process each notebook
        # Note: We process sequentially here. For massive datasets, one might batch
        # the reading, but embedding is the bottleneck which is batched internally.
        for _, row in tqdm(
            df_meta.iterrows(), total=len(df_meta), desc=f"Processing {split}"
        ):
            nb_id = row["id"]
            json_path = os.path.join(self.config.input_dir, row["filepath"])

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    nb_json = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine Cell Lists and Labels
            if split == "test":
                # For test, we rely on the JSON order (insertion order) for code cells
                all_keys = list(cell_types.keys())
                code_ids = [k for k in all_keys if cell_types[k] == "code"]
                markdown_ids = [k for k in all_keys if cell_types[k] == "markdown"]
                # Dummy labels for test
                labels = [-1] * len(markdown_ids)
            else:
                # For train/val, use ground truth order
                cell_order = row["cell_order"].split()
                code_ids = [k for k in cell_order if cell_types.get(k) == "code"]
                markdown_ids = [
                    k for k in cell_order if cell_types.get(k) == "markdown"
                ]

                # Generate Labels: Index of the *next* code cell
                labels = []
                current_code_idx = 0

                for cid in cell_order:
                    ctype = cell_types.get(cid)
                    if ctype == "code":
                        current_code_idx += 1
                    elif ctype == "markdown":
                        labels.append(current_code_idx)

            # Extract Text
            code_texts = [
                self._preprocess_text(sources.get(cid, "")) for cid in code_ids
            ]
            md_texts = [
                self._preprocess_text(sources.get(cid, "")) for cid in markdown_ids
            ]

            # Compute Embeddings
            # Handle empty cases gracefully
            if code_texts:
                code_embs = self.model.encode(
                    code_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            else:
                code_embs = np.zeros((0, self.config.input_dim), dtype=np.float32)

            if md_texts:
                md_embs = self.model.encode(
                    md_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            else:
                md_embs = np.zeros((0, self.config.input_dim), dtype=np.float32)

            # Store Record
            # We serialize numpy arrays to bytes to store in Parquet safely without pickle objects
            data_rows.append(
                {
                    "id": nb_id,
                    "code_embeddings": self._serialize_numpy(code_embs),
                    "markdown_embeddings": self._serialize_numpy(md_embs),
                    "code_ids": json.dumps(code_ids),
                    "markdown_ids": json.dumps(markdown_ids),
                    "labels": json.dumps(labels),
                }
            )

        # Create DataFrame and Save
        df_out = pd.DataFrame(data_rows)

        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Save to Parquet
        # engine='pyarrow' is standard and robust
        df_out.to_parquet(cache_path, engine="pyarrow")
        print(f"Saved features to {cache_path}")

        return df_out


class NotebookDataset(Dataset):
    """
    PyTorch Dataset that serves notebook data from the cached DataFrame.
    """

    def __init__(self, data_df: pd.DataFrame):
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def _deserialize_numpy(self, b: bytes) -> np.ndarray:
        with io.BytesIO(b) as f:
            return np.load(f)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Deserialize embeddings
        code_embs = self._deserialize_numpy(row["code_embeddings"])
        md_embs = self._deserialize_numpy(row["markdown_embeddings"])

        # Parse lists
        labels = json.loads(row["labels"])
        md_ids = json.loads(row["markdown_ids"])

        return {
            "id": row["id"],
            "code_embeddings": torch.tensor(code_embs, dtype=torch.float32),
            "markdown_embeddings": torch.tensor(md_embs, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "markdown_ids": md_ids,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable numbers of cells per notebook.
    Pads sequences to the maximum length in the batch.
    """
    # Determine max lengths in this batch
    max_code_len = max(item["code_embeddings"].size(0) for item in batch)
    max_md_len = max(item["markdown_embeddings"].size(0) for item in batch)

    # Get embedding dimension
    input_dim = batch[0]["code_embeddings"].size(1) if batch else 0

    # Containers
    code_batch = []
    md_batch = []
    labels_batch = []
    ids = []
    markdown_ids_batch = []

    for item in batch:
        c_emb = item["code_embeddings"]
        m_emb = item["markdown_embeddings"]
        lbl = item["labels"]

        # Pad Code Embeddings (Anchors)
        # Pad with zeros
        c_len = c_emb.size(0)
        pad_c = max_code_len - c_len
        if pad_c > 0:
            c_padded = torch.cat([c_emb, torch.zeros(pad_c, input_dim)], dim=0)
        else:
            c_padded = c_emb

        # Pad Markdown Embeddings (Queries)
        # Pad with zeros
        m_len = m_emb.size(0)
        pad_m = max_md_len - m_len
        if pad_m > 0:
            m_padded = torch.cat([m_emb, torch.zeros(pad_m, input_dim)], dim=0)
            # Pad labels with -100 (ignore index)
            l_padded = torch.cat(
                [lbl, torch.full((pad_m,), -100, dtype=torch.long)], dim=0
            )
        else:
            m_padded = m_emb
            l_padded = lbl

        code_batch.append(c_padded)
        md_batch.append(m_padded)
        labels_batch.append(l_padded)
        ids.append(item["id"])
        markdown_ids_batch.append(item["markdown_ids"])

    return {
        "code_embeddings": torch.stack(code_batch),  # (Batch, Max_Code, Dim)
        "markdown_embeddings": torch.stack(md_batch),  # (Batch, Max_MD, Dim)
        "labels": torch.stack(labels_batch),  # (Batch, Max_MD)
        "ids": ids,
        "markdown_ids": markdown_ids_batch,
    }
