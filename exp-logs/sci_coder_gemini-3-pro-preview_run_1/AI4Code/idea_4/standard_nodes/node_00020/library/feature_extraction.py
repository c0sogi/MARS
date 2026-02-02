import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class TextDataset(Dataset):
    """
    Dataset class to hold source text for cells.
    """

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx])


class SmartCollator:
    """
    Collator to handle dynamic padding and tokenization.
    """

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch_texts):
        return self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )


class FeatureExtractor:
    """
    Extracts [CLS] embeddings from notebook cells using a pre-trained CodeBERT model.
    """

    def __init__(self):
        self.device = Config.DEVICE
        print(
            f"Initializing FeatureExtractor with {Config.MODEL_NAME} on {self.device}..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def extract_and_save_features(self, load_cached_data=True):
        """
        Main method to process train, val, and test splits.
        """
        set_seed(Config.SEED)

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Process Train
        self._process_split(
            metadata_path=Config.TRAIN_METADATA_PATH,
            output_path=Config.TRAIN_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_labeled=True,
        )

        # Process Validation
        self._process_split(
            metadata_path=Config.VAL_METADATA_PATH,
            output_path=Config.VAL_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_labeled=True,
        )

        # Process Test
        self._process_split(
            metadata_path=Config.TEST_METADATA_PATH,
            output_path=Config.TEST_FEATURES_PATH,
            load_cached_data=load_cached_data,
            is_labeled=False,
        )

    def _process_split(self, metadata_path, output_path, load_cached_data, is_labeled):
        """
        Internal method to process a specific dataset split.
        """
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features from {output_path}")
            return

        print(f"Processing metadata from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        if Config.DEBUG:
            print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks.")
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        # Containers for data
        texts = []
        meta_records = []

        # 1. Parse Notebooks
        # We iterate sequentially; JSON load is fast, bottleneck is usually Model Inference.
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = os.path.join(Config.INPUT_DIR, row["filepath"])
            ancestor_id = row.get(
                "ancestor_id", nb_id
            )  # Default to self if missing (e.g. test)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    nb_json = json.load(f)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine Ranks
            rank_map = {}
            if is_labeled and "cell_order" in row and pd.notna(row["cell_order"]):
                cell_order = row["cell_order"].split()
                rank_map = {cid: i for i, cid in enumerate(cell_order)}

            # Iterate over all cells in the notebook
            for cell_id, c_type in cell_types.items():
                source_text = sources.get(cell_id, "")
                rank = rank_map.get(cell_id, -1)  # -1 for test or unlisted cells

                texts.append(source_text)
                meta_records.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "rank": rank,
                        "ancestor_id": ancestor_id,
                    }
                )

        if not texts:
            print(f"Warning: No data found for {metadata_path}")
            return

        # 2. Inference
        print(f"Extracting embeddings for {len(texts)} cells...")
        dataset = TextDataset(texts)
        collator = SmartCollator(self.tokenizer, Config.MAX_LEN)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collator,
            pin_memory=True,
        )

        embeddings_list = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Cite solution_lesson_node_00017: Use Mean Pooling for Sentence Transformers
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                embeddings_list.append(embeddings.cpu().numpy())

        # Concatenate all batches
        full_embeddings = np.vstack(embeddings_list)

        # 3. Save to Parquet
        print("Constructing DataFrame...")
        df_out = pd.DataFrame(meta_records)

        # Assign embeddings as a list of floats for Parquet compatibility
        # (PyArrow handles list columns efficiently)
        df_out["embedding"] = list(full_embeddings)

        print(f"Saving features to {output_path}...")
        df_out.to_parquet(output_path, index=False)
        print("Done.")
