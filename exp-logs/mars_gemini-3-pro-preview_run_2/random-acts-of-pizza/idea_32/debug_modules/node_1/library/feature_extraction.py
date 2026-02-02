import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from library.config import Config
from library.data_loader import PizzaDataLoader


class FeatureExtractor:
    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = PizzaDataLoader()

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def extract_features(self, df, split, load_cached_data=True):
        """
        Extracts all required feature views for the given dataframe and split.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            split (str): The split name (train, val, test) for naming cache files.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: A dictionary containing numpy arrays for 'anchor', 'semantic_aux', and 'affective_aux'.
        """
        # Get text corpus from data loader (concatenates title + body)
        texts = self.data_loader.get_text_corpus(df)

        # Define cache paths with model names to avoid collisions if config changes
        anchor_name_safe = Config.ANCHOR_MODEL_NAME.replace("/", "-")
        semantic_aux_name_safe = Config.SEMANTIC_AUX_MODEL_NAME.replace("/", "-")
        affective_aux_name_safe = Config.AFFECTIVE_AUX_MODEL_NAME.replace("/", "-")

        cache_paths = {
            "anchor": os.path.join(
                self.working_dir, f"{split}_anchor_{anchor_name_safe}.npy"
            ),
            "semantic_aux": os.path.join(
                self.working_dir, f"{split}_semantic_aux_{semantic_aux_name_safe}.npy"
            ),
            "affective_aux": os.path.join(
                self.working_dir, f"{split}_affective_aux_{affective_aux_name_safe}.npy"
            ),
        }

        features = {}

        # 1. Semantic Anchor (MiniLM)
        features["anchor"] = self._get_or_compute(
            texts,
            cache_paths["anchor"],
            Config.ANCHOR_MODEL_NAME,
            method="embedding",
            load_cached_data=load_cached_data,
        )

        # 2. Semantic Aux (MPNet)
        features["semantic_aux"] = self._get_or_compute(
            texts,
            cache_paths["semantic_aux"],
            Config.SEMANTIC_AUX_MODEL_NAME,
            method="embedding",
            load_cached_data=load_cached_data,
        )

        # 3. Affective Aux (GoEmotions Logits)
        features["affective_aux"] = self._get_or_compute(
            texts,
            cache_paths["affective_aux"],
            Config.AFFECTIVE_AUX_MODEL_NAME,
            method="logits",
            load_cached_data=load_cached_data,
        )

        return features

    def _get_or_compute(self, texts, cache_path, model_name, method, load_cached_data):
        """
        Helper to handle caching logic: Load if exists and requested, else compute and save.
        """
        if load_cached_data and os.path.exists(cache_path):
            # print(f"Loading cached features from {cache_path}")
            return np.load(cache_path)

        print(f"Computing {method} features using {model_name}...")
        if method == "embedding":
            data = self._compute_embeddings(texts, model_name)
        elif method == "logits":
            data = self._compute_affective_logits(texts, model_name)
        else:
            raise ValueError(f"Unknown method: {method}")

        np.save(cache_path, data)
        print(f"Saved features to {cache_path}")
        return data

    def _compute_embeddings(self, texts, model_name, batch_size=32):
        """
        Computes mean-pooled embeddings for sentence-transformers compatible models.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        model.eval()

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                model_output = model(**encoded_input)

            # Mean Pooling
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input["attention_mask"]

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            batch_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _compute_affective_logits(self, texts, model_name, batch_size=32):
        """
        Computes raw logits for classification models (e.g., GoEmotions).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        model.eval()

        all_logits = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**encoded_input)

            # Extract logits
            batch_logits = outputs.logits
            all_logits.append(batch_logits.cpu().numpy())

        return np.vstack(all_logits)
