import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DeepAveragingEncoder(nn.Module):
    """
    Encodes a sequence of tokens into a dense vector by averaging embeddings
    and passing them through a shallow MLP.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        embedding_matrix=None,
        freeze_embeddings=True,
    ):
        super(DeepAveragingEncoder, self).__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if embedding_matrix is not None:
            # Load pre-trained embeddings
            # embedding_matrix is expected to be a numpy array
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            print(f"Loaded embedding matrix with shape {embedding_matrix.shape}")

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # 2. MLP (Shallow)
        # Averaged vector -> Linear -> ReLU -> Linear -> ReLU
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_PROB),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len) - Token indices
        Returns: (batch_size, hidden_dim) - Encoded vectors
        """
        # Create mask for non-padding tokens (assuming 0 is PAD)
        mask = (x != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)

        # Get embeddings
        embeds = self.embedding(x)  # (batch, seq_len, emb_dim)

        # Sum embeddings ignoring padding
        sum_embeds = (embeds * mask).sum(dim=1)  # (batch, emb_dim)

        # Count non-padding tokens to compute average
        counts = mask.sum(dim=1)  # (batch, 1)
        # Avoid division by zero for empty sequences
        counts = torch.clamp(counts, min=1.0)

        # Compute Unweighted Average
        avg_embeds = sum_embeds / counts

        # Pass through MLP
        out = self.mlp(avg_embeds)

        return out


class SentenceFactorizedModel(nn.Module):
    """
    Sentence-Level Siamese Network.
    Ranks sentences by similarity to question and predicts Yes/No.
    """

    def __init__(self, vocab_size, embedding_matrix=None):
        super(SentenceFactorizedModel, self).__init__()

        # Shared Encoder
        self.encoder = DeepAveragingEncoder(
            vocab_size=vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            embedding_matrix=embedding_matrix,
            freeze_embeddings=Config.FREEZE_EMBEDDINGS,
        )

        # Yes/No Classifier Head
        # Input: Concatenation of Question Vector and Best Sentence Vector
        self.yn_classifier = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM * 2, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_PROB),
            nn.Linear(Config.HIDDEN_DIM, 3),  # Classes: NONE(0), YES(1), NO(2)
        )

    def forward(self, questions, sentences, doc_lengths):
        """
        Args:
            questions: (batch_size, max_q_len)
            sentences: (total_sentences, max_sent_len) - Flattened list of all sentences in batch
            doc_lengths: list/tensor of number of sentences per document in the batch

        Returns:
            scores: (total_sentences,) - Cosine similarity scores for each sentence
            yn_logits: (batch_size, 3) - Logits for Yes/No classification
        """

        # 1. Encode Questions
        q_vecs = self.encoder(questions)  # (batch_size, hidden_dim)

        # 2. Encode Sentences
        s_vecs = self.encoder(sentences)  # (total_sentences, hidden_dim)

        # 3. Align Questions with Sentences
        # Since 'sentences' is a flat list of all sentences from all docs,
        # we need to repeat the question vector for each sentence belonging to that question.
        device = q_vecs.device
        if not isinstance(doc_lengths, torch.Tensor):
            doc_lengths_tensor = torch.tensor(doc_lengths, device=device)
        else:
            doc_lengths_tensor = doc_lengths.to(device)

        q_vecs_expanded = torch.repeat_interleave(
            q_vecs, doc_lengths_tensor, dim=0
        )  # (total_sentences, hidden_dim)

        # 4. Interaction Layer: Cosine Similarity
        # Compute cosine similarity between Q and S vectors
        scores = F.cosine_similarity(
            q_vecs_expanded, s_vecs, dim=1
        )  # (total_sentences,)

        # 5. Yes/No Prediction Logic
        # We need to find the "best" sentence for each document to feed into the Y/N classifier.

        # Split the flattened sentence vectors and scores back into per-document groups
        # torch.split returns a tuple of tensors
        s_vecs_split = torch.split(s_vecs, doc_lengths)
        scores_split = torch.split(scores, doc_lengths)

        best_s_vecs_list = []

        for i in range(len(doc_lengths)):
            doc_s_vecs = s_vecs_split[i]
            doc_scores = scores_split[i]

            if len(doc_scores) > 0:
                # Find index of the sentence with the highest similarity score
                best_idx = torch.argmax(doc_scores)
                best_s_vec = doc_s_vecs[best_idx]
            else:
                # Fallback for empty documents
                best_s_vec = torch.zeros(Config.HIDDEN_DIM, device=device)

            best_s_vecs_list.append(best_s_vec)

        best_s_vecs = torch.stack(best_s_vecs_list)  # (batch_size, hidden_dim)

        # Concatenate Question and Best Sentence vectors
        yn_input = torch.cat(
            [q_vecs, best_s_vecs], dim=1
        )  # (batch_size, 2 * hidden_dim)

        # Classify
        yn_logits = self.yn_classifier(yn_input)  # (batch_size, 3)

        return scores, yn_logits
