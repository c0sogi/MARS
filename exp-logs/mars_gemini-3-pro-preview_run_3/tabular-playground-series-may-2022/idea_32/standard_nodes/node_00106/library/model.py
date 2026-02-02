import torch
import torch.nn as nn


class SingleStream(nn.Module):
    """
    A single stream of the Heterogeneous Parallel Funnel Ensemble (HPFE).

    Removed Batch Normalization (Cite solution_lesson_node_00091) and
    Deep Supervision (Cite solution_lesson_node_00105) to improve generalization.
    """

    def __init__(self, vocab_sizes, num_cont, hidden_dims, dropout_rate, embed_dim):
        super(SingleStream, self).__init__()

        # Independent embeddings for this stream to ensure decorrelated representations
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Input dimension: Continuous features + Flattened embeddings
        input_dim = num_cont + (len(vocab_sizes) * embed_dim)

        # Layer 1 (Wide)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Layer 2 (Compressing)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        # Layer 3 (Bottleneck)
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rate)

        # Main Head attached to the final layer
        self.main_head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x_cat, x_cont):
        # 1. Embedding Lookup
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)

        # 2. Early Fusion
        x = torch.cat([x_cont, x_emb], dim=1)

        # 3. Layer 1 Processing
        x = self.drop1(self.act1(self.fc1(x)))

        # 4. Subsequent Layers
        x = self.drop2(self.act2(self.fc2(x)))
        x = self.drop3(self.act3(self.fc3(x)))

        # 5. Main Prediction
        main_out = self.main_head(x)

        return main_out


class DSPFE(nn.Module):
    """
    Heterogeneous Parallel Funnel Ensemble (HPFE).
    Renamed class kept as DSPFE for compatibility with existing imports,
    but logic reflects HPFE.
    """

    def __init__(self, vocab_sizes, num_cont, stream_configs, embed_dim):
        super(DSPFE, self).__init__()

        self.streams = nn.ModuleList()
        for config in stream_configs:
            stream = SingleStream(
                vocab_sizes=vocab_sizes,
                num_cont=num_cont,
                hidden_dims=config["hidden_dims"],
                dropout_rate=config["dropout"],
                embed_dim=embed_dim,
            )
            self.streams.append(stream)

    def forward(self, x_cat, x_cont):
        main_outputs = []

        # Forward pass through all parallel streams
        for stream in self.streams:
            m_out = stream(x_cat, x_cont)
            main_outputs.append(m_out)

        return main_outputs
