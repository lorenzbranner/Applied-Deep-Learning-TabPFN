"""Mini-PFN: a small version of the TabPFN architecture (paper Fig. 1b).

Input:  a whole table. Labelled training rows plus unlabelled test rows.
Output: class probabilities for the test rows, in ONE forward pass.

Like in the paper, every cell of the table is one token. Each layer does
  1) attention across the cells of a row        ("1D feature attention")
  2) attention across the rows of a column      ("1D sample attention")
  3) an MLP on every cell
each followed by a residual connection and a layer norm.

No weight depends on the number of columns: the same encoder is used for every
feature cell, and attention works on sequences of any length. So the model
accepts tables with more columns than it saw in pre-training (paper, Methods:
"it can generalize to more training samples as well as features than seen
during training").

Simplifications versus the paper: 3 layers instead of 12, embedding size 64
instead of 192, two classes, no feature grouping, no random feature
embeddings, no ensembling over preprocessings, no train-state caching.
"""
import torch
import torch.nn as nn


class TwoWayLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.feature_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.sample_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, h, n_train):
        # h: (batch, rows, columns, d_model). Rows = train rows then test rows.
        # Columns = the features, then the target column.
        B, R, C, D = h.shape

        # 1) Feature attention: every row is a sequence of its C cells.
        s = h.reshape(B * R, C, D)
        update, _ = self.feature_attention(s, s, s, need_weights=False)
        s = self.norm1(s + update)
        h = s.reshape(B, R, C, D)

        # 2) Sample attention: every column is a sequence of its R cells.
        #    Keys and values are the TRAIN rows only, so test rows can read the
        #    training data but never each other (paper, Methods).
        s = h.transpose(1, 2).reshape(B * C, R, D)
        train = s[:, :n_train]
        update, _ = self.sample_attention(s, train, train, need_weights=False)
        s = self.norm2(s + update)
        h = s.reshape(B, C, R, D).transpose(1, 2)

        # 3) MLP on every cell.
        return self.norm3(h + self.mlp(h))


class MiniPFN(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=3, d_ff=128):
        super().__init__()
        self.config = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        self.x_encoder = nn.Linear(1, d_model)   # one feature value -> one token
        self.y_encoder = nn.Linear(2, d_model)   # (label, is_unknown) -> one token
        self.layers = nn.ModuleList(
            [TwoWayLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, 2))

    def forward(self, x, y_train):
        """x: (B, R, F) all rows, the first n_train of them are labelled.
        y_train: (B, n_train) labels in {0, 1}.
        Returns logits for the unlabelled rows: (B, R - n_train, 2)."""
        B, R, F = x.shape
        n_train = y_train.shape[1]

        # z-normalise every feature with the statistics of the training rows.
        mean = x[:, :n_train].mean(1, keepdim=True)
        std = x[:, :n_train].std(1, keepdim=True) + 1e-6
        x = (x - mean) / std

        # Target column: the label for train rows, an "unknown" flag for test rows.
        y = torch.zeros(B, R, 2)
        y[:, :n_train, 0] = y_train.float() * 2 - 1    # classes 0/1 -> -1/+1
        y[:, n_train:, 1] = 1.0

        # One token per cell: (B, R, F + 1, d_model).
        h = torch.cat([self.x_encoder(x.unsqueeze(-1)),
                       self.y_encoder(y).unsqueeze(2)], dim=2)
        for layer in self.layers:
            h = layer(h, n_train)

        # Read the prediction from the target cell of every test row.
        return self.decoder(h[:, n_train:, -1])

    @torch.no_grad()
    def predict_proba(self, x_train, y_train, x_test):
        """NumPy in, NumPy out. Returns P(class 1) for every test row.
        This is the whole "fit + predict": no weight is updated."""
        self.eval()
        x = torch.as_tensor(x_train, dtype=torch.float32)
        x = torch.cat([x, torch.as_tensor(x_test, dtype=torch.float32)])[None]
        y = torch.as_tensor(y_train, dtype=torch.long)[None]
        return self(x, y).softmax(-1)[0, :, 1].numpy()


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = MiniPFN(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.tables_seen = checkpoint["step"] * checkpoint["batch_size"]
    return model.eval()
