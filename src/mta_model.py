"""
Multi-Token Attention (MTA) Model for Taiwan Stock Index Futures
================================================================
Reconstructed from:
  "Application of Deep Learning in Taiwan Stock Index Futures Trading Strategies"
  Hsu, Hao — National Yang Ming Chiao Tung University, July 2025

Architecture (Section 3.3.1):
  1. Input Projection       — linear(F → d_model)
  2. Positional Encoding    — sinusoidal (Vaswani et al., 2017)
  3. MTA Encoder Block      — Multi-Token Attention + FFN + LayerNorm
     a. QKV projection
     b. Scaled dot-product attention
     c. Key-Query 2-D local convolution (per head)
     d. Cross-head 1-D mixing convolution + GroupNorm
     e. Weighted sum with V, linear output projection
  4. Attention Pooling      — 2-layer MLP scores each time-step
  5. Classification Head    — linear(d_model → num_classes)

Loss functions (Section 3.4):
  - CrossEntropyLoss  (CE)
  - FocalLoss         (gamma = 2)
  - MADLFocalLoss     (direction + return weighting + focal term)

StockDataManager Integration
-----------------------------
  MTAStockModel is a drop-in companion to TransferLearningModel.
  It shares the same save/load/train/predict/evaluate interface so it
  can be plugged into StockDataManager with minimal changes.
"""

import math
import os
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1.  Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in Vaswani et al. (2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2.  Multi-Token Attention layer
# ---------------------------------------------------------------------------

class MultiTokenAttention(nn.Module):
    """
    Multi-Token Attention (MTA) - Golovneva et al. (2025).

    Enhancements over standard multi-head self-attention:
      (a) Key-Query 2-D convolution  - captures local temporal patterns per head
      (b) Cross-head 1-D convolution - mixes information across heads
          followed by GroupNorm (groups = num_heads)
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        conv_kernel: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Single linear layer -> Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # (a) Per-head 2-D convolution on the L x L attention score matrix
        padding = conv_kernel // 2
        self.attn_conv = nn.Conv2d(
            in_channels=num_heads,
            out_channels=num_heads,
            kernel_size=(conv_kernel, conv_kernel),
            padding=(padding, padding),
            groups=num_heads,
            bias=False,
        )

        # (b) Cross-head 1-D mixing convolution
        self.head_mix_conv = nn.Conv1d(
            in_channels=num_heads,
            out_channels=num_heads,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.head_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape
        H, d_k = self.num_heads, self.d_k

        # QKV projection
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, L, H, d_k).transpose(1, 2)  # (B, H, L, d_k)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, L, L)

        # (a) Per-head 2-D convolution
        scores = self.attn_conv(scores)  # (B, H, L, L)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (b) Cross-head 1-D mixing convolution
        aw = attn_weights.permute(0, 2, 1, 3)   # (B, L, H, L)
        aw = aw.reshape(B * L, H, L)
        aw = self.head_mix_conv(aw)
        aw = self.head_norm(aw)
        aw = aw.reshape(B, L, H, L).permute(0, 2, 1, 3)  # (B, H, L, L)

        # Weighted sum with V
        out = torch.matmul(aw, V)                          # (B, H, L, d_k)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        out = self.out_proj(out)
        return self.out_dropout(out)


# ---------------------------------------------------------------------------
# 3.  MTA Encoder Block
# ---------------------------------------------------------------------------

class MTAEncoderBlock(nn.Module):
    """MTA + Feed-Forward + Add & LayerNorm (x2)."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 256,
        conv_kernel: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mta = MultiTokenAttention(d_model, num_heads, conv_kernel, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.mta(x))
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# 4.  Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Learnable time-step weighting:
        alpha_t = softmax( W2 . tanh(W1 . x_t) )
        z       = sum_t  alpha_t . x_t
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model // 2)
        self.W2 = nn.Linear(d_model // 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, d_model)"""
        weights = F.softmax(self.W2(torch.tanh(self.W1(x))), dim=1)  # (B, L, 1)
        return (weights * x).sum(dim=1)


# ---------------------------------------------------------------------------
# 5.  Core MTA Torch Module (classification)
# ---------------------------------------------------------------------------

class _MTAModule(nn.Module):
    """
    Pure PyTorch classification module.
    Separated from the sklearn-style wrapper so it can be saved/loaded cleanly.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 256,
        conv_kernel: int = 9,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=256, dropout=dropout)
        self.encoder = nn.ModuleList([
            MTAEncoderBlock(d_model, num_heads, ffn_dim, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        self.pool = AttentionPooling(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) -> logits (B, num_classes)"""
        x = self.pos_enc(self.input_proj(x))
        for block in self.encoder:
            x = block(x)
        return self.classifier(self.pool(x))


# ---------------------------------------------------------------------------
# 6.  Loss Functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """FL(p_t) = -(1-p_t)^gamma . log(p_t)   (Lin et al., 2017, gamma=2)"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-(1 - pt) ** self.gamma * log_pt).mean()


class MADLFocalLoss(nn.Module):
    """
    MADL-Focal Loss (thesis Section 3.4.3):
        Loss = CE x (1 + |r| . (-sign(r) x dir_score)) x (1-p_t)^gamma
    """

    def __init__(self, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,   # (B, C)
        targets: torch.Tensor,  # (B,)  0=bear, 1=sideways, 2=bull
        returns: torch.Tensor,  # (B,)  actual price return
    ) -> torch.Tensor:
        B = logits.size(0)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp(min=self.eps))
        ce = F.nll_loss(log_probs, targets, reduction="none")

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** self.gamma

        pred_class = probs.argmax(dim=-1)
        sign_r = torch.sign(returns)
        pred_dir = torch.where(
            pred_class == 2, torch.ones(B, device=logits.device),
            torch.where(pred_class == 0, -torch.ones(B, device=logits.device),
                        torch.zeros(B, device=logits.device))
        )
        dir_score = sign_r * pred_dir
        madl_w = 1.0 + returns.abs() * (-sign_r * dir_score)

        return (ce * madl_w * focal_w).mean()


# ---------------------------------------------------------------------------
# 7.  MTAStockModel - drop-in companion for TransferLearningModel
# ---------------------------------------------------------------------------

class MTAStockModel:
    """
    Sklearn-style wrapper around _MTAModule that mirrors the
    TransferLearningModel interface used by StockDataManager:

        .train(X, y, ...)
        .predict(X)
        .evaluate(X, y)
        .save(filepath)
        .load(filepath)
        .copy()
        .exists(filepath)   [static]

    Key differences from TransferLearningModel
    -------------------------------------------
    * Uses PyTorch instead of TensorFlow/Keras.
    * Treats the problem as 3-class classification
      (0 = bear / down, 1 = sideways, 2 = bull / up)
      derived from the sign of the next-day return.
    * loss_fn can be 'ce', 'focal', or 'madl_focal'.
    * seq_length and input_dim must match TransferLearningModel
      so _prepare_training_data works unchanged.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        input_dim: int = 14,         # must match len(feature_map) in utility.py
        seq_length: int = 30,        # must match TransferLearningModel default
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 256,
        conv_kernel: int = 9,
        dropout: float = 0.3,
        num_classes: int = 3,
        learning_rate: float = 0.001,
        loss_fn: str = "ce",         # 'ce' | 'focal' | 'madl_focal'
    ):
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.conv_kernel = conv_kernel
        self.dropout = dropout
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn.lower()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self._model = self._build_model()

    def _build_model(self) -> _MTAModule:
        return _MTAModule(
            n_features=self.input_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ffn_dim=self.ffn_dim,
            conv_kernel=self.conv_kernel,
            dropout=self.dropout,
            num_classes=self.num_classes,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale 3-D array (B, L, F) using StandardScaler on the feature axis."""
        B, L, F = X.shape
        X2d = X.reshape(-1, F)
        X2d = self.scaler.fit_transform(X2d) if fit else self.scaler.transform(X2d)
        return X2d.reshape(B, L, F)

    def update_scaler(self, X: np.ndarray):
        """Incrementally updates the scaler's mean and std using partial_fit."""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 3:
            self.scaler.partial_fit(X.reshape(-1, X.shape[2]))
        else:
            self.scaler.partial_fit(X)

    @staticmethod
    def _return_to_label(y: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Convert continuous next-day returns to 3-class labels.
            y > +threshold  -> 2  (bull)
            y < -threshold  -> 0  (bear)
            else            -> 1  (sideways)
        """
        labels = np.ones(len(y), dtype=np.int64)   # default: sideways
        labels[y >  threshold] = 2
        labels[y < -threshold] = 0
        return labels

    def _get_loss_fn(self):
        if self.loss_fn == "focal":
            return FocalLoss(gamma=2.0)
        if self.loss_fn == "madl_focal":
            return MADLFocalLoss(gamma=2.0)
        return nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Public API (mirrors TransferLearningModel)
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        fit_scaler: bool = False,
        patience: int = 5,
        verbose: int = 1,
    ):
        """
        Train the MTA model.

        Parameters
        ----------
        X          : (N, seq_length, input_dim)  feature sequences
        y          : (N,)  continuous next-day returns (same as TransferLearningModel)
        fit_scaler : True only for Stage 1 (first time scaler is fitted)
        patience   : early-stopping patience (0 = disabled)
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        X = self._scale(X, fit=fit_scaler).astype(np.float32)
        labels = self._return_to_label(y)

        X_t     = torch.as_tensor(X)
        y_t     = torch.as_tensor(labels, dtype=torch.long)
        y_ret   = torch.as_tensor(y,      dtype=torch.float32)  # raw returns for MADL

        dataset = torch.utils.data.TensorDataset(X_t, y_t, y_ret)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate,
            steps_per_epoch=len(loader), epochs=epochs
        )
        loss_fn = self._get_loss_fn()

        best_loss = float("inf")
        patience_counter = 0

        self._model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch, ret_batch in loader:
                X_batch   = X_batch.to(self.device)
                y_batch   = y_batch.to(self.device)
                ret_batch = ret_batch.to(self.device)

                optimizer.zero_grad()
                logits = self._model(X_batch)

                if self.loss_fn == "madl_focal":
                    loss = loss_fn(logits, y_batch, ret_batch)
                else:
                    loss = loss_fn(logits, y_batch)

                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if verbose > 0:
                print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

            # Early stopping
            if patience > 0:
                if avg_loss < best_loss - 1e-5:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose > 0:
                            print(f"  Early stopping at epoch {epoch}.")
                        break

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Returns a pseudo-return signal compatible with TransferLearningModel output.

        The model outputs class probabilities (bear / sideways / bull).
        We convert to a signed score so predict_day() in StockDataManager works:
            signal = P(bull) - P(bear)   in (-1, +1)

        Shape returned: (N, 1)  matching TF model output.
        """
        probs = self.predict_proba(X, batch_size=batch_size)
        pseudo_return = probs[:, 2] - probs[:, 0]   # bull - bear
        return pseudo_return.reshape(-1, 1)

    def predict_proba(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Returns raw class probabilities (N, 3): [bear, sideways, bull]."""
        X = np.array(X, dtype=np.float32)
        X = self._scale(X, fit=False)
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)

        self._model.eval()
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                X_batch = X_t[i : i + batch_size]
                probs = F.softmax(self._model(X_batch), dim=-1).cpu().numpy()
                all_probs.append(probs)
                
        return np.concatenate(all_probs, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns Mean Absolute Error between predicted signal and actual return.
        Mirrors TransferLearningModel.evaluate() which also returns MAE.
        """
        preds = self.predict(np.array(X, dtype=np.float32)).flatten()
        return float(np.mean(np.abs(preds - np.array(y, dtype=np.float32))))

    def save(self, filepath: str):
        """
        Saves model weights to <filepath>.pt (auto-replaces .keras extension).
        Saves scaler to <filepath>.pt.scaler
        """
        filepath = str(filepath)
        if filepath.endswith(".keras"):
            filepath = filepath.replace(".keras", ".pt")

        state = {
            "model_state": self._model.state_dict(),
            "config": {
                "input_dim":    self.input_dim,
                "seq_length":   self.seq_length,
                "d_model":      self.d_model,
                "num_heads":    self.num_heads,
                "num_layers":   self.num_layers,
                "ffn_dim":      self.ffn_dim,
                "conv_kernel":  self.conv_kernel,
                "dropout":      self.dropout,
                "num_classes":  self.num_classes,
                "learning_rate":self.learning_rate,
                "loss_fn":      self.loss_fn,
            },
        }
        torch.save(state, filepath)

        with open(filepath + ".scaler", "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"MTAStockModel saved to {filepath}")

    def load(self, filepath: str):
        """Loads model weights and scaler from disk."""
        filepath = str(filepath)
        if filepath.endswith(".keras"):
            filepath = filepath.replace(".keras", ".pt")

        state = torch.load(filepath, map_location=self.device)
        for k, v in state["config"].items():
            setattr(self, k, v)

        self._model = self._build_model()
        self._model.load_state_dict(state["model_state"])

        scaler_path = filepath + ".scaler"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            print(f"Warning: Scaler not found at {scaler_path}.")

    def copy(self) -> "MTAStockModel":
        """Deep-copies the model (used for Stage 1 -> 2 -> 3 transfer learning)."""
        new = MTAStockModel(
            input_dim=self.input_dim,    seq_length=self.seq_length,
            d_model=self.d_model,        num_heads=self.num_heads,
            num_layers=self.num_layers,  ffn_dim=self.ffn_dim,
            conv_kernel=self.conv_kernel,dropout=self.dropout,
            num_classes=self.num_classes,learning_rate=self.learning_rate,
            loss_fn=self.loss_fn,
        )
        new._model.load_state_dict(copy.deepcopy(self._model.state_dict()))
        new.scaler = copy.deepcopy(self.scaler)
        return new

    @staticmethod
    def exists(filepath: str) -> bool:
        """Mirrors TransferLearningModel.exists() — checks for .pt and .pt.scaler."""
        filepath = str(filepath)
        if filepath.endswith(".keras"):
            filepath = filepath.replace(".keras", ".pt")
        return os.path.exists(filepath) and os.path.exists(filepath + ".scaler")
