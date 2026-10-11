import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_PAIR_TOKENS, SEQ_LEN, create_dataloader


class PairPositionEmbedding(nn.Module):
    """
    Pair token embedding + learnable position embedding.

    Input:
        pair_ids: LongTensor, shape [B, 20]
    Output:
        Tensor, shape [B, 20, pair_embed_dim]
    """

    def __init__(
        self,
        num_pair_tokens: int = 16,
        seq_len: int = SEQ_LEN,
        pair_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pair_embedding = nn.Embedding(num_embeddings=num_pair_tokens, embedding_dim=pair_embed_dim)
        self.position_embedding = nn.Embedding(num_embeddings=seq_len, embedding_dim=pair_embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self, pair_ids: torch.Tensor):
        bsz, seq_len = pair_ids.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {seq_len}")

        pos_ids = torch.arange(self.seq_len, device=pair_ids.device).unsqueeze(0).expand(bsz, -1)

        pair_emb = self.pair_embedding(pair_ids)      # [B, 20, 64]
        pos_emb = self.position_embedding(pos_ids)    # [B, 20, 64]
        x = pair_emb + pos_emb                        # [B, 20, 64]
        x = self.embedding_dropout(x)
        return x, pair_emb, pos_emb


class MultiKernelCNN(nn.Module):
    """
    Multi-kernel 1D CNN block with kernel sizes 3/5/7.

    Input:
        x: Tensor, shape [B, 20, in_channels]
    Output:
        x_out: Tensor, shape [B, 20, out_channels*3]
        debug dict with intermediate tensors
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3)
        self.dropout = nn.Dropout(dropout)

    def _branch(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        y = conv(x)
        y = F.gelu(y)
        y = self.dropout(y)
        return y

    def forward(self, x: torch.Tensor):
        x_cnn = x.transpose(1, 2)  # [B, C, L] = [B, 64, 20]

        b3 = self._branch(self.conv3, x_cnn)  # [B, 64, 20]
        b5 = self._branch(self.conv5, x_cnn)  # [B, 64, 20]
        b7 = self._branch(self.conv7, x_cnn)  # [B, 64, 20]

        cat = torch.cat([b3, b5, b7], dim=1)  # [B, 192, 20]
        out = cat.transpose(1, 2)              # [B, 20, 192]

        debug = {
            "cnn_input": x_cnn,
            "cnn_k3": b3,
            "cnn_k5": b5,
            "cnn_k7": b7,
            "cnn_concat": cat,
            "cnn_transpose": out,
        }
        return out, debug


class OffTargetModel(nn.Module):
    """
    PAM-agnostic 20nt off-target active/inactive binary classifier.

    Input:
        pair_ids: LongTensor, shape [B, 20]
    Output:
        logits: Tensor, shape [B]
    """

    def __init__(
        self,
        num_pair_tokens: int = 16,
        seq_len: int = SEQ_LEN,
        pair_embed_dim: int = 64,
        cnn_channels: int = 64,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.embedding = PairPositionEmbedding(
            num_pair_tokens=num_pair_tokens,
            seq_len=seq_len,
            pair_embed_dim=pair_embed_dim,
            dropout=dropout,
        )

        self.multi_cnn = MultiKernelCNN(
            in_channels=pair_embed_dim,
            out_channels=cnn_channels,
            dropout=dropout,
        )

        self.proj = nn.Linear(cnn_channels * 3, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_dropout = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, pair_ids: torch.Tensor) -> torch.Tensor:
        # 1-3) pair embedding + position embedding
        x, _, _ = self.embedding(pair_ids)  # [B, 20, 64]

        # 4-6) multi-kernel CNN + concat + projection
        x, _ = self.multi_cnn(x)            # [B, 20, 192]
        x = self.proj(x)                    # [B, 20, 128]
        x = self.proj_norm(x)
        x = F.gelu(x)
        x = self.proj_dropout(x)

        # 7-8) self-attention + residual + layernorm
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)  # [B, 20, 128]
        x = self.attn_norm(x + attn_out)                           # [B, 20, 128]

        # 9) mean pooling
        x = x.mean(dim=1)  # [B, 128]

        # 10-11) MLP classifier -> logits
        logits = self.classifier(x).squeeze(-1)  # [B]
        return logits

    def forward_with_shapes(self, pair_ids: torch.Tensor) -> torch.Tensor:
        print("Input pair_ids:", pair_ids.shape)

        x, pair_emb, pos_emb = self.embedding(pair_ids)
        print("Pair embedding:", pair_emb.shape)
        print("Position embedding:", pos_emb.shape)
        print("Embedding output:", x.shape)

        x_cnn_out, dbg = self.multi_cnn(x)
        print("CNN input:", dbg["cnn_input"].shape)
        print("CNN kernel3:", dbg["cnn_k3"].shape)
        print("CNN kernel5:", dbg["cnn_k5"].shape)
        print("CNN kernel7:", dbg["cnn_k7"].shape)
        print("CNN concat:", dbg["cnn_concat"].shape)
        print("CNN transpose:", dbg["cnn_transpose"].shape)

        x = self.proj(x_cnn_out)
        x = self.proj_norm(x)
        x = F.gelu(x)
        x = self.proj_dropout(x)
        print("Projection:", x.shape)

        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        print("Attention output:", attn_out.shape)

        x = self.attn_norm(x + attn_out)
        print("After residual:", x.shape)

        x = x.mean(dim=1)
        print("Mean pooled:", x.shape)

        logits = self.classifier(x).squeeze(-1)
        print("Logits:", logits.shape)
        return logits


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optional model self-test; not used by the inference pipeline.")
    parser.add_argument("--input", required=True, help="User-supplied labelled CSV for the self-test.")
    train_csv = parser.parse_args().input

    # Test 1: dummy input
    print("=== Dummy Input Test ===")
    model = OffTargetModel(num_pair_tokens=NUM_PAIR_TOKENS)
    dummy_pair_ids = torch.randint(0, NUM_PAIR_TOKENS, size=(4, SEQ_LEN))
    dummy_logits = model.forward_with_shapes(dummy_pair_ids)
    print("Dummy logits shape:", dummy_logits.shape)

    # Test 2: real dataset batch
    print("\n=== Real Batch Test ===")
    loader = create_dataloader(
        csv_path=train_csv,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    pair_ids = batch["pair_ids"]
    logits = model(pair_ids)

    print("Real batch pair_ids:", pair_ids.shape)
    print("Real batch logits:", logits.shape)
    print("Logits all finite:", torch.isfinite(logits).all().item())

    total_params, trainable_params = count_parameters(model)
    print("\nTotal parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    if dummy_logits.shape != torch.Size([4]):
        raise RuntimeError(f"Expected dummy logits shape [4], got {dummy_logits.shape}")
    if logits.shape != torch.Size([32]):
        raise RuntimeError(f"Expected real batch logits shape [32], got {logits.shape}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Found non-finite values in logits.")

    print("Model test PASS.")
