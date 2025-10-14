import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class TinyTransformerLM(nn.Module):
    """
    Wraps:
      - token+pos embeddings -> (B,T,d_model)
      - N x TransformerBlock (your block)
      - linear vocab head
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        seq_len: int,
        emb_p_drop: float = 0.0,
        ff_p_drop: float = 0.1,
        block_ctor=None,   # pass your TransformerBlock class
    ):
        super().__init__()
        assert block_ctor is not None, "Pass your TransformerBlock class as block_ctor"
        self.embed = TokenPositionalEmbedding(vocab_size, d_model, max_len=seq_len)
        self.emb_drop = nn.Dropout(emb_p_drop)
        self.blocks = nn.ModuleList([
            block_ctor(n_heads=n_heads, seq_len=seq_len, d_model=d_model, sa_p_drop=ff_p_drop, ff_dim=d_ff, ff_p_drop=ff_p_drop)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):  # (B, T) -> logits (B, T, vocab)
        x = self.emb_drop(self.embed(token_ids))          # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x)                                    # (B, T, d_model)
        logits = self.lm_head(x)                          # (B, T, vocab)
        return logits


def run_copy_task(
    block_ctor,              #  TransformerBlock class
    n_steps=400,
    device=None,
    B=64, T=16,
    vocab_size=32,
    d_model=64,
    n_heads=4,
    d_ff=256,
    n_layers=2,
    lr=3e-4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_ds = CopyShiftDataset(num_samples=5000, T=T, vocab_size=vocab_size, seed=123)
    val_ds   = CopyShiftDataset(num_samples=500,  T=T, vocab_size=vocab_size, seed=999)
    train_dl = DataLoader(train_ds, batch_size=B, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=B, shuffle=False, drop_last=False)

    # model
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        seq_len=T,
        emb_p_drop=0.0,
        ff_p_drop=0.1,
        block_ctor=block_ctor,
    ).to(device)

    # opt & loss
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  # expects (B*T, vocab) vs (B*T,)

    def evaluate(dloader):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for inp, tgt in dloader:
                inp, tgt = inp.to(device), tgt.to(device)         # (B,T)
                logits = model(inp)                               # (B,T,V)
                loss = loss_fn(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                tot += loss.item() * inp.size(0)
                cnt += inp.size(0)
        model.train()
        return tot / cnt

    # train
    model.train()
    step = 0
    it = iter(train_dl)
    best_val = float("inf")
    while step < n_steps:
        try:
            inp, tgt = next(it)
        except StopIteration:
            it = iter(train_dl)
            inp, tgt = next(it)

        inp, tgt = inp.to(device), tgt.to(device)                 # (B,T)
        logits = model(inp)                                       # (B,T,V)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step % 50) == 0:
            val_loss = evaluate(val_dl)
            best_val = min(best_val, val_loss)
            print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")

        step += 1

    # quick qualitative check on a small batch
    model.eval()
    with torch.no_grad():
        inp, tgt = next(iter(val_dl))
        inp = inp.to(device)
        logits = model(inp)                                       # (B,T,V)
        pred = logits.argmax(dim=-1).cpu()                        # (B,T)
        print("\nSample predictions (first 3 rows):")
        for i in range(min(3, inp.size(0))):
            print("inp :", inp[i].tolist())
            print("tgt :", ([0] + inp[i].tolist()[:-1]))
            print("pred:", pred[i].tolist())
            print("---")