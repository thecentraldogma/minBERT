#minGPT

import torch
from sa import TokenPositionalEmbedding, TransformerBlock, IntegerSequenceLoopDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tiktoken
import random


# basic setup is similar to minBERT
# the main differences are: 
# 1. Collate function does not do MLM anymore. Instead we set the labels to be the same as the input sequence except shifted right by one position. Last position in labels is set to -100. 
# 2. We now use a causal attention mask because we want to predict the next token, so given an input sequence token i can only attend to prior N tokens. 
# 3. We need a generate function, that takes in a given sequence and predicts the next token
# 4. The test harness can be similar to that of BERT: i.e. the dataset class generates sequences of integers from 0 to vocab_size-1 and wraps around if needed, upto a max of seq_len tokens. 


def gpt_collate_fn(input_ids):
	input_ids = torch.stack(input_ids, dim = 0) # (B, T) # collecting tensors into a batch is the main job of the default collate fn
	labels = torch.full_like(input_ids, -100)
	labels[:, :-1] = input_ids[:, 1:]
	return input_ids, labels




class MinGPT(nn.Module):
    def __init__(self,
        vocab_size: int,
        d_model: int,
        n_heads: int,	
        d_ff: int,
        n_layers: int,
        seq_len: int,
        emb_p_drop: float = 0.1,
        trx_p_drop: float = 0.1): # used in transformer in the self attention and ff parts
        super().__init__()
        self.embed = TokenPositionalEmbedding(vocab_size = vocab_size, d_model = d_model, max_len = seq_len)
        self.dropout = nn.Dropout(emb_p_drop)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads=n_heads, seq_len=seq_len, d_model=d_model, sa_p_drop=trx_p_drop, ff_dim=d_ff, ff_p_drop=trx_p_drop, causal=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) # this converts to vocab_size logits. (B, T, d_model) -> (B, T, vocab_size)
        self.seq_len = seq_len

    def forward(self, token_ids):
        # token_ids has shape (B, T) and contains indices from 0...vocab_size-1. 
        x = self.embed(token_ids)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)                                    # (B, T, d_model)
        x = self.norm(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)
        return logits



@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate new tokens from a trained autoregressive model.

    Args:
        model: trained GPT-like model
        idx: (B=1, T) tensor of token indices (the prompt)
        max_new_tokens: how many tokens to generate
        temperature: scaling factor for sampling randomness (>1 = more random, <1 = more greedy)
        top_k: if set, only keep top_k logits for sampling (nucleus/top-k sampling)

    Returns:
        (B=1, T + max_new_tokens) tensor of token indices
    """

    for i in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len:]
        out = model(idx_cond)
        new_logits = out[:, -1, :]/temperature
        probs = torch.softmax(new_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) 
        idx = torch.cat((idx, next_token), dim=1)
    return idx


    

def test_gpt_model(vocab_size, 
                    d_model, 
                    n_heads, 
                    d_ff, 
                    n_layers, 
                    emb_p_drop, 
                    trx_p_drop, 
                    B=64, 
                    T=16,
                    lr = 3e-4,
                    num_samples=1000,
                    n_steps=300,
                    device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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


    
    # Create model, optimizer, loss function, dataset, dataloader
    model = MinGPT(vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        seq_len=T,
        emb_p_drop=emb_p_drop,
        trx_p_drop=trx_p_drop)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index = -100)
    train_dataset = IntegerSequenceLoopDataset(T, vocab_size, num_samples)
    val_dataset = IntegerSequenceLoopDataset(T, vocab_size, num_samples)
    train_dataloader = DataLoader(dataset=train_dataset, 
                            batch_size=B, 
                            shuffle=True, 
                            drop_last=True,
                            collate_fn=lambda batch: gpt_collate_fn(batch)
                            )
    val_dataloader = DataLoader(dataset=val_dataset, 
                            batch_size=B, 
                            shuffle=True, 
                            drop_last=True,
                            collate_fn=lambda batch: gpt_collate_fn(batch)
                            )


    # train
    model.train()
    step = 0
    it = iter(train_dataloader)
    best_val = float("inf")
    while step < n_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dataloader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step % 50) == 0:
            val_loss = evaluate(val_dataloader)
            best_val = min(best_val, val_loss)
            print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")

        step += 1

    model.eval()
    with torch.no_grad():
        inp, tgt = next(iter(val_dataloader))
        inp = inp.to(device)
        logits = model(inp)                                       # (B,T,V)
        pred = logits.argmax(dim=-1).cpu()                        # (B,T)
        print("\nSample predictions (first 3 rows):")
        for i in range(min(10, inp.size(0))):
            print("inp :", inp[i].tolist())
            #print("tgt :", ([0] + inp[i].tolist()[:-1]))
            print("tgt :", tgt[i].tolist())
            print("pred:", pred[i].tolist())
            print("---")


    return model






def test_gpt_shakespeare(d_model, 
                    n_heads, 
                    d_ff, 
                    n_layers, 
                    emb_p_drop, 
                    trx_p_drop,
                    txt_filepath, 
                    B=64, 
                    T=16,
                    lr = 3e-4,
                    num_samples=1000,
                    n_steps=300,
                    device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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



    # read the shakespeare dataset once and split it to create ids for train and val
    enc = tiktoken.get_encoding("gpt2")   # byte-level BPE
    vocab_size = enc.n_vocab              # e.g., 50257

    def encode(s: str):  # str -> List[int]
        return enc.encode(s)

    def decode(ids):     # List[int] -> str
        return enc.decode(ids)

    text = open(txt_filepath, "r", encoding="utf-8").read()
    ids = encode(text)  # long list of ints
    split = int(0.8 * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_dataset = ShakespeareDataset(train_ids, T)
    val_dataset = ShakespeareDataset(val_ids, T)
    train_dataloader = DataLoader(dataset=train_dataset, 
                            batch_size=B, 
                            shuffle=True, 
                            drop_last=True,
                            )
    val_dataloader = DataLoader(dataset=val_dataset, 
                            batch_size=B, 
                            shuffle=True, 
                            drop_last=True,
                            )

    # Create model, optimizer, loss function, dataset, dataloader
    model = MinGPT(vocab_size=enc.n_vocab,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        seq_len=T,
        emb_p_drop=emb_p_drop,
        trx_p_drop=trx_p_drop)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index = -100)



    # Train and eval loop
    model.train()
    step = 0
    it = iter(train_dataloader)
    best_val = float("inf")
    while step < n_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dataloader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step % 50) == 0:
            val_loss = evaluate(val_dataloader)
            best_val = min(best_val, val_loss)
            print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")

        step += 1

    model.eval()
    with torch.no_grad():
        inp, tgt = next(iter(val_dataloader))
        inp = inp.to(device)
        logits = model(inp)                                       # (B,T,V)
        pred = logits.argmax(dim=-1).cpu()                        # (B,T)
        print("\nSample predictions (first 3 rows):")
        for i in range(min(10, inp.size(0))):
            print("inp :", inp[i].tolist())
            #print("tgt :", ([0] + inp[i].tolist()[:-1]))
            print("tgt :", tgt[i].tolist())
            print("pred:", pred[i].tolist())
            print("---")


    return model




class ShakespeareDataset(Dataset):

    def __init__(self, ids, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ids = torch.tensor(ids, dtype = torch.long)
        self.n = len(self.ids) - self.seq_len 

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # pick a random starting index i between 0 and self.n - 1, both inclusive
        x = self.ids[i:i+self.seq_len]
        y = self.ids[i+1:i+self.seq_len+1]
        return x, y

