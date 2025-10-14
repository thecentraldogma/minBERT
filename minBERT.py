import torch
from sa import TokenPositionalEmbedding, TransformerBlock
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



def bert_collate_fn(input_ids, mask_id, mask_prob = 0.15):
    # input is a list of items returned by the _get_item_ method of the Dataset class, length of list is batch_size
    # output is tensor X containing enough elements for one batch, with the mask_id applied p% of the time. 
    
    input_ids = torch.stack(input_ids, dim = 0) # (B, T)
    labels = torch.full_like(input_ids, -100)

    # change the input_ids p% of the time to mask_id
    probs = torch.rand_like(input_ids.float())
    mask_positions = probs < mask_prob
    labels[mask_positions] = input_ids[mask_positions]
    input_ids[mask_positions] = mask_id
    attn_mask = torch.ones_like(input_ids)
    return input_ids, labels, attn_mask


class bert_test_dataset(Dataset):
    # this is a Dataset class that generates token id sequences
    def __init__(self, seq_len, vocab_size, num_samples): 
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        # return a tensor of length T that starts at a random integer between 0 and vocab_size-1 inclusize and wraps around if needed
        import random
        i = random.randrange(self.vocab_size)
        x = list(range(i, min(i + self.seq_len, self.vocab_size)))  + list(range(0, max(0, i + self.seq_len - self.vocab_size)))
        return torch.tensor(x)



class minBERT(nn.Module):
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
            TransformerBlock(n_heads=n_heads, seq_len=seq_len, d_model=d_model, sa_p_drop=trx_p_drop, ff_dim=d_ff, ff_p_drop=trx_p_drop)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) # this converts to vocab_size logits. (B, T, d_model) -> (B, T, vocab_size)

    def forward(self, token_ids):
        # token_ids has shape (B, T) and contains indices from 0...vocab_size-1. 
        x = self.embed(token_ids)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)                                    # (B, T, d_model)
        x = self.norm(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)
        return logits




def test_bert_model(vocab_size, 
                    d_model, 
                    n_heads, 
                    d_ff, 
                    n_layers, 
                    emb_p_drop, 
                    trx_p_drop, 
                    B=64, 
                    T=16,
                    mask_id = -200,
                    mask_prob = 0.2,
                    lr = 3e-4,
                    num_samples=1000,
                    n_steps=300,
                    device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(dloader):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for inp, tgt, am in dloader:
                inp, tgt = inp.to(device), tgt.to(device)         # (B,T)
                logits = model(inp)                               # (B,T,V)
                loss = loss_fn(logits.reshape(-1, vocab_size), tgt.reshape(-1))
                tot += loss.item() * inp.size(0)
                cnt += inp.size(0)
        model.train()
        return tot / cnt


    
    # Create model, optimizer, loss function, dataset, dataloader
    model = minBERT(vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        seq_len=T,
        emb_p_drop=emb_p_drop,
        trx_p_drop=trx_p_drop)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(ignore_index = -100)
    train_dataset = bert_test_dataset(T, vocab_size, num_samples)
    val_dataset = bert_test_dataset(T, vocab_size, num_samples)
    train_dataloader = DataLoader(dataset=train_dataset, 
                            batch_size=B, 
                            shuffle=True, 
                            drop_last=True,
                            collate_fn=lambda batch: bert_collate_fn(batch, mask_id=mask_id, mask_prob=mask_prob)
                            )
    val_dataloader = DataLoader(dataset=val_dataset, 
                            batch_size=B, 
                            shuffle=True, 
                            drop_last=True,
                            collate_fn=lambda batch: bert_collate_fn(batch, mask_id=mask_id, mask_prob=mask_prob)
                            )


    # train
    model.train()
    step = 0
    it = iter(train_dataloader)
    best_val = float("inf")
    while step < n_steps:
        try:
            x, y, am = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y, am = next(it)

        x, y, am = x.to(device), y.to(device), am.to(device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step % 50) == 0:
            val_loss = evaluate(val_dataloader)
            best_val = min(best_val, val_loss)
            print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")

        step += 1

