import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class SelfAttentionHead(nn.Module):
	# this class implements a single head of a self attention block
	def __init__(self, seq_len, d_model, d_k, d_v):
		super().__init__()
		self.value_network = nn.Linear(d_model, d_v)
		self.query_network = nn.Linear(d_model, d_k)
		self.key_network = nn.Linear(d_model, d_k)
		self.d_k = d_k
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		# x has shape (B, T, d_model)
		values = self.value_network(x) # (B, T, d_v)
		queries = self.query_network(x) # (B, T, d_k)
		keys = self.key_network(x) # (B, T, d_k)
		attention_weights = self.softmax(queries @ keys.transpose(-2, -1)/np.sqrt(self.d_k)) # (batch, T, T)
		result = attention_weights @ values # (B, T, d_v)
		return result


class SelfAttentionBlock(nn.Module):
	# this class implements self attention
	# input tensor has dimensions: (batch, seq_len, d_model)
	# batch: batch_size 
	# seq_len: a sequence of tokens is fed at the same time to the module, rather than one by one
	# model_dim: the embedding dimensionality that is used to reprent each token
	# the output consists of the same dimensionality as the input: (batch, seq_len, model_dim)
	def __init__(self, n_heads, seq_len, d_model, p_drop):
		super().__init__()
		assert d_model % n_heads == 0, 'model_dim should be a multiple of n_heads'
		d_k = d_model // n_heads
		d_v = d_model // n_heads
		self.heads = [SelfAttentionHead(seq_len, d_model, d_k, d_v) for _ in range(n_heads)]
		self.linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(p=p_drop)


	def forward(self, x):
		# x has shape (B, T, d_model)
		head_outputs = [head(x) for head in self.heads]
		concatenated_output = torch.cat(head_outputs, dim = -1)
		out = self.dropout(concatenated_output)  # (B, T, d_model)

		return out


class TransformerBlock(nn.Module): 
	# this class implements a transformer, using the selfAttentionBlock
	def __init__(self, n_heads, seq_len, d_model, sa_p_drop, ff_dim, ff_p_drop):
		super().__init__()
		self.self_attention = SelfAttentionBlock(n_heads=n_heads, seq_len=seq_len, d_model=d_model, p_drop=sa_p_drop)
		self.layernorm_sa = nn.LayerNorm(d_model)
		self.layernorm_ff = nn.LayerNorm(d_model)

		self.ff = nn.Sequential(
			nn.Linear(d_model, ff_dim),
			nn.GELU(),
			nn.Dropout(p = ff_p_drop),
			nn.Linear(ff_dim, d_model),
			nn.Dropout(p = ff_p_drop),
			)


	def forward(self, x): 
		# x is the input tensor fo shape (B, T, d_model)
		# Stage 1: self attention with pre-LN
		a = self.self_attention(self.layernorm_sa(x)) 
		x = x + a

		# Stage 2: mlp
		f = self.ff(self.layernorm_ff(x))
		x = x + f
		
		return x



class CopyShiftDataset(Dataset):
    """
    Each sample is a fixed-length sequence of token IDs:
      input:  [x0, x1, x2, ..., x_{T-1}]
      target: [BOS, x0, x1, ..., x_{T-2}]
    BOS is token_id 0; random tokens are sampled from 1..(vocab_size-1)
    """
    def __init__(self, num_samples: int, T: int, vocab_size: int, seed: int = 42):
        assert vocab_size >= 3, "Use vocab_size >= 3 so we have BOS(0) + >=2 tokens."
        self.num_samples = num_samples
        self.T = T
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # tokens 1..vocab_size-1 are "content" tokens; 0 is BOS
        seq = [self.rng.randint(1, self.vocab_size-1) for _ in range(self.T)]
        inp = torch.tensor(seq, dtype=torch.long)                         # (T,)
        tgt = torch.tensor([0] + seq[:-1], dtype=torch.long)             # (T,)
        return inp, tgt



class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, token_ids):  # token_ids: (B, T)
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)  # (1, T)
        x = self.tok(token_ids) + self.pos(positions)                      # (B, T, d_model)
        return x



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




