import torch
import torch.nn as nn
import numpy as np


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




