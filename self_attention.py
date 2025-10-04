class selfAttentionHead(nn.Module):
	# this class implements a single head of a self attention block
	def __init__(self, seq_len, d_model, d_k, d_v):
		super().__init__()
		self.value_network = nn.Linear(model_dim, d_v)
		self.query_network = nn.Linear(model_dim, d_k)
		self.key_network = nn.Linear(model_dim, d_k)

	def forward(self, x):
		# x has shape (B, T, d_model)
		values = self.value_network(x) # (B, T, d_v)
		queries = self.query_network(x) # (B, T, d_k)
		keys = self.key_network(x) # (B, T, d_k)
		attention_weights = nn.softmax(queries @ keys.transpose(-2, -1))/np.sqrt(d_k) # (batch, T, T)
		result = attention_weights @ values # (B, T, d_v)
		return result




class selfAttentionBlock(nn.Module):
	# this class implements self attention
	# input tensor has dimensions: (batch, seq_len, d_model)
	# batch: batch_size 
	# seq_len: a sequence of tokens is fed at the same time to the module, rather than one by one
	# model_dim: the embedding dimensionality that is used to reprent each token
	# the output consists of the same dimensionality as the input: (batch, seq_len, model_dim)
	def __init__(self, n_heads, seq_len, d_model):
		super().__init__()
		assert d_model % n_heads == 0, 'model_dim should be a multiple of n_heads'
		d_k = d_model // n_heads
		d_v = d_model // n_heads
		self.heads = [selfAttentionHead(seq_len, d_model, d_k, d_v) for _ in range(n_heads)]

	def forward(self, x):
		# x has shape (B, T, d_model)
		head_outputs = [head(x) for head in self.heads]
		concatenated_output = torch.cat(head_outputs, dim = -1)
		


