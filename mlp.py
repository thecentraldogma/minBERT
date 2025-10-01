# simple mlp

import torch
import torch.nn as nn



class model_ff(nn.Module):
	def __init__(self, input_size, num_layers, width, batch_norm):
		self.batch_norm = batch_norm
		self.batch_norms = []
		self.linear_layers = []
		last_size = input_size
		for i in range(0, num_layers):
			self.linear_layers.append(nn.Linear(last_size, width))
			last_size = width
			if batch_norm == True: 
				self.batch_norms.append(nn.batch_norm(width)) # output size of last layer



	def forward(self, x, target = None): 
		# x is the input tensor
		for i in range(0, len(self.linear_layers)):
			layer = self.linear_layers[i]
			x = layer(x)
			x = self.relu(x)
			if self.batch_norm:
				x = self.batch_norms[i](x)
			

		# if targets are provided, compute a loss too
		if target == None: 


		return x