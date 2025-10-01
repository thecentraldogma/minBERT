# simple mlp

import torch
import torch.nn as nn
from base_model import base_model


def mse_loss(x, y):
    return ((x-y)**2).mean()

class model_ff(base_model):

	def __init__(self, input_size):
		super().__init__()
		self.linear = nn.Linear(input_size, 1)
		self.relu = nn.ReLU()

	def forward(self, x, target = None): 
		# x is the input tensor
		x = self.linear(x)
		x = self.relu(x)

			

		# if targets are provided, compute a loss too
		loss = None
		if target == None: 
			loss = mse_loss(x, target)

		return x, loss
