# trainer.py: general purpose trainer
# set up dataloader
# run the training loop per batch

import torch
from torch import nn
from torch.utils.data import DataLoader

class trainer():
	def __init__(self, model, train_dataset, test_dataset, batch_size, learning_rate, optimizer, num_epochs):
		self.model = model
		self.train_dataloader = Dataloader(dataset = train_dataset, batch_size = batch_size)
		self.test_dataloader = Dataloader(dataset = test_dataset, batch_size = batch_size)

	def eval(self):
		# runs eval on the test set

	def train(self):
		self.model.train()
		for epoch in range(num_epochs):
			for batch in train_dataloader: 
				x, y = batch
				preds, loss = self.model(x, y)
				self.optimizer.step()







