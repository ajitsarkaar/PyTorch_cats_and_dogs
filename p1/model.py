import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PetNet(nn.Module):
	"""docstring for PetNet"""
	def __init__(self):
		super(PetNet, self).__init__()
		
		self.relu = nn.ReLU(inplace=True)
		self.nfeat = 64

		def conv_pool_block(fan_in, fan_out):
			conv = nn.Conv2d(fan_in, fan_out, 3)
			pool = nn.MaxPool2d(2)

			return nn.Sequential(conv, self.relu, pool)

		# features = conv_pool_block(3, self.nfeat)

		self.conv1 = conv_pool_block(3, self.nfeat)
		self.conv2 = conv_pool_block(self.nfeat, self.nfeat)
		self.conv3 = conv_pool_block(self.nfeat, self.nfeat)

		# for i in range(2):
		# 	features += conv_pool_block(self.nfeat, self.nfeat)
		# self.features = nn.Sequential(*features)

		flat_size = 18496
		self.fc1 = nn.Linear(flat_size, 32)
		self.fc2 = nn.Linear(32, 2)

		self.dense = nn.Sequential(self.fc1, self.relu, self.fc2)

	def forward(self, data):
		conv1_out = self.conv1(data)
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)

		flats = torch.flatten(conv3_out, 1)
		dense = self.dense(flats)
		return dense, (conv1_out, conv2_out, conv3_out)
