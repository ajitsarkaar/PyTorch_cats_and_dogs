import os
from os.path import join as pjoin
import shutil
import glob
import random

import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader

def create_dataset(root_path=None, dest_path=None, train_size=1):
	if (not root_path) or (not dest_path):
		raise ValueError('Value "root_path" or "dest_path" not specified.')

	dnames = [el.split('.')[0].split('/')[-1] for el in glob.glob(pjoin(root_path, '*.jpg'))]
	random.shuffle(dnames)

	train_path = pjoin(dest_path, 'train', root_path.split('/')[-1])
	valid_path = pjoin(dest_path, 'val', root_path.split('/')[-1])
	for p in [train_path, valid_path]:
		if not os.path.isdir(p):
			os.makedirs(p, exist_ok=True)

	for fname in dnames[:train_size]:
		shutil.copy(pjoin(root_path, '{}.jpg'.format(fname)), 
			pjoin(train_path, '{}.jpg'.format(fname)))

	for fname in dnames[train_size:]:
		shutil.copy(pjoin(root_path, '{}.jpg'.format(fname)), 
			pjoin(valid_path, '{}.jpg'.format(fname)))

class PetDataset(Dataset):

	def __init__(self, datapath, transform=None):

		if not datapath:
			raise ValueError('Value "datapath" for initializing dataset not specified.')

		dp = pjoin(datapath, 'Dog')
		cp = pjoin(datapath, 'Cat')
		data = []
		
		# data += [el.split('.')[0].split('/')[-1] for el in glob.glob(pjoin(dp, '*.jpg'))]
		if (not os.path.isdir(dp)) and (not os.path.isdir(cp)):
			data += glob.glob(pjoin(datapath, '*.jpg'))
			labels = [-1]
		else:	
			data += glob.glob(pjoin(dp, '*.jpg'))
		
			dlen = len(data)
			labels = [0] * dlen
			
			data += glob.glob(pjoin(cp, '*.jpg'))
			tlen = len(data)
			labels += [1] * (tlen - dlen)

		self.data = np.array(data)
		self.labels = np.array(labels)
		
		if transform:
			self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		
		f = self.data[idx]
		im = Image.open(f)
		data = np.asarray(im.resize((150, 150)))
		if len(data.shape) == 3 and data.shape[-1] != 3:
			data = data[:, :, :-1]

		if len(data.shape) < 3:
			ndata = np.zeros((150, 150, 3))
			for i in range(3):
				ndata[:, :, i] = data
			data = ndata
		label = self.labels[idx]
		
		if self.transform:
			data, label = self.transform((data, label))

		return data, label

class ToTensor(object):
	"""
	Convert ndarrays in sample to Tensors.
	"""
	def __call__(self, data):
		data, label = data
		return (torch.from_numpy(data) if data is not None else data,
				torch.from_numpy(label) if label is not None else label)

class CustomNorm(object):
	"""
	Convert ndarrays in sample to Tensors.
	"""
	def __call__(self, data):
		data, label = data
		data = data / np.amax(data)
		data = data.transpose(2, 0, 1)

		return data.astype(np.float32) if data is not None else data, \
		np.array(label) if label is not None else label

if __name__ == '__main__':
	
	HOME = os.path.expanduser('~')
	dog_path = pjoin(HOME, 'Documents', 'spring20', 'deeplearning', \
		'p1', 'kagglecatsanddogs_3367a', 'PetImages', 'Dog')
	cat_path = pjoin(HOME, 'Documents', 'spring20', 'deeplearning', \
		'p1', 'kagglecatsanddogs_3367a', 'PetImages', 'Cat')
	dest_path = pjoin(HOME, 'Documents', 'spring20', 'deeplearning', \
		'p1', 'data')
	
	for a in [dog_path, cat_path]:
		create_dataset(a, dest_path, 10000)