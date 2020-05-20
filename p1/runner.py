import os
from os.path import join as pjoin
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from prepare_data import PetDataset, CustomNorm, ToTensor
from model import PetNet

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

time_now = datetime.datetime.now()
log_filename = time_now.strftime("%d%b%Y_%Hh%Mm%Ss")
HOME = os.path.expanduser('~')
project_path = pjoin(HOME, 'ece6524') 
output_path = pjoin(HOME, 'ece6524', log_filename)
if not os.path.isdir(output_path):
	os.mkdir(output_path)

import logging
logs_path = pjoin(output_path, 'LOGS_{}.txt'.format(log_filename))
head = '%(asctime)-15s | %(filename)-10s | line %(lineno)-3d: %(message)s'
logging.basicConfig(filename=logs_path, format=head)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(head))
logger.addHandler(console)


context = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
logger.info('COMPUTE DEVICE: {}'.format(context))
gpu_flag = (context != torch.device('cpu'))

def main(c):
	lr = c['lr']
	bs = c['bs']
	ep = c['ep']

	composed_transforms = transforms.Compose([CustomNorm(), ToTensor()])
	tpath = c['train_path']
	train_dataset = PetDataset(tpath, transform=composed_transforms)
	vpath = c['val_path']
	valid_dataset = PetDataset(vpath, transform=composed_transforms)
	batch_range, valid_range = [], []

	tloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
	vloader = DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=4)
	nbatches = len(tloader)
	model = PetNet().to(context)

	batch_tracker, valid_tracker = [], []

	for en in range(1, ep + 1):
		logger.info('################################################################')
		logger.info('#### EPOCH # {:4d}'.format(en))
		logger.info('################################################################')
		bat_num, batch_acc = train(model, tloader, c)
		valid_acc = validate(model, vloader, c)
		batch_tracker += batch_acc
		batch_range += [el + ((en - 1) * nbatches) for el in bat_num]
		valid_tracker += valid_acc
		valid_range += [bat_num[-1] + ((en - 1) * nbatches)]

	plt.figure(figsize=(12, 8))
	plt.plot(batch_range, batch_tracker, 'r-')
	plt.plot(valid_range, valid_tracker, 'b-')
	plt.xlabel('batch')
	plt.ylabel('accuracy')
	plt.title('Training vs. Validation Performance')
	plt.legend(['Train Accuracy', 'Validation Accuracy'])
	plt.savefig(pjoin(output_path, '{}.png'.format('plots')))
	mpath = save_model(model, output_path)
	if c['is_analyze']:
		clss = ['dog', 'cat']
		for anim in clss:
			ana_path = pjoin(project_path, 'data', '{}_analyze'.format(anim))
			anacfg = {
				'analyze_path': ana_path,
				'pretrained_model': mpath
			}
			
			analyze(anacfg)


def analyze(c):
	if not os.path.isdir(pjoin(output_path, 'filters')):
		os.mkdir(pjoin(output_path, 'filters'))
	filt_path = pjoin(output_path, 'filters')
	composed_transforms = transforms.Compose([CustomNorm(), ToTensor()])
	ana_path = c['analyze_path']
	ana_dataset = PetDataset(ana_path, transform=composed_transforms)
	ana_loader = DataLoader(ana_dataset)

	model = PetNet()
	if c['pretrained_model'] == '':
		raise ValueError('pretrained_model path not specified.')
	model.load_state_dict(torch.load(c['pretrained_model']))
	model.to(context)
	nimal = c['analyze_path'].split('/')[-1].split('_')[0]
	
	with torch.no_grad():
		for bn, batch in enumerate(ana_loader):
			data, label = batch
			data = data.to(context) if gpu_flag else data
			label = label.to(context) if gpu_flag else label
			
			logits, conv_outputs = model(data)
			conv_outputs = [c.cpu().numpy() for c in conv_outputs]
			for i in range(len(conv_outputs)):
				rows = []
				for ix in range(8):
					filt = conv_outputs[i][0][ix * 8: (ix+1) * 8]
					edgesize = filt.shape[-1]
					filt = filt.reshape((8, -1))
					els = []
					for el in filt:
						inter_filter = ((el - np.amin(el)) / np.amax(el)) * 255.0
						els.append(inter_filter.reshape(edgesize, edgesize))
					filt_row = np.hstack((els))
					rows.append(filt_row)
				
				ims = np.vstack(rows)
				filt_name = '{}_conv{:1d}.png'.format(nimal, i+1)
				cv2.imwrite(pjoin(filt_path, filt_name), ims)


def train(model, dl, c):
	lr = c['lr']
	bs = c['bs']
	ep = c['ep']
	optimizer = O.SGD(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
	criterion = nn.CrossEntropyLoss()
	log_freq = len(dl) // 10
	batch_num, batch_history = [], []

	for bn, batch in enumerate(dl):
		data, label = batch
		data = data.to(context) if gpu_flag else data
		label = label.to(context) if gpu_flag else label
		
		optimizer.zero_grad()
		logits, _ = model(data)
		loss = criterion(logits, label)
		ncorrect = ((torch.argmax(logits, dim=1) == (label)).sum().item())
		
		if (bn + 1) % log_freq == 0:
			train_acc = ncorrect/bs
			batch_history.append(train_acc)
			batch_num.append((bn + 1))
			logger.info('# BATCH: {:4d} | LOSS: {:9.4f} | TRAIN ACC: {:4.2f}'.format(bn+1, loss, train_acc))
		
		loss.backward()
		optimizer.step()

	return batch_num, batch_history

def validate(model, dl, c):
	bs = c['bs']
	nbtchs = len(dl)
	ncorrect = 0
	ntotal = 0
	cc = 1
	model.eval()
	logger.info('')
	logger.info('**** RUNNING VALIDATION...')
	valid_history = []
	with torch.no_grad():
		for bn, batch in enumerate(dl):
			data, label = batch
			data = data.to(context) if gpu_flag else data
			label = label.to(context) if gpu_flag else label
			
			logits, conv_outputs = model(data)
			ntotal += len(label)
			ncorrect += ((torch.argmax(logits, dim=1) == (label)).sum().item())
			# if (bn+1) % (nbtchs//10) == 0:
			# 	if cc == 5:
			# 		logger.info('{:4d}/{:4d}... '.format(bn+1, nbtchs))
			# 		cc = 1
			# 	else:
			# 		logger.info('{:4d}/{:4d}... '.format(bn+1, nbtchs), end='')
			# 		cc += 1
	
	valid_acc = (ncorrect/ntotal)
	valid_history.append(valid_acc)
	logger.info('**** VALIDATION ACCURACY = {:5.2f}%'.format((ncorrect/ntotal) * 100))
	logger.info('')
	logger.info('')
	model.train()

	return valid_history

def save_model(model, opdir):
	time_now = datetime.datetime.now()
	model_name = time_now.strftime("%d%b%Y_%Hh%Mm%Ss")
	model_path = pjoin(opdir, '{}/{}.pt'.format('.', model_name))
	torch.save(model.state_dict(), model_path)
	return model_path

if __name__ == '__main__':
	tpath = pjoin(project_path, 'data', 'train')
	vpath = pjoin(project_path, 'data', 'val')
	cfg = {
		'train_path': tpath,
		'val_path': vpath,
		'is_analyze': True,
		'lr': 0.025,
		'weight_decay': 0.005,
		'bs': 10,
		'ep': 25
	}
	
	main(cfg)
