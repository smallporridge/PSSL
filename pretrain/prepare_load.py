import torch
import pickle
import numpy as np
import os
from premodel import contrastive
import torch
import torch.nn as nn
from metric import AP
import random
from tqdm import tqdm
contrastive_file = ['demo_qd_pair.txt' ,'demo_self_pair.txt', 'demo_user_pair.txt']
#contrastive_file = ['qd_contrastive.txt']
vocab = pickle.load(open('vocab.dict', 'rb'))
torch.manual_seed(2019) # cpu
torch.cuda.manual_seed(2019)
cudaid=1
batch_size = 200
max_querylen = 20
max_hislen = 50
def collate_fn_train(insts):
	''' Pad the instance to the max seq length in batch '''
	his_pos_1, his_pos_2, his_contrastive_1, his_contrastive_2 = zip(*insts)

	his_pos_1 = torch.LongTensor(his_pos_1)
	his_pos_2 = torch.LongTensor(his_pos_2)
	his_contrastive_1 = torch.LongTensor(his_contrastive_1)
	his_contrastive_2 = torch.LongTensor(his_contrastive_2)

	return his_pos_1, his_pos_2, his_contrastive_1, his_contrastive_2

class Dataset_train(torch.utils.data.Dataset):
	def __init__(
		self, his_pos_1, his_pos_2, his_contrastive_1, his_contrastive_2):
		self.his_pos_1 = his_pos_1
		self.his_pos_2 = his_pos_2
		self.his_contrastive_1 = his_contrastive_1
		self.his_contrastive_2 = his_contrastive_2

	def __len__(self):
		return len(self.his_pos_1)

	def __getitem__(self, idx):
		his_pos_1 = self.his_pos_1[idx]
		his_pos_2 = self.his_pos_2[idx]
		his_contrastive_1 = self.his_contrastive_1[idx]
		his_contrastive_2 = self.his_contrastive_2[idx]
		return his_pos_1, his_pos_2, his_contrastive_1, his_contrastive_2

def sen2id(sen):
	idx = []
	for word in sen.split():
		if word in vocab:
			idx.append(vocab[word])
		else:
			idx.append(vocab['<unk>'])
	idx = idx[:max_querylen]
	padding = [0] * (max_querylen - len(idx))
	idx = idx + padding
	return	idx

def process(sen1, sen2):
	init_his_sequence1 = np.zeros((max_hislen+1, max_querylen))
	init_his_sequence2 = np.zeros((max_hislen+1, max_querylen))
	init_his_pos_1 = np.zeros((max_hislen+1))
	init_his_pos_2 = np.zeros((max_hislen+1))
	sentences1 = sen1.split("\t")[-max_hislen:]
	sentences2 = sen2.split("\t")[-max_hislen:]
	init_his_pos_1[-1] = len(sentences1) + 1
	init_his_pos_2[-1] = len(sentences2) + 1
	init_his_sequence1[-1] = sen2id("CLS CLS CLS")
	init_his_sequence2[-1] = sen2id("CLS CLS CLS")
	for i in range(max_hislen):
		init_his_sequence1[i][0]=1
		init_his_sequence2[i][0]=1
	for i in range(len(sentences1)):
		init_his_sequence1[i+1] = sen2id(sentences1[i])
		init_his_pos_1[i] = i+1
	for i in range(len(sentences2)):
		init_his_sequence2[i+1] = sen2id(sentences2[i])
		init_his_pos_2[i] = i+1
	return init_his_sequence1, init_his_sequence2, init_his_pos_1, init_his_pos_2

def process_qd(sen1, sen2):
	his_contrastive_1 = sen2id(sen1)
	his_contrastive_2 = sen2id(sen2)
	his_pos_1 = [1]*max_querylen
	his_pos_2 = [1]*max_querylen
	return his_contrastive_1, his_contrastive_2, his_pos_1, his_pos_2

def sum_count(file_name):
	return sum(1 for _ in open(file_name))

def predata(filename):
	his_pos_1_train = []
	his_pos_2_train = []
	his_contrastive_1_train = []
	his_contrastive_2_train = []
	filenum=0
	key=0
	if 'qd' in filename:
		qd=True
	else:
		qd=False
	filelen = sum_count(filename)
	f=open(filename,"r")
	for line in tqdm(f, total=filelen):
		sentences1, sentences2 = line.strip().split("=====")	
		if qd:
			his_contrastive_1, his_contrastive_2, his_pos_1, his_pos_2 = process_qd(sentences1, sentences2)
		else:
			his_contrastive_1, his_contrastive_2, his_pos_1, his_pos_2 = process(sentences1, sentences2)
		his_pos_1_train.append(his_pos_1)
		his_pos_2_train.append(his_pos_2)
		his_contrastive_1_train.append(his_contrastive_1)
		his_contrastive_2_train.append(his_contrastive_2)
	return his_pos_1_train, his_pos_2_train, his_contrastive_1_train, his_contrastive_2_train

model = contrastive()
#model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(cudaid)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 4e-4)
def train(train_loader, filename):
	model.train()
	epoch_iterator = tqdm(train_loader, ncols=120)
	if 'qd' in filename:
		qd=True
	else:
		qd=False
	for batch_idx, (his_pos_1, his_pos_2, his_contrastive_1, his_contrastive_2) in enumerate(epoch_iterator):
		optimizer.zero_grad()

		his_pos_1 = his_pos_1.cuda(cudaid)
		his_pos_2 = his_pos_2.cuda(cudaid)
		his_contrastive_1 = his_contrastive_1.cuda(cudaid)
		his_contrastive_2 = his_contrastive_2.cuda(cudaid)

		loss, acc, _ = model(his_pos_1, his_pos_2, his_contrastive_1, his_contrastive_2, qd)
		loss.backward()
		optimizer.step()
		epoch_iterator.set_postfix(cont_loss=loss.item(), acc=acc.item())
	torch.save(model.state_dict(),'premodel.zyj')

if __name__ == '__main__':
	for filename in contrastive_file:
		print("load contrastive pairs: " + filename)
		his_pos_1_train, his_pos_2_train, his_contrastive_1_train, his_contrastive_2_train = predata(filename)
		for epoch in range(10):
			train_loader = torch.utils.data.DataLoader(
			Dataset_train(
				his_pos_1=his_pos_1_train, his_pos_2=his_pos_2_train, his_contrastive_1=his_contrastive_1_train, his_contrastive_2=his_contrastive_2_train),
				batch_size=batch_size,
				collate_fn=collate_fn_train,
				shuffle=True,
				num_workers=2)
			train(train_loader, filename)