import torch
import pickle
import numpy as np
import os
from Models import Contextual
import torch
import torch.nn as nn
from metric import AP, MRR, Precision
from tqdm import tqdm
in_path = 'datasample/demo_log'
filenames = sorted(os.listdir(in_path))
vocab = pickle.load(open('vocab.dict', 'rb'))
adhoc = pickle.load(open('adhoc.dict', 'rb'))
torch.manual_seed(2019) # cpu
torch.cuda.manual_seed(2019)
cudaid=1
#device_ids = [0, 1]
batch_size = 64
max_querylen = 20
max_hislen = 50
def collate_fn_train(insts):
	''' Pad the instance to the max seq length in batch '''
	querys, docs1, docs2, label, his_sequence, his_pos, features1, features2 = zip(*insts)

	querys = torch.LongTensor(querys)
	docs1 = torch.LongTensor(docs1)
	docs2 = torch.LongTensor(docs2)
	label = torch.LongTensor(label)
	his_sequence = torch.LongTensor(his_sequence)
	his_pos = torch.LongTensor(his_pos)
	features1 = torch.FloatTensor(features1)
	features2 = torch.FloatTensor(features2)

	return querys, docs1, docs2, label, his_sequence, his_pos, features1, features2

class Dataset_train(torch.utils.data.Dataset):
	def __init__(
		self, querys, docs1, docs2, label, his_sequence, his_pos, features1, features2):
		self.querys = querys
		self.docs1 = docs1
		self.docs2 = docs2
		self.label = label
		self.his_sequence = his_sequence
		self.his_pos = his_pos
		self.features1 = features1
		self.features2 = features2

	def __len__(self):
		return len(self.querys)

	def __getitem__(self, idx):
		querys = self.querys[idx]
		docs1 = self.docs1[idx]
		docs2 = self.docs2[idx]
		label = self.label[idx]
		his_sequence = self.his_sequence[idx]
		his_pos = self.his_pos[idx]
		features1 = self.features1[idx]
		features2 = self.features2[idx]
		return querys, docs1, docs2, label, his_sequence, his_pos, features1, features2

def collate_fn_score(insts):
	''' Pad the instance to the max seq length in batch '''
	querys, docs, his_sequence, his_pos, features, lines = zip(*insts)

	querys = torch.LongTensor(querys)
	docs = torch.LongTensor(docs)
	his_sequence = torch.LongTensor(his_sequence)
	his_pos = torch.LongTensor(his_pos)
	features = torch.FloatTensor(features)

	return querys, docs, his_sequence, his_pos, features, lines

class Dataset_score(torch.utils.data.Dataset):
	def __init__(
		self, querys, docs, his_sequence, his_pos, features, lines):
		self.querys = querys
		self.docs = docs
		self.his_sequence = his_sequence
		self.his_pos = his_pos
		self.features = features
		self.lines = lines

	def __len__(self):
		return len(self.querys)

	def __getitem__(self, idx):
		querys = self.querys[idx]
		docs = self.docs[idx]
		his_sequence = self.his_sequence[idx]
		his_pos = self.his_pos[idx]
		features = self.features[idx]
		lines = self.lines[idx]
		return querys, docs, his_sequence, his_pos, features, lines

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

def divide_dataset(filename):  # 检查用户的实验数据中有几个session，滤掉少于一定session的用户
	session_sum = 0
	query_sum = 0 
	last_queryid = 0
	last_sessionid = 0
	with open(os.path.join(in_path, filename)) as fhand:
		for line in fhand:
			try:
				line, features = line.strip().split('###')
			except:
				line = line.strip()
			user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t') 
			queryid = sessionid + querytime + query

			if querytime < '2006-04-03 00:00:00':
				if queryid != last_queryid:
					last_queryid = queryid
				query_sum += 1
			elif querytime < '2006-05-16 00:00:00':
				if query_sum < 2:
					return False
				if sessionid != last_sessionid:
					session_sum += 1
					assert(last_queryid != queryid)
					last_sessionid = sessionid
				if queryid != last_queryid:
					last_queryid = queryid
			else:
				if sessionid != last_sessionid:  # 这里不区分valid 和 test
					session_sum += 1
					assert(last_queryid != queryid)
					last_sessionid = sessionid
				if queryid != last_queryid:
					last_queryid = queryid

		if session_sum < 3:
			return False
	return True

def cal_delta(targets):
	n_targets = len(targets)
	deltas = np.zeros((n_targets, n_targets))
	total_num_rel = 0
	total_metric = 0.0
	for i in range(n_targets):
		if targets[i] == '1':
			total_num_rel += 1
			total_metric += total_num_rel / (i + 1.0)
	metric = (total_metric / total_num_rel) if total_num_rel > 0 else 0.0
	num_rel_i = 0
	for i in range(n_targets):
		if targets[i] == '1':
			num_rel_i += 1
			num_rel_j = num_rel_i
			sub = num_rel_i / (i + 1.0)
			for j in range(i+1, n_targets):
				if targets[j] == '1':
					num_rel_j += 1
					sub += 1 / (j + 1.0)
				else:
					add = (num_rel_j / (j + 1.0))
					new_total_metric = total_metric + add - sub
					new_metric = new_total_metric / total_num_rel
					deltas[i, j] = new_metric - metric
		else:
			num_rel_j = num_rel_i
			add = (num_rel_i + 1) / (i + 1.0)
			for j in range(i + 1, n_targets):
				if targets[j] == '1':
					sub = (num_rel_j + 1) / (j + 1.0)
					new_total_metric = total_metric + add - sub
					new_metric = new_total_metric / total_num_rel
					deltas[i, j] = new_metric - metric
					num_rel_j += 1
					add += 1 / (j + 1.0)
	return deltas

querys_train = []
docs1_train = []
docs2_train = []
label_train = []
his_sequence_train = []
his_pos_train = []
features1_train = []
features2_train = []

querys_test = []
docs_test = []
his_sequence_test = []
his_pos_test = []
features_test = []
lines = []

def prepare_pairdata(sat_list, doc_list, feature_list, qids, his_sequence):
	cutted_his_sequence = his_sequence[-max_hislen+1:]+[qids]
	init_his_sequence = np.zeros((max_hislen+1, max_querylen))
	init_his_sequence[-1] = sen2id("CLS CLS CLS")
	#init_his_sequence[-1] = qids
	init_his_pos = np.zeros((max_hislen+1))
	init_his_pos[-1] = len(cutted_his_sequence) + 1
	for i in range(max_hislen):
		init_his_sequence[i][0]=1
	for i in range(len(cutted_his_sequence)):
		init_his_sequence[i] = cutted_his_sequence[i]
		init_his_pos[i] = i+1
	delta = cal_delta(sat_list)
	n_targets = len(sat_list)
	for i in range(n_targets):
		for j in range(i+1, n_targets):
			if delta[i, j]>0:
				rel_doc = doc_list[j]
				rel_features = feature_list[j]
				irr_doc = doc_list[i]
				irr_features = feature_list[i]
				lbd = delta[i, j]
			elif delta[i, j]<0:
				rel_doc = doc_list[i]
				rel_features = feature_list[i]
				irr_doc = doc_list[j]
				irr_features = feature_list[j]
				lbd = -delta[i, j]
			else:
				continue
			if True:
				querys_train.append(qids)
				docs1_train.append(rel_doc)
				docs2_train.append(irr_doc)
				label_train.append(0)
				features1_train.append(rel_features)
				features2_train.append(irr_features)
				his_sequence_train.append(init_his_sequence)
				his_pos_train.append(init_his_pos)
				#delta_train.append(lbd)

def predata():
	x_train = []
	filenum=0
	key=0
	for filename in tqdm(filenames):
		if not divide_dataset(filename): # 判断该用户的log是否符合要求
			continue
		if key == 1 and querytime >= '2006-04-03 00:00:00': #There is a SAT-click in the sesssion
			prepare_pairdata(sat_list, doc_list, feature_list, last_qids, his_sequence)
		filenum += 1
		last_queryid = 0
		last_sessionid = 0
		last_qids = 0
		queryid = 0
		sessionid = 0
		key = 0
		doc_list = []
		sat_list = []
		his_sequence = []
		current_doc_sequence = []
		#feature_list = []

		fhand = open(os.path.join(in_path, filename))
		for line in fhand:
			try:
				line, features = line.strip().split('###')
			except:
				line = line.strip()
				#features = [0]*110
			user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t')
			#ser, sessionid, querytime, query, url, title, sat = line.strip().split('\t')
			queryid = sessionid + querytime + query
			if query.strip() == "":
				continue
			qids = sen2id(query)
			dids = sen2id(title)
			if querytime <= '2006-05-16 00:00:00':
				if queryid != last_queryid:
					#query_session_count += 1
					if key == 1 and querytime >= '2006-04-03 00:00:00': #There is a SAT-click in the sesssion
						prepare_pairdata(sat_list, doc_list, feature_list, last_qids, his_sequence)
						key = 0
					if last_qids != 0:
						his_sequence.append(last_qids)
						his_sequence.extend(current_doc_sequence)
					current_doc_sequence = []
					doc_list = []
					sat_list = []
					feature_list = []
					last_queryid = queryid
					last_qids = qids
				doc_list.append(dids)
				sat_list.append(sat)
				try:
					feature_list.append(float(adhoc[user + sessionid + querytime + query + url + urlrank]))
				except:
					feature_list.append(0)
				if int(sat) == 1:
					key = 1
					current_doc_sequence.append(dids)
			else:
				if queryid != last_queryid:
					his_sequence.append(last_qids)
					his_sequence.extend(current_doc_sequence)

					cutted_his_sequence = his_sequence[-max_hislen+1:]+[qids]
					init_his_sequence = np.zeros((max_hislen+1, max_querylen))
					init_his_sequence[-1] = sen2id("CLS CLS CLS ")
					init_his_pos = np.zeros((max_hislen+1))
					init_his_pos[-1] = len(cutted_his_sequence) + 1
					for i in range(max_hislen):
						init_his_sequence[i][0]=1
					for i in range(len(cutted_his_sequence)):
						init_his_sequence[i] = cutted_his_sequence[i]
						init_his_pos[i] = i+1

					current_doc_sequence = []
					last_queryid = queryid
					last_qids = qids
				if int(sat) == 1:
					current_doc_sequence.append(dids)
				features = float(adhoc[user + sessionid + querytime + query + url + urlrank])
				querys_test.append(qids)
				docs_test.append(dids)
				his_sequence_test.append(init_his_sequence)
				his_pos_test.append(init_his_pos)
				features_test.append(features)
				lines.append(line.strip('\n'))

pretrained_dict = torch.load("premodel.zyj")
model = Contextual()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
#model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(cudaid)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
def train(train_loader):
	model.train()
	epoch_iterator = tqdm(train_loader, ncols=120)
	for batch_idx, (query, docs1, docs2, label, his_sequence, his_pos, features1, features2) in enumerate(epoch_iterator):
		optimizer.zero_grad()

		query = query.cuda(cudaid)
		docs1 =docs1.cuda(cudaid)
		docs2 =docs2.cuda(cudaid)
		label = label.cuda(cudaid)
		his_sequence = his_sequence.cuda(cudaid)
		his_pos = his_pos.cuda(cudaid)
		features1 = features1.cuda(cudaid)
		features2 = features2.cuda(cudaid)

		score, pre, p_score = model(query, docs1, docs2, his_sequence, his_pos, features1, features2)
		#correct = torch.max(pre,1)[1].eq(label).cpu().sum()
		loss = criterion(p_score, label)
		loss.backward()
		optimizer.step()
		epoch_iterator.set_postfix(cont_loss=loss.item())
	torch.save(model.state_dict(),'model.zyj')

def score(score_loader, load=False):
	if load == True:
		model.load_state_dict(torch.load('model.zyj'))
	model.eval()
	test_loss = 0
	correct = 0
	f = open('test_score.txt','w')
	for query, docs, his_sequence, his_pos, features, lines in score_loader:

		query = query.cuda(cudaid)
		docs =docs.cuda(cudaid)
		his_sequence = his_sequence.cuda(cudaid)
		his_pos = his_pos.cuda(cudaid)
		features = features.cuda(cudaid)

		score, pre, p_score = model(query, docs, docs, his_sequence, his_pos, features, features)
		for line, sc in zip(lines, score):
			f.write(line+'\t'+str(float(sc[0]))+'\n')


if __name__ == '__main__':
	predata()
	train_loader = torch.utils.data.DataLoader(
		Dataset_train(
			querys=querys_train, docs1=docs1_train, docs2=docs2_train, label=label_train, his_sequence=his_sequence_train, his_pos=his_pos_train,  features1=features1_train, features2=features2_train),
		batch_size=batch_size,
		collate_fn=collate_fn_train)

	score_loader = torch.utils.data.DataLoader(
		Dataset_score(
			querys=querys_test, docs=docs_test, his_sequence=his_sequence_test, his_pos=his_pos_test, features=features_test, lines=lines),
		batch_size=batch_size,
		collate_fn=collate_fn_score)
	
	#model.load_state_dict(torch.load('model.zyj'))
	for epoch in range(1):
		train(train_loader)
		score(score_loader)
		evaluation = AP()
		with open('test_score.txt', 'r') as f:
			evaluation.evaluate(f)
		evaluation = MRR()
		with open('test_score.txt', 'r') as f:
			evaluation.evaluate(f)
		evaluation = Precision()
		with open('test_score.txt', 'r') as f:
			evaluation.evaluate(f)

