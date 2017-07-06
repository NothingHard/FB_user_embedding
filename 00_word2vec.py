import gensim as gs
import numpy as np
import os
import csv
import time
import random
from scipy.io import mmread

# read configuration
exec(open("config.py").read())

if 'data' not in globals():
	data = mmread(dataset)

d = data.tocsr()

wseq = range(0,d.shape[1])
wseq = ['W{}'.format(i) for i in range(len(wseq))]
wseq = np.array(wseq)

print("nrow: %d, ncol: %d" % (d.shape[0],d.shape[1]))

def get_hot_idx(binary_arr):
	return np.where(binary_arr==1)[0]

def get_word_seq(idx,wseq):
	if np.max(idx) > len(wseq):
		assert 'index %d is out of dimension' % np.max(idx)
	else:
		return wseq[idx]

class iterate_corpus(object):
	def __init__(self,mtx,wseq):
		# mtx is a sparse matrix
		# wseq is a pre-defined word sequence, like "W0","W1","W2"
		self.mtx = mtx
		self.wseq = wseq

	def __iter__(self):
		for i in range(0,self.mtx.shape[0]):
			b = np.asarray(self.mtx[i,:].todense())[0]
			idx = get_hot_idx(b)
			doc = get_word_seq(idx,self.wseq)
			yield doc

# for every data size to build a model
for b in range(len(cuts)): 
	corpus = iterate_corpus(d[:cuts[b],:],wseq)
	
	print("===============")
	print("data size = "+str(cuts[b]))
	print("04_UCBOW(continuous bag of words) starting...")

	start = time.time()

	model_CBOW = gs.models.Word2Vec(size=fsize,window=8,min_count=1,sg=0,workers=6)
	model_CBOW.build_vocab(corpus)
	for _ in range(ite):
		model_CBOW.train(corpus)
	model_CBOW.save("04_UCBOW/model_d"+str(fsize)+"_size"+str(cuts[b])+".word2vec")

	end = time.time()
	print("take "+str(end-start)+" seconds")


	print("03_USG(skip-gram) starting...")

	start = time.time()

	model_USG = gs.models.Word2Vec(size=fsize,window=8,min_count=1,sg=1,workers=6)
	model_USG.build_vocab(corpus)
	for _ in range(ite):
		model_USG.train(corpus)
	model_USG.save("03_USG/model_d"+str(fsize)+"_size"+str(cuts[b])+".word2vec")

	end = time.time()

	print("take "+str(end-start)+" seconds")

print("word2vec embedding model finished.")