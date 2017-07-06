import numpy as np
import os
import csv
import time
import random
from glove import Corpus
from glove import Glove
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
	return np.asarray(np.where(binary_arr==1)[0])

def get_word_seq(idx,wseq):
	# if np.max(idx) > len(wseq):
	# 	assert 'index %d is out of dimension' % np.max(idx)
	# else:
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
			yield list(doc)
			
for b in range(len(cuts)):
	corpus = iterate_corpus(d[:cuts[b],:],wseq)

	corpus_model = Corpus()

	corpus_model.fit(corpus,window=8)

	# corpus_model.save('08_Glove/corpus.model')

	print('Dict size: %s' % len(corpus_model.dictionary))
	print('Collocations: %s' % corpus_model.matrix.nnz)

	glove = Glove(no_components=fsize, learning_rate=0.05)

	glove.fit(corpus_model.matrix, epochs=ite, no_threads=6, verbose=True)

	glove.add_dictionary(corpus_model.dictionary)

	glove.save('08_Glove/model_d'+str(fsize)+'_size'+str(cuts[b])+'.model')


