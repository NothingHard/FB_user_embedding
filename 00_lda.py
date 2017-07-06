import numpy as np
import os
import csv
import time
import random
import gensim as gs
import numpy as np
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
class lda_corpus(object):
	def __init__(self,corpus,dictionary):
		self.corpus = corpus
		self.dictionary = dictionary
	def __iter__(self):
		for tokens in corpus:
			yield self.dictionary.doc2bow(tokens)

for b in range(len(cuts)):
	corpus = iterate_corpus(d[:cuts[b],:],wseq)

	doc_stream = (tokens for tokens in corpus)
	id2word = gs.corpora.Dictionary(doc_stream)
	print(id2word)
	final_corpus = lda_corpus(corpus,id2word)

	start = time.time()
	lda_model = gs.models.LdaModel(final_corpus, num_topics=fsize, id2word=id2word, passes=4)
	end = time.time()
	print("take "+str(end-start)+" seconds")
	print("lda_model with corpus size="+str(cuts[b])+" constructed.")

	list_vec = []
	for tokens in final_corpus:
		# doc = id2word.doc2bow(tokens)
		lda_doc = lda_model[tokens]
		this = np.repeat(0.0,fsize)
		for idx,val in lda_doc:
			this[idx] = val
		list_vec.append(this)
	f = open("06_LDA_n"+str(least)+"_d"+str(fsize)+"_size"+str(cuts[b])+".csv","w")
	w = csv.writer(f)
	w.writerows(list_vec)
	f.close()

	lda_model.save('06_LDA/model_d'+str(fsize)+'_size'+str(cuts[b])+'.model')
	id2word.save('06_LDA/model_d'+str(fsize)+'_size'+str(cuts[b])+'.dictionary')

