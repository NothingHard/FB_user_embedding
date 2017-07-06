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

# by ppl as document word 2 vec on each like entity
d = data.tocsr()

wseq = range(0,d.shape[1])
wseq = ['W{}'.format(i) for i in range(len(wseq))]
wseq = np.array(wseq)

print("nrow: %d, ncol: %d" % (d.shape[0],d.shape[1]))


def get_hot_idx(binary_arr):
    return np.asarray(np.where(binary_arr==1)[0])

def get_word_seq(idx,wseq):
    # if np.max(idx) > len(wseq):
    #     assert 'index %d is out of dimension' % np.max(idx)
    # else:
    return wseq[idx]

def form_corpus(d,wseq):
    corpus = []
    for i in range(0,d.shape[0]):
        b = np.asarray(d[i,:].todense())[0]
        idx = get_hot_idx(b)
        doc = get_word_seq(idx,wseq)
        corpus.append(doc)
    return corpus

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
            yield gs.models.doc2vec.TaggedDocument(doc,[i])

# for every data size to build a model
for b in range(len(cuts)): 
    corpus = iterate_corpus(d[:cuts[b],:],wseq) # iterable corpus used for training 
    input_like = form_corpus(d[:cuts[b],:],wseq) # used for output

    print("===============")
    print("data size = "+str(cuts[b]))
    print("00_PVDBOW starting...")

    start = time.time()

    model_pvdbow = gs.models.doc2vec.Doc2Vec(size=fsize,min_count=1,dm=0,window=8,workers=6)
    model_pvdbow.build_vocab(corpus)
    for _ in range(ite):
        model_pvdbow.train(corpus)
    model_pvdbow.save("00_PVDBOW/model_d"+str(fsize)+"_size"+str(cuts[b])+".doc2vec")

    end = time.time()
    print("take "+str(end-start)+" seconds")

    
    list_doc2vec = []
    for i in range(len(input_like)):
        this = model_pvdbow.infer_vector(input_like[i])
        list_doc2vec.append(this)
    f = open("00_PVDBOW_n"+str(least)+"_d"+str(fsize)+"_size"+str(cuts[b])+".csv","w")
    w = csv.writer(f)
    w.writerows(list_doc2vec)
    f.close()

    print("01_PDB starting...")

    start = time.time()

    model_pdb = gs.models.doc2vec.Doc2Vec(size=fsize, min_count=1, window=8, workers=6)
    model_pdb.build_vocab(corpus)
    for _ in range(ite):
        model_pdb.train(corpus)
    model_pdb.save("01_PDB/model_d"+str(fsize)+"_size"+str(cuts[b])+".doc2vec")

    end = time.time()
    print("take "+str(end-start)+" seconds")

    list_doc2vec = []
    for i in range(len(input_like)):
        this = model_pdb.infer_vector(input_like[i])
        list_doc2vec.append(this)
    f = open("01_PDB_n"+str(least)+"_d"+str(fsize)+"_size"+str(cuts[b])+".csv","w")
    w = csv.writer(f)
    w.writerows(list_doc2vec)
    f.close()
print("doc2vec embedding model finished.")