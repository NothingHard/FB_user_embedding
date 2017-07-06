import time
import numpy as np
from scipy.io import mmread
from sklearn.decomposition import TruncatedSVD

# read configuration
exec(open("config.py").read())

if 'data' not in globals():
	data = mmread(dataset)

d = data.tocsr()
svd_er = TruncatedSVD(fsize)

# for every data size to build a model
for b in range(len(cuts)):
	print("===============")
	print("data size = "+str(cuts[b]))
	print("SVD starting...")

	start = time.time()
	corpus = d[:cuts[b],:]
	Xsvd = svd_er.fit_transform(corpus)

	end = time.time()

	print("take "+str(end-start)+" seconds")

	import csv
	f = open("05_SVD/model_d"+str(fsize)+"_size"+str(cuts[b])+".csv","w")
	w = csv.writer(f)
	w.writerows(Xsvd)
	f.close()
