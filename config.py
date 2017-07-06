# current dir
dirn = ["00_PVDBOW/","01_PDB/","03_USG/","04_UCBOW/"]

# used dataset
dataset = "/home/cmchang/SR/data/page_10k_v2/user_174k_page_10k.mtx"

# used data in Lasso
data_csv = "/home/cmchang/SR/data/page_10k_v2/user_first1k_page_10k.csv"

# used label
label_csv = "/home/cmchang/SR/data/page_10k_v2/label_income.csv"

# assumed number of users with labels
least = 1000

# current feature size
fsize = 100

# number of iteration while training like-embedding model
ite = 50 # if needed

# datasize in experiment
a = np.array([20,50,100])
cuts = np.hstack((a*10,a*100, np.array(range(12000,20001,2000))))
