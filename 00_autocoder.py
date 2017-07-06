from scipy.io import mmread
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Input
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.layers import GRU,LSTM,Reshape
from keras.regularizers import l2

# read configuration
exec(open("config.py").read())

if 'data' not in globals():
    data = mmread(dataset)

d = data.tocsr()

input_dim = d.shape[1]

# for every data size to build a model
for b in range(len(cuts)):
    # model name
    model_name = "02_AE/AE_d500_d100_size"+str(cuts[b])+".h5"

    # dtrain, dtest
    dtrain = d[:cuts[b],:]
    dtest = dtrain

    # structure of neural network
    origin_input = Input(shape=(input_dim,))
    # encoding part
    encoding = Dense(500,activation='relu',activity_regularizer=l2(1e-9))(origin_input)
    encoded_output = Dense(100,activation='relu',activity_regularizer=l2(1e-9))(encoding)
    # decoding part
    decoding = Dense(500,activation='relu',activity_regularizer=l2(1e-9))(encoded_output)
    decoded_output = Dense(input_dim,activation='sigmoid')(decoding)

    ### AutoEncoder ###
    def BatchGenerator(batch_size):
        global dtrain
        number_of_batches = dtrain.shape[0]/batch_size
        counter=0
        shuffle_index = np.arange(dtrain.shape[0])
        np.random.shuffle(shuffle_index)
        ds =  dtrain[shuffle_index, :]
        while 1:
            index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
            d_batch = ds[index_batch,:].todense()
            counter += 1
            yield np.array(d_batch), np.array(d_batch)
            if (counter > number_of_batches):
                np.random.shuffle(shuffle_index)
                counter=0

    # construct autoencoder model
    autoencoder = Model(input=origin_input, output=decoded_output)
    autoencoder.summary()
    autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

    # construct encoder model at layer 1
    # encoder1 = Model(input=origin_input, output=encoding)

    # construct encoder model at layer 2
    # encoder = Model(input=origin_input, output=encoded_output)

    from keras.callbacks import Callback
    class LossHistory(Callback):
        def on_train_begin(self,logs={}):
            self.loss=[]
            self.val_loss=[]
        def on_epoch_end(self,epoch,logs={}):
            self.loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))
    loss_history = LossHistory()

    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(model_name,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='auto')

    autoencoder.fit_generator(BatchGenerator(100),
                            samples_per_epoch=dtrain.shape[0],
                            nb_epoch=50,
                            verbose=0,
                            validation_data = (dtest.todense(),dtest.todense()),
                            callbacks=[loss_history,checkpoint])

    from keras.models import load_model
    autoencoder = load_model(model_name)
    autoencoder.summary()

    encoder = Sequential()
    encoder.add(Dense(500, input_shape=(d.shape[1],), weights=autoencoder.layers[1].get_weights(),activation='relu'))
    encoder.add(Dense(100, weights=autoencoder.layers[2].get_weights(),activation='relu'))
    encoder.summary()

    list_encoded = []
    for i in range(dtest.shape[0]):
        cont = encoder.predict(dtest[i,:].todense())[0]
        list_encoded.append(cont)
    import csv

    f = open("02_AE/n1000_d100_size"+str(cuts[b])+".csv","w")
    w = csv.writer(f)
    w.writerows(list_encoded)
    f.close()
