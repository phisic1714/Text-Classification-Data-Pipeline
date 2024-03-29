import logging
from trbawarepipeline.job import Job
import numpy as np
import json
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.models import Sequential

class ModelBuilder(Job):
    def __init__(self, desc: dict):
        self.desc = desc
        super().__init__(desc)
        
    def init(self):
        logging.info(f"hello")
        
    def run(self):
        vectedTrainDataFile = self.desc["input"][0]
        vectedTestDataFile = self.desc["input"][1]
        densevecpath = self.desc["input"][2]
        modelsoutput = self.desc["output"]
        filtersize = self.desc["filter"]
        kernelsize = self.desc["kernel"]
        Xtrain_idx = []
        Xtest_idx = []
        Ytrain = []
        Ytest = []
        W = np.loadtxt(densevecpath)
        with open(vectedTrainDataFile, encoding="utf8") as readtrainjson,open(vectedTestDataFile, encoding="utf8") as readtestjson:
            trainData = json.load(readtrainjson)
            testData = json.load(readtestjson)
            for trainrow in trainData['data'] :
                trainvector = np.fromstring(trainrow['Vector'],sep=' ' , dtype=np.uint32)
                Xtrain_idx.append(trainvector)
                Ytrain.append(trainrow['Label'])
                print(trainvector)
            for testrow in testData['data'] :
                testvector = np.fromstring(testrow['Vector'],sep=' ', dtype=np.uint32)
                Xtest_idx.append(testvector)
                Ytest.append(testrow['Label'])
        n_epoch = 1 
        maxlen = trainData['metadata']['maxlength']
        print(f"w[0]={W.shape[0]}, w[1]={W.shape[1]}") 
        vocab_size = W.shape[0]
        embedding_dim = W.shape[1]
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen, name='embed')) 
        model.add(Conv1D(filtersize, kernelsize, activation='relu')) 
        model.add(GlobalMaxPooling1D()) 
        model.add(Dense(300, activation='relu')) 
        model.add(keras.layers.Dense(2, activation='softmax')) 
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
        model.summary()
        model.get_layer('embed').set_weights([W])
        model.get_layer('embed').trainable = False
        model.fit(np.array(Xtrain_idx), np.array(Ytrain),
                            epochs=n_epoch,
                            validation_data=(np.array(Xtest_idx), np.array(Ytest)),
                            batch_size=10)
        model.save(modelsoutput)
                    
    def teardown(self):
        logging.info(f"whatsup!")