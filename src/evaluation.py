import logging
from trbawarepipeline.job import Job
import numpy as np
import json
import tensorflow as tf
import datetime
import pytz

class ModelEvaluate(Job):
    def __init__(self, desc: dict):
        self.desc = desc
        super().__init__(desc)
        
    def init(self):
        logging.info(f"hello")
        
    def run(self):
        modelpath = self.desc['input'][0]
        vectedTrainDataFile = self.desc['input'][1]
        vectedTestDataFile = self.desc['input'][2]
        reportfile = self.desc['output']
        model = tf.keras.models.load_model(modelpath)
        Xtrain_idx = []
        Xtest_idx = []
        Ytrain = []
        Ytest = []
        with open(vectedTrainDataFile, encoding="utf8") as readtrainjson,open(vectedTestDataFile, encoding="utf8") as readtestjson:
            trainData = json.load(readtrainjson)
            testData = json.load(readtestjson)
            for trainrow in trainData['data'] :
                trainvector = np.fromstring(trainrow['Vector'], sep=' ')
                Xtrain_idx.append(trainvector)
                Ytrain.append(trainrow['Label'])
                print(trainvector)
            for testrow in testData['data'] :
                testvector = np.fromstring(testrow['Vector'], sep=' ')
                Xtest_idx.append(testvector)
                Ytest.append(testrow['Label'])
                
        train = Evaluation()
        train_loss, train_accuracy = model.evaluate(np.array(Xtrain_idx), np.array(Ytrain), verbose=False)
        train_result = model.predict(np.array(Xtrain_idx)) 
        train_pred = np.argmax(train_result, axis=1)
        train_confusion = train.get_confusion_value(Ytrain, train_pred.tolist()) 
        train_pre, train_rec, train_f1 = train.calculate_confusion_metrics(train_confusion)
        
        test = Evaluation()
        test_loss, test_accuracy = model.evaluate(np.array(Xtest_idx), np.array(Ytest), verbose=False)
        test_result = model.predict(np.array(Xtest_idx)) 
        test_pred = np.argmax(test_result, axis=1) 
        test_confusion = test.get_confusion_value(Ytest, test_pred.tolist()) 
        test_pre, test_rec, test_f1 = test.calculate_confusion_metrics(test_confusion) 
        dictionary = {
                "File name" : "model_report.json",
                "Embedding name" : "word2vec",
                "Time stamp": str(datetime.datetime.now(pytz.timezone('Asia/Bangkok'))),
                "Train Eval" :{
                    "Accuracy": train_accuracy,
                    "Loss": train_loss,
                    "Precision": train_pre,
                    "Recall": train_rec,
                    "F1-score": train_f1,
                },
                "Test Eval":{
                    "Accuracy": test_accuracy,
                    "Loss": test_loss ,
                    "Precision": test_pre ,
                    "Recall": test_rec,
                    "F1-score": test_f1,
                }
            }
        with open(reportfile, 'w') as writejson:
            writejson.write(json.dumps(dictionary, indent=4))
                        
    def teardown(self):
        logging.info(f"whatsup!")
        
        
class Evaluation:
 
  def get_confusion_value(self,y_true, y_pred):
    size = len(y_true)
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for i in range(size):
      if y_true[i] == 1 and y_pred[i] == 1:
        true_positive += 1
      if y_true[i] == 1 and y_pred[i] == 0:
        false_negative += 1
      if y_true[i] == 0 and y_pred[i] == 1:
        false_positive += 1
      if y_true[i] == 0 and y_pred[i] == 0:
        true_negative  += 1

    return true_positive, false_negative, false_positive, true_negative

  def calculate_confusion_metrics(self, confusion):
    tp, fn, fp, tn = confusion
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*((precision*recall)/(precision + recall))
    return precision, recall, f1

