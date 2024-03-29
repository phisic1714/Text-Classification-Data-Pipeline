import logging
from job import Job
import numpy as np
import json
from tqdm import tqdm
from gensim.models import KeyedVectors
import datetime
import pytz

class DataToIndex(Job):
    
    def __init__(self, desc: dict):
        self.desc = desc
        super().__init__(desc)
        
    def init(self):
        logging.info(f"Cleansing Data...")
        
    def run(self):
        tag = self.desc["tag"]
        model = KeyedVectors.load_word2vec_format(r'/home/peerapat/wongnai-sentiment/resources/trb_aware/thwiki_data/models/thai2vec.bin',binary=True)
        vocab = model.index_to_key
        vocab = [''] + vocab
        if tag == "Training":
            cleanedTrainDataFile = self.desc['input'][0]
            cleanedTestDataFile = self.desc['input'][1]
            writetrainfilepath = self.desc['output'][0]
            writetestfilepath = self.desc['output'][1]
            writeDensefilepath = self.desc['output'][2]
            traintext = []
            testtext = []
            with open(cleanedTrainDataFile, encoding="utf8") as trainfile , open(cleanedTestDataFile, encoding="utf8") as testfile:
                    trainjsonData = json.load(trainfile)
                    testjsonData = json.load(testfile)
                    traintext = [trainrow['Token'] for trainrow in trainjsonData['data']]
                    testtext = [testrow['Token'] for testrow in  testjsonData['data']]
                    
            print("Getting Vocab size...")
            dense_vec = None
            with tqdm(total=len(vocab)) as pbar:
                for v in vocab:
                    if dense_vec is None:
                        dense_vec = model[v].copy()
                    else:
                        dense_vec = np.vstack((dense_vec,  model[v]))
                    pbar.update()
            maxlen = trainjsonData['metadata']['maxlength']        
            train_idx = text_to_idx(traintext,vocab)
            train_idx_list = index_padding(train_idx,maxlen)    
            test_idx = text_to_idx(testtext,vocab)
            test_idx_list = index_padding(test_idx,maxlen)
            traindata = []
            testdata = []
            for trainrow, train_idx  in zip(trainjsonData['data'], train_idx_list):
                trainrow['Index'] = ' '.join(str(e) for e in train_idx)
                trainrow['Length'] = len(train_idx.tolist())
                traindata.append(trainrow)
            for testrow ,test_idx in zip( testjsonData['data'], test_idx_list):
                testrow['Index'] = ' '.join(str(e) for e in test_idx)
                testrow['Length'] = len(test_idx.tolist())
                logging.debug(f"test_idx {test_idx}")
                testdata.append(testrow)
            
            traindict = dictformat(trainjsonData,traindata,tag)
            testdict = dictformat(testjsonData,testdata,tag)
            with open(writetrainfilepath, 'w') as writetrainjson ,open(writetestfilepath, 'w') as writetestjson :
                writetrainjson.write(json.dumps(
                    traindict, indent=4, ensure_ascii=False))
                writetestjson.write(json.dumps(
                    testdict, indent=4, ensure_ascii=False))
            np.savetxt(writeDensefilepath, dense_vec)
            
        elif tag == "Predict":
            inputpath = self.desc['input']
            outputpath = self.desc['output']
            with open(inputpath, encoding="utf8") as inputdata :
                    jsonData = json.load(inputdata)
                    text = [row['Token'] for row in jsonData['data']]
                    
            maxlen = jsonData['metadata']['maxlength']        
            index = text_to_idx(text,vocab)
            index_list = index_padding(index,maxlen)    
            data = []
            for row, idx  in zip(jsonData['data'], index_list):
                row['Index'] = ' '.join(str(e) for e in idx)
                row['Length'] = len(idx.tolist())
                data.append(row)
                
            datadict = dictformat(jsonData,data,tag)
            
            with open(outputpath, 'w') as outputdata :
                outputdata.write(json.dumps(
                    datadict, indent=4, ensure_ascii=False))
        
    def teardown(self):
        logging.info(f"Complete!")      
        
        
def dictformat(jsondata,data,tag):
    timestamp = str(datetime.datetime.now(pytz.timezone('Asia/Bangkok')))
    total = len(data)
    datadict = None
    
    if tag == "Training":
        datadict = { "metadata" : {
                "timestamp": timestamp, 
                "positive": jsondata['metadata']['positive'], 
                "negative": jsondata['metadata']['negative'], 
                "total": total,
                "maxlength": jsondata['metadata']['maxlength'] ,
                }, 
                "data" : data
                    }
        
    elif tag == "Predict":
        datadict = { "metadata" : {
                "timestamp": timestamp, 
                "total": total,
                "maxlength": jsondata['metadata']['maxlength']
                }, 
                "data" : data
                    }
    return datadict
        

        
def text_to_idx(input,vocab):
    output = []
    print("Converting word to vector...")
    with tqdm(total=len(input)) as pbar:
        for text in input :
            vec = []
            wordArray = text.split(" ")
            for word in wordArray :  
                if word in vocab:
                    vec.append(vocab.index(word))
            pbar.update()
            output.append(vec)  
    print("complete!")
    return output
        
def index_padding (input,maxlen):
    print("Padding Vector...")
    with tqdm(total=len(input)) as pbar:
        for index in range(len(input)) :
            if len(input[index]) <  maxlen :
                input[index] = np.hstack((input[index], np.zeros(maxlen-len(input[index]))))
            pbar.update()
    output = np.array(input, dtype=np.uint32)
    print("complete!")
    return output