import logging
from trbawarepipeline.job import Job
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
                
        model = KeyedVectors.load_word2vec_format(r'resources/trb_aware/thwiki_data/models/thai2vec.bin',binary=True)
        vocab = model.index_to_key
        vocab = [''] + vocab
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
        trainvec = text_to_vec(traintext,vocab)
        train_idx_list = index_padding(trainvec,maxlen)    
        testvec = text_to_vec(testtext,vocab)
        test_idx_list = index_padding(testvec,maxlen)
        traindata = []
        testdata = []
        for trainrow, train_idx  in zip(trainjsonData['data'], train_idx_list):
            trainrow['Vector'] = ' '.join(str(e) for e in train_idx)
            trainrow['Length'] = len(train_idx.tolist())
            traindata.append(trainrow)
            
            
        for testrow ,test_idx in zip( testjsonData['data'], test_idx_list):
            testrow['Vector'] = ' '.join(str(e) for e in test_idx)
            testrow['Length'] = len(test_idx.tolist())
            logging.debug(f"test_idx {test_idx}")
            testdata.append(testrow)
        
        traindict = dictformat(trainjsonData,traindata)
        testdict = dictformat(testjsonData,testdata)
        with open(writetrainfilepath, 'w') as writetrainjson ,open(writetestfilepath, 'w') as writetestjson :
            writetrainjson.write(json.dumps(
                traindict, indent=4, ensure_ascii=False))
            writetestjson.write(json.dumps(
                testdict, indent=4, ensure_ascii=False))
            
        np.savetxt(writeDensefilepath, dense_vec)
        
    def teardown(self):
        logging.info(f"Complete!")
        
        
def dictformat(jsondata,data):
    timestamp = str(datetime.datetime.now(pytz.timezone('Asia/Bangkok')))
    total = len(data)
    datadict = { "metadata" : {
            "timestamp": timestamp, 
            "positive": jsondata['metadata']['positive'], 
            "negative": jsondata['metadata']['negative'], 
            "total": total,
            "maxlength": jsondata['metadata']['maxlength'] ,
            }, 
            "data" : data
                }
    
    return datadict
        
    
def text_to_vec(input,vocab):
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
            # logging.debug(f"Index {vec}")
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

# np.fromstring(v, dtype=int,sep=' ')
# v = ' '.join(str(e) for e in a)