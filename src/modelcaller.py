import logging
from job import Job
import numpy as np
import json
import csv
import requests
from gensim.models import KeyedVectors

from datacleansing import Cleansing


class ModelPredict(Job):
    
    def __init__(self, desc: dict):
        self.desc = desc
        super().__init__(desc)
        
    def init(self):
        logging.info(f"Cleansing Data...")
        
    def run(self):
        inputpath = self.desc['input']
        outputpath = self.desc['output']
        index = [] 
        # with open(csvinputpath, encoding="utf8") as csvr:
        #   csvReader = csv.DictReader(csvr , delimiter=',' ,fieldnames=headerList )
          
        #   for row in csvReader: 
        #     text.append(row['Text'])
        # print(text)
        # text_cleaned = data_cleansing(text)
        # print(text_cleaned)
        # text_idx = sent2idx(text_cleaned)
        # print(text_idx)
        with open(inputpath, encoding="utf8") as inputdata :
            data = json.load(inputdata)
            for row in data['data'] :
                print(row['Index'])
                data_idx = np.fromstring(row['Index'],sep=' ' , dtype=np.uint32)
                index.append(data_idx)
          
        predict = make_prediction(index)    
        print(f"Result: {predict}\n")
        fieldnames = ["id","predict","negative probability","positive probability"]
        with open(outputpath, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predict)
        
    def teardown(self):
        logging.info(f"Complete!")

def data_cleansing(input):
    Obj_text = Cleansing()
    output = []
    for text in  input: 
      print(f"input {text}")
      tokenized = Obj_text.token_comment(text)
        # corrected = Obj_text.correct_word(tokenized)
      print(f"tokenized {tokenized}")
      normalized = Obj_text.normalize_comment(tokenized)
        # removedStWord = Obj_text.remove_stopwords(normalized)
      print(f"normalized {normalized}")
      cleaned_array = Obj_text.remove_special_characters(normalized)
      print(f"cleaned_array {cleaned_array}")
      output.append(cleaned_array)
    return output

def sent2idx(input):
    # f=np.array([])
    model = KeyedVectors.load_word2vec_format(r'/home/peerapat/wongnai-sentiment/resources/trb_aware/thwiki_data/models/thai2vec.bin',binary=True)
    vocab = model.index_to_key
    vocab = [''] + vocab
    output = []
    textlenlist = []
    for text in input :
      xidx = []
      wordArray = text.split(" ")
      textlenlist.append(len(wordArray))
      print(text)
      for word in wordArray:  
          if word in vocab:
            xidx.append(vocab.index(word))
            print(f"{word} index = {vocab.index(word)}")
               
      output.append(xidx)    
    print(f"input {input}")
    maxlen = max(textlenlist)
    print(maxlen)
    for index in range(len(output)) :
      if len(output[index]) <  maxlen :
         output[index] = np.hstack((output[index], np.zeros(maxlen-len(output[index]))))
    print(output)  
         
    return np.array(output)

def make_prediction(instances):
    url = "http://localhost:8502/v1/models/mymodel:predict"  
    print(f"Index :{instances}")
    data = json.dumps({ "instances": np.array(instances).tolist()}) #
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    print(f"Result :{predictions}")
    result = []
    id = 1
    for d in predictions :
       
      datadict = { "id" : id ,
        "predict" : np.argmax(d) ,
        "negative probability": round(d[0],2),
        "positive probability": round(d[1],2) 
      }
      result.append(datadict)
      id += 1
    return  result

   