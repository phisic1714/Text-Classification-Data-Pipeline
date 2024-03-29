import logging
from trbawarepipeline.job import Job

from tqdm import tqdm
import json
import datetime
import pytz

class DataSplit(Job):
    
    def __init__(self, desc: dict):
        self.desc = desc
        super().__init__(desc)
        
    def init(self):
        logging.info(f"Cleansing Data...")
        
    def run(self):
        readjsonpath = self.desc["input"]
        trainfilepath = self.desc["output"][0]
        testfilepath = self.desc["output"][1]
        traindata = []
        traintokenlen = []
        testdata = []
        testtokenlen = []
        with open(readjsonpath, encoding="utf8") as readjson:       
            jsonData = json.load(readjson)
            labelsize = [jsonData['metadata']['positive'],jsonData['metadata']['negative'],jsonData['metadata']['neutral']]
            minval = min(labelsize)
            logging.info(f"Minlabel size :{minval}")
            testsize = minval * (20/100)
            logging.info(f"{testsize}")
            trainsize = minval * (80/100)
            logging.info(f"{trainsize}")
            postest = 0
            negtest = 0
            postrain = 0
            negtrain = 0
            print("Spliting Data...")
            with tqdm(total=testsize*2) as testbar,tqdm(total=trainsize*2) as trainbar:
                for row in jsonData['data']:
                    if row['Label'] == 1:
                        if postest < testsize :
                            testdata.append(row)
                            testtokenlen.append(len(row['Token'].split(" ")))
                            postest += 1
                            testbar.update()
                        elif postrain < trainsize :
                            traindata.append(row)
                            traintokenlen.append(len(row['Token'].split(" ")))
                            postrain += 1
                            trainbar.update()      
                    if row['Label'] == 0:
                        if negtest < testsize:
                            testdata.append(row)
                            testtokenlen.append(len(row['Token'].split(" ")))
                            negtest += 1
                            testbar.update()
                        elif negtrain < trainsize:
                            traindata.append(row)
                            traintokenlen.append(len(row['Token'].split(" ")))
                            negtrain += 1
                            trainbar.update()
                        
                        
            traindict = dictformat(minval,trainsize,traindata,traintokenlen,postrain,negtrain)
            testdict = dictformat(minval,testsize,testdata,testtokenlen,postest,negtest)
            
            
        with open(trainfilepath, 'w') as writetrainjson ,open(testfilepath, 'w') as writetestjson:
            writetrainjson.write(json.dumps(
                traindict, indent=4, ensure_ascii=False))
            writetestjson.write(json.dumps(
                testdict, indent=4, ensure_ascii=False))
            
    def teardown(self):
        logging.info(f"Complete!")
        
        
def dictformat(minval,splitsize,data,tokenlen,poscount,negcount):
    timestamp = str(datetime.datetime.now(pytz.timezone('Asia/Bangkok')))
    total = len(data)
    maxlen = max(tokenlen)
    splitpercent = (splitsize/minval)*100
    datadict = { "metadata" : {
            "timestamp": timestamp, 
            "positive": poscount, 
            "negative": negcount, 
            "total": total,
            "maxlength": maxlen ,
            "Split percentage": splitpercent
            }, 
            "data" : data
                }
    
    return datadict
        