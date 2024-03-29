import logging
from job import Job

import csv
import json
import datetime
import pytz

class DataPreparationJob(Job): 
    
    def __init__(self, desc: dict):
        super().__init__(desc)
        self.desc = desc

    def init(self): 
        logging.info(f"this is init from {__name__}")

    def run(self): 
        readcsvpath = self.desc["input"]
        writejsonpath = self.desc["output"]
        tag = self.desc["tag"]
        datadict = None
        if tag == "Training":
           datadict = trainingprocess(readcsvpath)
        elif tag == "Predict":
           datadict = predictprocess(readcsvpath)
            
        
        with open(writejsonpath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(datadict, indent=4 , ensure_ascii=False ))
            
        print("Complete !")

    def teardown(self):
        logging.info(f"this is teardown from {__name__}")
    
def trainingprocess(readcsvpath):
        headerList = ['Text', 'Rating', 'Label']
        jsondata = []
        #read csv file
        with open(readcsvpath, encoding="utf8") as csvf:
            #load csv file data using csv library's dictionary reader
            next(csv.reader(csvf), None)
            csvReader = csv.DictReader(csvf , delimiter=';' ,fieldnames=headerList )
            #convert each csv row into python dict
            negcount = 0
            poscount = 0
            neucount = 0
            index = 0
            print("Converting CSV to JSON Data...")
            for row in csvReader:
                try:
                    if int(row['Rating']) > 3 :
                        row['Label'] = 1
                        poscount += 1
                    elif int(row['Rating']) == 3 :
                        row['Label'] = -1
                        neucount += 1
                    elif int(row['Rating']) < 3 :
                        row['Label'] = 0
                        negcount += 1
                        
                    #add this python dict to json array
                    row['ID'] = index
                    index += 1
                    jsondata.append(row)
                except:
                    jsondata.append(row)
                    
        datadict = { "metadata" : {
        "timestamp": str(datetime.datetime.now(pytz.timezone('Asia/Bangkok'))), 
        "positive": poscount, 
        "negative": negcount, 
        "neutral": neucount, 
        "total": len(jsondata)
        }, 
        "data" : jsondata
            }
        return  datadict
        
def predictprocess(readcsvpath): 
    
        headerList = ['ID', 'Text']
        jsondata = [] 
        with open(readcsvpath, encoding="utf8") as csvr:
          csvReader = csv.DictReader(csvr , delimiter=',' ,fieldnames=headerList )
          for row in csvReader: 
            jsondata.append(row)
        datadict = { "metadata" : {
        "timestamp": str(datetime.datetime.now(pytz.timezone('Asia/Bangkok'))), 
        "total": len(jsondata),
        }, 
        "data" : jsondata
            }
        return  datadict