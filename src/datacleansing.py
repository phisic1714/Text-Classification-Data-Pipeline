import logging
from job import Job
from tqdm import tqdm
import json
import datetime
import pytz
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords

special_char = re.compile('[@_!#+$%ๆฯ฿^&*()<>,?/\\|-}{.-~:;“”\"\']')
numberic = re.compile("[0-9]+")
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)





class DataCleansing(Job):
    
    def __init__(self, desc: dict):
        self.desc = desc
        super().__init__(desc)
        
    def init(self):
        logging.info(f"Cleansing Data...")
        
    def run(self):
        readjsonpath = self.desc["input"]
        writejsonpath = self.desc["output"]
        tag = self.desc["tag"]
        cleaneddata = []
        tokenlen = []
        with open(readjsonpath, encoding="utf8") as readjson :
            jsonData = json.load(readjson)
            Obj_text = Cleansing()
            with tqdm(total=len(jsonData['data'])) as pbar:
                for row in jsonData['data']:
                    tokenized = Obj_text.token_comment(row['Text'])
                    # corrected = Obj_text.correct_word(tokenized)
                    normalized = Obj_text.normalize_comment(tokenized)
                    # removedStWord = Obj_text.remove_stopwords(normalized)
                    row['Token'] = Obj_text.remove_special_characters(normalized)
                    tokenlen.append(len(row['Token'].split(" ")))
                    cleaneddata.append(row)
                    pbar.update()
        if tag == "Training":
            datadict = { "metadata" : {
                "timestamp": str(datetime.datetime.now(pytz.timezone('Asia/Bangkok'))), 
                "positive": jsonData["metadata"]["positive"], 
                "negative": jsonData["metadata"]["negative"], 
                "neutral": jsonData["metadata"]["neutral"], 
                "total": len(cleaneddata),
                "maxlength": max(tokenlen)

                }, 
                "data" : cleaneddata
                    }
            with  open(writejsonpath, 'w') as writejson :
                json.dump(
                    datadict,writejson,indent=4, ensure_ascii=False)
                
        elif tag == "Predict":
            datadict = { "metadata" : {
                "timestamp": str(datetime.datetime.now(pytz.timezone('Asia/Bangkok'))), 
                "total": len(cleaneddata),
                "maxlength": max(tokenlen)
                }, 
                "data" : cleaneddata
                    }
            with  open(writejsonpath, 'w') as writejson :
                    json.dump(datadict,writejson,indent=4, ensure_ascii=False)
        
    def teardown(self):
        logging.info(f"Complete!")
        
        
class Cleansing:
       
    def token_comment(self,text):
        #  tokenized = word_tokenize(self.text, keep_whitespace=False)
        #  tokenized = word_tokenize(self.text, engine="mm",keep_whitespace=False)
        tokenized = word_tokenize(
            text, engine="attacut", keep_whitespace=False)
        #  tokenized = word_tokenize(self.text, engine="nercut",keep_whitespace=False)
        return tokenized
    # Method ทำการตัดอักขระพิเศษ

    def remove_special_characters(self,text):
        # remove special character
        removed = special_char.sub(r'', ' '.join(text))
        # remove emoji
        removed = emoji_pattern.sub(r'', removed)
        # remove number
        removed = numberic.sub(r'', removed)
        # print(f"before : {removed}")
        removed = re.sub(' +', ' ', removed)
        # print(f"after : {removed}")
        return removed
    # Method ทำการตัดคำที่ไม่จำเป็นหรือไม่สำคัญ

    def remove_stopwords(self,text):
        stopwords = list(thai_stopwords())
        removed = [i for i in text if i not in stopwords]
        return removed
    # Method ทำการแก้ข้อความที่พิมพ์ตัวอักษรซ้ำ หรือพิมพ์ตก
 
    def normalize_comment(self,text):
        normalized = []
        for word in text:
            word = normalize(word)
            normalized.append(word)
        return normalized