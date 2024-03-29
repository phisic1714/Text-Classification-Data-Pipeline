import logging

class Job(object): 
    def __init__(self, desc: dict): 
        self.desc = desc

    def init(self): 
        pass

    def run(self): 
        pass

    def teardown(self):
        pass