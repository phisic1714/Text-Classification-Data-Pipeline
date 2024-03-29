import imp
import json
import os
from pathlib import Path
import logging
import logging.config
import sys
sys.path.append("/home/peerapat/wongnai-sentiment/resources/trb_aware")

from trbawarepipeline.job import Job
# from trbawarepipeline.jobimpl import DataPreparationJob
from trbawarepipeline import preparedata
from trbawarepipeline import datacleansing
from trbawarepipeline import datasplit
from trbawarepipeline import datatoindex
from trbawarepipeline import buildmodel
from trbawarepipeline import evaluation
from trbawarepipeline import modelcaller

TRB_AWARE_DEFAULT_HOME = "resources/trb_aware"
# TRB_AWARE_DEFAULT_LOGGING_CONF = 'config.ini'
TRB_AWARE_DEFAULT_LOGGING_CONF = 'config_debug.ini'


class TestJob(Job): 

    def __init__(self, desc: dict):
        super().__init__(desc)

    def init(self): 
        logging.info(f"this is init from {__name__}")

    def run(self): 
        logging.info(f"this is run from {__name__}")

    def teardown(self):
        logging.info(f"this is teardown from {__name__}")


class JobManager(object): 
    
    def __init__(self): 
        self._env = dict()
        self.init_env()

    def init_env(self): 
        self._env['TRB_AWARE_HOME'] = os.environ.get('TRB_AWARE_DEFAULT_HOME', TRB_AWARE_DEFAULT_HOME)
        self._env['LOGGING_CONFIG'] = os.environ.get('LOGGING_CONFIG', TRB_AWARE_DEFAULT_LOGGING_CONF)

        self.home_path = Path(self._env['TRB_AWARE_HOME'])
        log_file = self.home_path / "logger" / self._env['LOGGING_CONFIG']
        print(f"Loading log configuration from {log_file}")
        # logging.config.fileConfig(fname=log_file, disable_existing_loggers=False)

    def createjob(self, job_file): 
        # sys.path.append("/Users/hook/Works/Nectec/dev/nhso/pythonrulesengine")
        logging.info(f"Loadding job from {job_file}")
        with open(job_file, 'r') as f:
            job_desc = json.load(f)

        # job_module = 'TestJob'
        # job_module = 'DataPreparationJob'
        # job_package = "trbawarepipeline.jobimpl"
        # job_module = 'jobimpl.DataPreparationJob'
        job_module = job_desc['jobmodule']
        
        # load_package = 'from {} import {}'.format(job_package, job_class)
        # logging.debug(f"Loading job package {load_package}")
        # eval(load_package)

        load_class = '{}(job_desc)'.format(job_module)
        logging.debug(f"Loading job class {job_module}")
        job = eval(load_class)
        print("fff")
        return job


    def execute(self, job: Job): 
        job.init()
        job.run()
        job.teardown()

