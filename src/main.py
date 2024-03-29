from math import log
import sys
import logging
import json


sys.path.append("/home/peerapat/wongnai-sentiment/resources/trb_aware/trbawarepipeline")
from job import Job
from jobmanager import JobManager


def job_recieve (jobdesc):
    print(jobdesc)
    jb = JobManager()
    job = jb.createjob(jobdesc)
    jb.execute(job)
    
# if __name__ == "__main__":

#     if len(sys.argv) < 2 : 
#         logging.error("No input argument")
#         logging.error("Usage: python main.py <jobfile>")
#         exit(0)

#     jb = JobManager()
#     job = jb.createjob(sys.argv[1])
#     jb.execute(job)