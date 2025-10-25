"""
@Purpose: Main script for initializing environment settings and start procesing the dinamizadores project, handling main modes:
@Usage: Run `python mainProcess.py`.
"""

from sources.common.common import processControl, logger, log_
import os

from sources.dinamProcess import dinamizaProcess
from sources.common.paramsManager import getConfigs

def mainProcess():

    result = dinamizaProcess()
    return True


if __name__ == '__main__':

    log_("info", logger, "********** STARTING Main Dinamizadores Process **********")
    getConfigs()
    mainProcess()
    log_("info", logger, "********** PROCESS COMPLETED **********")
