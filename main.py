"""
@Purpose: Main script for initializing environment settings and start procesing the Image Captioning project, handling main modes:
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
    """
    Entry point for starting the main image caption process.

    This block of code is executed when the script is run directly. It logs the start of the process, retrieves configuration settings,
    and then triggers the main process. After the main process completes, it logs the completion of the task.

    The function performs the following steps:
    - Logs the start of the process.
    - Calls `getConfigs()` to retrieve necessary configurations.
    - Executes `mainProcess()` to handle model training or application.
    - Logs the completion of the process.

    :return: None
    """

    log_("info", logger, "********** STARTING Main Image Caption Process **********")
    getConfigs()


    mainProcess()
    log_("info", logger, "********** PROCESS COMPLETED **********")
