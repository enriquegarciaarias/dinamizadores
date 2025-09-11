"""
@Purpose: Handles project-wide parameters
@Usage: Functions called by the main process
"""

import argparse
import os
import sys
import socket
from sources.common.common import processControl
from sources.common.utils import configLoader, dbTimestamp

# Constants for parameter files
JSON_PARMS = "config.json"

def manageArgs():
    """
    @Desc: Parse command-line arguments to configure the process.
    @Result: Returns parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Main process for Corpus handling.")
    parser.add_argument('--proc', type=str, help="Process type: MODEL, APPLY", default="APPLY")
    parser.add_argument('--act', type=str, help="Activity: actividad1,,,", default="actividad3")
    parser.add_argument('--unzip', type=str, help="Activity: actividad1,,,", default=True)
    args = parser.parse_args()

    # Establecer valores por defecto internos


    return args


def manageEnv():
    """
    @Desc: Defines environment paths and variables.
    @Result: Returns a dictionary containing environment paths.
    """

    config = configLoader()
    environment = config.get_environment()

    env_data = {}
    for key, value in environment.items():
        if "realPath" in key:
            env_data[key] = value
        else:
            env_data[key] = os.path.join(environment["realPath"], value)


    os.makedirs(env_data['.pycache'], exist_ok=True)
    os.environ['PYTHONPYCACHEPREFIX'] = env_data['.pycache']
    sys.pycache_prefix = env_data['.pycache']
    env_data['systemName'] = socket.getfqdn()
    return env_data

def manageDefaults():
    config = configLoader()
    environment = config.get_defaults()
    return environment

def manageModels():
    config = configLoader()
    models = config.get_models()
    return models

def getConfigs():
    """
    @Desc: Load environment settings, arguments, and hyperparameters.
    @Result: Stores configurations in processControl variables.
    """

    processControl.env = manageEnv()
    processControl.args = manageArgs()
    processControl.models = manageModels()
    processControl.defaults = manageDefaults()


