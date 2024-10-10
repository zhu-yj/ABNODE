import pandas
import numpy
import matplotlib
import os
import torch
import time
import datetime
import argparse
import sys
import sklearn
import pysindy

def print_imported_packages_versions():
    for name, module in sys.modules.items():
        if hasattr(module, '__version__'):
            print(f"{name} version: {module.__version__}")

print_imported_packages_versions()