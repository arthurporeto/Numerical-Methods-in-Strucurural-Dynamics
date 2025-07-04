import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
ex3_dir = os.path.join(parent_dir, 'EX3')
sys.path.append(ex3_dir)
from Exercise_3_tools import *
#from Exercise
import numpy as np