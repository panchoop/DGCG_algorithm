import curves
import numpy as np
import pickle
import config
import matplotlib.pyplot as plt

foldername = 'temp-main3-5-GT3Ce_RotLn_T050_a0.10b0.10_N40_RotLn15-5-1/'
filename = foldername + 'iter_036_step_3_grad_flow_saved_variables.pickle'

# Load data
saved_files = open(filename, 'rb')
data = pickle.load(saved_files)
saved_files.close

# Get object of interest
current_measure = data['current_measure']

# General warning setting
print("When visualizing, remember to set by hand the alpha/beta values")
curves.alpha = 1
curves.beta = 1


import code; code.interact(local=dict(globals(), **locals()))

