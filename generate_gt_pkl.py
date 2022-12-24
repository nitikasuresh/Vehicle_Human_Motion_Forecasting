import pandas as pd
import numpy as np
import os
import pickle
import re


path_to_data = "/home/mhatfalv/argoverse-forecasting/val_data/test_obs/data/"

feature_files = os.listdir(path_to_data)
# https://stackoverflow.com/questions/52737587/sorting-a-list-of-strings-numerically
feature_files.sort(key=lambda fname: int(fname.split('.')[0]))

gt = dict()

for i, file in enumerate(feature_files):
    df = pd.read_csv(path_to_data + file) 
    agent_track = df[df["OBJECT_TYPE"] == "AGENT"]
    gt[i+1] = np.array([agent_track["X"],agent_track["Y"]])


output = open('myfile_GT.pkl', 'wb')
pickle.dump(gt, output)
output.close()

