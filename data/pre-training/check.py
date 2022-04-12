# This script checks if the dataset created contains a train.h5 file
# Run this file from the root folder of the project.
from os import path
import os
dataset="anti-sars-"
dirs=54
BASE_DIR = path.dirname(__file__)
for i in range(0, dirs):
    current_dataset = dataset + str(i)
    p = '{}/{}/train.h5'.format(BASE_DIR, current_dataset)
    # print(i, p, path.exists(p))
    if not path.exists(p):
        print(i)


