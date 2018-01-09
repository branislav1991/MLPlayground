import os
import numpy as np
import pathlib
import requests
from enum import Enum

CHECKPOINT_DIR = "./checkpoints"

class Mode(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3

def create_checkpoint_folder():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

def download_mnist_data(): # download MNIST as csv files to /mnist
    dataset = [{
        "url": "https://pjreddie.com/media/files/mnist_train.csv",
        "path": "./mnist/mnist_train.csv"
    },
    {
        "url": "https://pjreddie.com/media/files/mnist_test.csv",
        "path": "./mnist/mnist_test.csv"
    }]
    if (not pathlib.Path(dataset[0]["path"]).exists()) or (not pathlib.Path(dataset[1]["path"]).exists()):
        pathlib.Path('./mnist').mkdir(parents=True, exist_ok=True) 
        for data in dataset:
            r = requests.get(data["url"], stream=True, verify=False)
            if r.status_code == 200:
                with open(data["path"], 'wb') as f:
                    for chunk in r:
                        f.write(chunk)