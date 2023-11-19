import os
from lib import evaluation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_PATH = "../checkpoint/checkpoint_24.pth_528.2.tar"

DATA_PATH = "../Flickr30K/"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="test")

