import os
from lib import evaluation

import torch
torch.set_num_threads(4)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_PATH = "../checkpoint/checkpoint_12.pth_535.4.tar"


DATA_PATH = "../MS-COCO/"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="testall", fold5=True)
