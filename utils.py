import random
import numpy as np
import torch


def setup_seed(seed, rank=0):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED


def get_source_list(file_path: str):
    files = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            l = line.replace("\n", "").rstrip()
            if '|' in l:
                files.append(l.split(' ')[-2])
            else:
                files.append(l.split(' ')[-1])
    return files

