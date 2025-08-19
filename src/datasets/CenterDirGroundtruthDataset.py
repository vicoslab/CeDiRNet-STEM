import os.path
import numpy as np

import torch
from torch.utils.data import Dataset
from datasets.LockableSeedRandomAccess import LockableSeedRandomAccess

class CenterDirGroundtruthDataset(Dataset, LockableSeedRandomAccess):

    def __init__(self, dataset, centerdir_groundtruth_op):

        self.dataset = dataset
        self.centerdir_groundtruth_op = centerdir_groundtruth_op

    def get_coco_api(self):
        return self.dataset.get_coco_api()

    def lock_samples_seed(self, index_list):
        if isinstance(self.dataset,LockableSeedRandomAccess):
            self.dataset.lock_samples_seed(index_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        return sample