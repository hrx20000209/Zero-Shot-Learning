import math
import os

import json
import random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image


def get_path(image_files):
    image_files = np.squeeze(image_files)
    new_image_files = []
    for image_file in image_files:
        image_file = image_file[0]
        image_file = '/'.join(image_file.split('/')[8:])
        new_image_files.append(image_file)
    new_image_files = np.array(new_image_files)
    return new_image_files


zeroshot_benchmark = ['AWA2', 'CUB', 'SUN']


def build_zeroshot_benchmark(set_id, root, transform, mode='train', n_shot=None):
    dataset = set_id
    return ZSL(root, dataset, mode, n_shot, transform)


class ZSL(Dataset):
    """ Zero-Shot Benchmark dataset """

    def __init__(self, root, dataset, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.dataset = dataset
        self.mode = mode

        matcontent = sio.loadmat(os.path.join(self.path, "/xlsa17/data/", self.dataset, 'res101.mat'))
        image_files = get_path(matcontent['image_files'])
        labels = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(os.path.join(self.path, "/xlsa17/data/", self.dataset, 'clip_splits.mat'))
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        test_seen_label = labels[test_seen_loc].astype(int)
        self.seenclasses = np.unique(test_seen_label)
        test_unseen_label = labels[test_unseen_loc].astype(int)
        self.unseenclasses = np.unique(test_unseen_label)
        self.allclasses = np.arange(len(self.seenclasses) + len(self.unseenclasses))

        self.cname = []
        allclasses_names = matcontent['allclasses_names']
        for item in allclasses_names:
            name = item[0][0]
            if dataset == 'AWA2':
                name = name.strip().replace('+', ' ')
            elif dataset == 'CUB':
                name = name.strip().split('.')[1].replace('_', ' ')
            elif dataset == 'SUN':
                name = name.strip().replace('_', ' ')
            self.cname.append(name)


        if self.mode == 'train':
            self.image_list = list(image_files[trainval_loc])
            self.label_list = list(labels[trainval_loc])
        elif self.mode == 'seen':
            self.image_list = list(image_files[test_seen_loc])
            self.label_list = list(labels[test_seen_loc])
        else:
            self.image_list = list(image_files[test_unseen_loc])
            self.label_list = list(labels[test_unseen_loc])


        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.dataset, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()

