import os
import torch
import numpy as np
import scipy.io as sio
import clip
from math import floor
from torchvision.datasets import Food101
from torchvision.utils import save_image
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


def get_classnames_dict(path):
    classnames_dict = {}
    with open(path + '/classnames.txt', 'r') as f:
        for line in f:
            label, class_name = line.replace('\n', '').split(' ')
            class_name = class_name.replace('_', ' ')
            label = eval(label) - 1
            classnames_dict[label] = class_name
    return classnames_dict


class ZeroShotDataset(torch.utils.data.Dataset):
    """ Zero-Shot Benchmark dataset """

    def __init__(self, root, dataset, preprocess, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.dataset = dataset
        self.mode = mode
        self.preprocess = preprocess

        matcontent = sio.loadmat(os.path.join(self.path, "xlsa17/data/", self.dataset, 'res101.mat'))
        image_files = get_path(matcontent['image_files'])
        labels = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(os.path.join(self.path, "xlsa17/data/", self.dataset, 'att_splits.mat'))
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
        elif self.mode == 'unseen':
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
        image = self.preprocess(Image.open(image_path).convert('RGB'))
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset, preprocess, mode='seen'):
        self.root = root
        self.dataset = dataset
        self.preprocess = preprocess
        self.path = os.path.join(self.root, self.dataset)

        self.classnames_dict = get_classnames_dict(self.path)
        self.text = list(self.classnames_dict.values())
        class_num = len(self.text)
        self.split = floor(class_num / 4)

        matcontent = sio.loadmat(self.path + '/imagelabels.mat')
        labels = np.squeeze(matcontent['labels'])
        labels = np.floor(labels - np.ones(len(labels)))
        text = []
        for idx in labels:
            label = self.classnames_dict.get(idx)
            text.append(label)

        if dataset == 'flowers-102':
            self.path = os.path.join(self.path, 'jpg')

        image_files = np.array(os.listdir(self.path))
        image_files.sort()

        if mode == 'unseen':
            unseen_idx = np.where(labels <= self.split)[0]
            self.image_files = image_files[unseen_idx]
            self.labels = labels[unseen_idx]
            # self.names = [self.classnames_dict[i] for i in self.labels]
            self.classnames = self.text[:self.split]
        else:
            self.image_files = np.zeros(0)
            self.labels = np.zeros(0)
            seen_idx = np.where(labels > self.split)[0]
            seen_image_files = image_files[seen_idx]
            seen_labels = labels[seen_idx]
            # self.names = [self.classnames_dict[i] for i in self.labels]
            self.classnames = self.text[self.split:]
            if mode == 'seen':
                for i in set(seen_labels):
                    idx = np.where(seen_labels == i)[0]
                    samples = seen_image_files[idx[:20]]
                    labels = seen_labels[idx[:20]]
                    self.image_files = np.concatenate((self.image_files, samples))
                    self.labels = np.concatenate((self.labels, labels))
            else:
                for i in set(seen_labels):
                    idx = np.where(seen_labels == i)[0]
                    samples = seen_image_files[idx[20:]]
                    labels = seen_labels[idx[20:]]
                    self.image_files = np.concatenate((self.image_files, samples))
                    self.labels = np.concatenate((self.labels, labels))

        self.class_num = len(self.classnames)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path, label = self.image_files[idx], self.labels[idx]
        image_path = os.path.join(self.path, path)
        image = self.preprocess(Image.open(image_path).convert('RGB'))

        return image, torch.tensor(label).long()

# if __name__ == "__main__":
#     clip_model, preprocess = clip.load("ViT-B/16", "cuda")
#     data = TestDataset('../../dataset', 'flowers-102', preprocess, 'train')
#     print(len(data.image_files))
#     save_image(data[100][0], './pic/test.jpg', normalize=True)
#     print(data[100])