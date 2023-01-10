import torch
import scipy.io as sio

data_root = "../dataset/xlsa17/data"
dataset = "CUB"
class_embedding = 'att'

matcontent = sio.loadmat(data_root + "/" + dataset + "/res101.mat")
label = matcontent['labels'].astype(int).squeeze() - 1
feature = matcontent['features'].T

matcontent = sio.loadmat(data_root + "/" + dataset + "/" + class_embedding + "_splits.mat")
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
train_loc = matcontent['train_loc'].squeeze() - 1
val_unseen_loc = matcontent['val_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
attribute = torch.from_numpy(matcontent['att'].T).float()
print(label)
