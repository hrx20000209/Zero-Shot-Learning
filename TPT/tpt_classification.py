import argparse
import os
import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import scipy.io as sio

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def cal_acc_per_class(predicted_list, label_list):
    predicted_list = np.array(predicted_list)
    label_list = np.array(label_list)
    target_classes = np.unique(label_list)
    acc_per_class = 0
    for i in target_classes:
        idx = (label_list == i)
        acc_per_class += np.sum(label_list[idx] == predicted_list[idx]) / np.sum(idx)
    acc_per_class /= target_classes.shape[0]

    return acc_per_class


def get_path(image_files):
    image_files = np.squeeze(image_files)
    new_image_files = []
    for image_file in image_files:
        image_file = image_file[0]
        image_file = '/'.join(image_file.split('/')[8:])
        new_image_files.append(image_file)
    new_image_files = np.array(new_image_files)
    return new_image_files


class MyDataset(torch.utils.data.Dataset):
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


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)

    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(inputs)

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx

    return


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args)  # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                             augmix=len(set_id) > 1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        # if len(set_id) > 1:
        #     # fine-grained classification datasets
        #     classnames = eval("{}_classes".format(set_id.lower()))
        # else:
        #     assert set_id in ['A', 'R', 'K', 'V', 'I']
        #     classnames_all = imagenet_classes
        #     classnames = []
        #     if set_id in ['A', 'R', 'V']:
        #         label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
        #         if set_id == 'R':
        #             for i, m in enumerate(label_mask):
        #                 if m:
        #                     classnames.append(classnames_all[i])
        #         else:
        #             classnames = [classnames_all[i] for i in label_mask]
        #     else:
        #         classnames = classnames_all
        # if args.cocoop:
        #     model.prompt_generator.reset_classnames(classnames, args.arch)
        #     model = model.cpu()
        #     model_state = model.state_dict()
        #     model = model.cuda(args.gpu)
        # else:

        if set_id == "CUB_trainval":
            pass
        elif set_id == "CUB_test_seen":
            val_dataset = MyDataset(args.root, "CUB", data_transform, mode='seen')
        elif set_id == "CUB_test_unseen":
            val_dataset = MyDataset(args.root, "CUB", data_transform, mode='unseen')
        else:
            val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)

        model.reset_classnames(classnames, args.arch)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}".format(set_id, results[set_id][0]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    predicted, label = [], []

    if not args.cocoop:  # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        if not args.cocoop:  # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, images, optimizer, scaler, args)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(image)

        predicted.extend(list(torch.max(output.data, 1)[1].cpu().numpy()))
        label.extend(list(target.cpu().numpy()))

        # measure accuracy and record loss
        acc1 = cal_acc_per_class(predicted, label)

        top1.update(acc1[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='../../dataset')
    parser.add_argument('--test_sets', type=str, default='CUB',
                        help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False,
                        help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    main()
