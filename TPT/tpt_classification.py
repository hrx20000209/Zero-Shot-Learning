import argparse
import os
import time

from copy import deepcopy

from PIL import Image
import numpy as np

import clip
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.utils import save_image

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

import sys

sys.path.append('..')
from visual_prompting.models import prompters

sys.path.append('..')
from EVP.main import Pertubation

sys.path.append('../..')
from dataset.CUB_200_2011.classes import cub_classes

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class VisualPrompter(torch.nn.Module):
    def __init__(self, prompt_size, image_size):
        super(VisualPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size

        self.base_size = image_size - pad_size * 2
        self.pad_up = torch.nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = torch.nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = torch.nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))
        self.pad_right = torch.nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


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
            if args.modal_choice == 2:
                inputs = model.visual_prompter(inputs)
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


def test_time_tuning_visual(model, inputs, args):
    inputs = model.visual_prompter(inputs)
    output = model(inputs)
    output, selected_idx = select_confident_samples(output, args.selection_p)
    return selected_idx


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
        if args.modal_choice == 1:
            print("=> Train text prompt in test time")
            trainable_param = model.prompt_learner.parameters()
            optimizer = torch.optim.AdamW(trainable_param, args.lr)
            optim_state = deepcopy(optimizer.state_dict())
        elif args.modal_choice == 2:
            print("=> Train visual prompt in test time")

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
        if len(set_id) > 1:
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, scaler, args, classnames, model)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for i in results.keys():
        print("{}".format(i), end="	")
    print("\n")
    for i in results.keys():
        print("{:.2f}".format(results[i][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, model_state, scaler, args, class_names, clip_model=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop:  # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()

    device = 'cuda'
    my_model, _, _ = clip.load("ViT-B/32", device=device, download_root='~/.cache/clip')
    # my_model = load('ViT-B/32', device, jit=False)
    convert_models_to_fp32(my_model)
    my_model.eval()

    visual_prompter = VisualPrompter(30, 224).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    visual_prompter.train()

    optimizer = torch.optim.SGD(visual_prompter.parameters(),
                                lr=40,
                                momentum=0.9,
                                weight_decay=0)
    scheduler = cosine_lr(optimizer, 40, 1000, len(val_loader))
    result_image = None
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
            # save_image(image, './pic/test_2.jpg', normalize=True)

        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        # if not args.cocoop:  # no need to reset cocoop because it's fixed
        #     if args.tta_steps > 0:
        #         with torch.no_grad():
        #             model.reset()
        #     optimizer.load_state_dict(optim_state)
        #     test_time_tuning(model, images, optimizer, scaler, args)
        # else:
        #     with torch.no_grad():
        #         with torch.cuda.amp.autocast():
        #             image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
        #     optimizer = None
        #     pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)
        #
        # # The actual inference goes here
        # if args.tpt:
        #     if args.cocoop:
        #         image_feature = image_feature[0].unsqueeze(0)
        #
        # with torch.no_grad():
        #     with torch.cuda.amp.autocast():
        #         if args.cocoop:
        #             output = model((image_feature, pgen_ctx))
        #         else:
        #             output = model(image)

        if args.modal_choice == 2:
            optimizer.zero_grad()

            scheduler(i)
            # text_tokens = model.text_encoder(target, model.prompt_learner.tokenized_prompts).to('cuda:0')
            template = 'This is a photo of a {}'
            texts = [template.format(label) for label in class_names]

            images = images.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            # with torch.cuda.amp.autocast():
            idx = test_time_tuning_visual(model, images, args)
            if i == 0:
                result_image = images
                save_image(visual_prompter(result_image), './pic/prompted_image.png')
            for j in range(len(idx)):
                item = images[idx[j]]
                # save_image(item, './pic/test_{}.png'.format(j))
                prompted_image = visual_prompter(item)
                with torch.no_grad():
                    image_features = my_model.encode_image(prompted_image)
                    text_features = my_model.encode_text(text_tokens)

                    logits_per_image, logits_per_text = my_model(prompted_image, text_tokens)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                # output, _ = my_model(prompted_image, text_tokens)
                ground_truth = torch.cat((target, target, target), 0)
                # print(ground_truth)
                loss = criterion(logits_per_image, ground_truth)
                # print(loss)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
            # scaler.update()

            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    result = visual_prompter(result_image)
    save_image(prompted_image, './pic/image_result.png')

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I',
                        help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
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
    parser.add_argument('--modal_choice', type=int, default=1, help='Tuning method choice(default: 1): '
                                                                    '1.text 2.visual 3.both')

    # visual prompt argument
    parser.add_argument('--vp_root', default=None, type=str, help='path to a trained visual prompter')
    parser.add_argument('--prompt_size', type=int, default=30, help='size for visual prompts')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')

    # EVP argument
    parser.add_argument('--is_EVP', action='store_true', default=False, help='use EVP method')

    main()
