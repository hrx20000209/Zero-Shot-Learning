import argparse
import pdb

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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.utils import save_image

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
from clip import load, tokenize
from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed

from prompter import Pertubation

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
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            output = model(inputs)
            output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

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
    if args.modal == 1:
        trainable_param = model.prompt_learner.parameters()
    else:
        # prepare the visual prompter
        clip_model, _, _ = load("ViT-B/16", "cuda")
        prompter = Pertubation(args.prompt_size, args.prompt_size, clip_model)
        prompter.model.requires_grad_(False)
        trainable_param = prompter.add_weight_decay(0.0, skip_list=("perturbation"))
        print(trainable_param)

        pad_length = int((224 - args.image_size) / 2)
        pad_dim = (pad_length, pad_length, pad_length, pad_length)

    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')
    if args.modal == 1:
        modal = "Text"
    else:
        modal = "Visual"
    print('Modal: {},  Learning Rate: {}'.format(modal, args.lr))

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.modal == 2:
            image_size = args.image_size
            preprocess = transforms.Compose([
                transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            image_size = args.resolution
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            # image_size = args.image_size
            # preprocess = transforms.Compose([
            #     transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC),
            #     transforms.ToTensor(),
            #     normalize
            # ])
        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=BICUBIC),
                transforms.CenterCrop(image_size)])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1, augmix=False)
            # augmix=len(set_id) > 1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        classnames = val_dataset.cname

        if args.modal == 2:
            text_inputs = torch.cat([tokenize(f"this is a photo of a {c}") for c in classnames])
            text_inputs.requires_grad = False
            text_inputs = text_inputs.cuda(args.gpu, non_blocking=True)

        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        print("evaluating: {}".format(set_id))
        model.reset_classnames(classnames, args.arch)

        if args.modal == 2:
            results[set_id] = test_time_adapt_eval(args, val_loader, model, model_state, optimizer, optim_state,
                                                   scaler, prompter, normalize, text_inputs, pad_dim)
        else:
            results[set_id] = test_time_adapt_eval(args, val_loader, model, model_state, optimizer,
                                                   optim_state, scaler)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}".format(set_id, results[set_id][0]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep \t lr \t batch_size")
    print("params: {} \t {} \t {}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(args, val_loader, model, model_state, optimizer, optim_state, scaler,
                         prompt=None, norm=None, text=None, pad_dim=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.4f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    end = time.time()

    # TensorBoard setting
    writer = SummaryWriter(log_dir="/root/tf-logs", flush_secs=60)

    # Loss
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    predicted, label = [], []
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
            origin_images = images

        if args.modal == 2:
            if args.tta_steps > 0:
                with torch.no_grad():
                    prompt.reset()

            # Pad the image
            pad_images = F.pad(images, pad_dim, "constant", value=0)
            pad_images = pad_images.cuda(args.gpu, non_blocking=True)

            # Normalize the image and noise
            noise = prompt.perturbation.cuda(args.gpu, non_blocking=True)
            noise = noise.repeat(images.size(0), 1, 1, 1)
            noise.retain_grad()
            batch_images = norm(pad_images + noise)
            batch_images.require_grad = True

            # reset the tunable prompt to its initial state
            optimizer.load_state_dict(optim_state)

            output = prompt(batch_images, text)

            batch_target = target.repeat(len(output))

            save_image(batch_images, './pic/selected_prompted_image.jpg', normalize=True)

            # loss
            loss = criterion(output, batch_target)
            loss.backward()

            # update the perturbation
            grad_p_t = noise.grad
            grad_p_t = grad_p_t.mean(0).squeeze(0)
            g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
            scaled_g = grad_p_t / (g_norm + 1e-10)
            scaled_g_pad = scaled_g * prompt.mask.cuda(args.gpu, non_blocking=True)
            updated_pad = scaled_g_pad * args.lr
            prompt.perturbation.data = prompt.perturbation.data - updated_pad.detach().cpu()
            prompt.zero_grad()

            # measure accuracy and record loss
            output, selected_idx = select_confident_samples(output, args.selection_p)

            # the actual inference
            pad_image = pad_images[0].unsqueeze(0)
            noise = prompt.perturbation.cuda(args.gpu, non_blocking=True)
            noise = noise.repeat(1, 1, 1)
            noise.retain_grad()
            inference_image = norm(pad_image + noise)

            output = prompt(inference_image, text)
            save_image(inference_image.squeeze(0), './inference_image.jpg', normalize=True)

            writer.add_scalar(tag='Loss', scalar_value=loss, global_step=i)
            writer.add_image('img/prompted_img', inference_image.squeeze(0), i)
        else:
            # reset the tunable prompt to its initial state
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()

            # images = F.pad(images, (30, 30, 30, 30), "constant", value=0)
            # save_image(images, './pic/paded_image_1.jpg', normalize=True)

            optimizer.load_state_dict(optim_state)

            test_time_tuning(model, images, optimizer, scaler, args)

            # image = F.pad(image, (30, 30, 30, 30), "constant", value=0)
            # save_image(image, './pic/paded_image_2.jpg', normalize=True)

            # The actual inference goes here
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(image)

        # measure accuracy and record loss
        predicted.extend(list(torch.max(output.data, 1)[1].cpu().numpy()))
        label.extend(list(target.cpu().numpy()))

        acc1 = cal_acc_per_class(predicted, label) * 100
        top1.update(acc1, image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()
    writer.close()

    return [top1.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I',
                        help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='seen', help='which split to use: seen/unseen')
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

    parser.add_argument("--prompt_size", type=int, default=30, help="size for visual prompts")
    parser.add_argument("--image_size", type=int, default=164, help="image size")
    parser.add_argument("--modal", type=int, default=1, help="Prompt modal choice: 1.text 2.visual")

    main()
