import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
from tqdm import tqdm
import time
import random
# import wandb

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import clip
from models import prompters
from PIL import Image
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from myDataset import ZeroShotDataset, TestDataset


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='ViT-B/16')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='../../dataset',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='flowers-102',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
        format(args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial)

    return args


best_acc1 = 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    global best_acc1, device

    args = parse_option()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    model, preprocess = clip.load(args.arch, device, jit=False)
    print("loading clip model: {}".format(args.arch))
    convert_models_to_fp32(model)
    model.eval()

    prompter = prompters.__dict__[args.method](args).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            prompter.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')

    # train_dataset = ZeroShotDataset(args.root, args.dataset, preprocess)

    # test_seen_dataset = ZeroShotDataset(args.root, args.dataset, preprocess, mode='seen')
    # test_unseen_dataset = ZeroShotDataset(args.root, args.dataset, preprocess, mode='unseen')

    train_dataset = TestDataset(args.root, args.dataset, preprocess)
    test_seen_dataset = TestDataset(args.root, args.dataset, preprocess, mode='seen')
    test_unseen_dataset = TestDataset(args.root, args.dataset, preprocess, mode='unseen')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)

    test_seen_loader = DataLoader(test_seen_dataset,
                                  batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False)
    test_unseen_loader = DataLoader(test_unseen_dataset,
                                    batch_size=args.batch_size, pin_memory=True,
                                    num_workers=args.num_workers, shuffle=False)

    print("build data loader successfully")

    # class_names = train_dataset.cname
    # class_names = refine_classname(class_names)
    class_names = train_dataset.classnames
    texts = [template.format(label) for label in class_names]

    # define criterion and optimizer
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()  # 用于降低训练显存的
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)  # 调整学习率

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # wandb
    if args.use_wandb:
        wandb.init(project='Visual Prompting')
        wandb.config.update(args)
        wandb.run.name = args.filename
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1 = validate(test_seen_loader, texts, model, prompter, criterion, args, "seen")
        acc2 = validate(test_unseen_loader, texts, model, prompter, criterion, args, "unseen")
        return

    epochs_since_improvement = 0

    print('==== Begin Training ====')

    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, texts, model, prompter, optimizer, scheduler, criterion,
              scaler, epoch, train_dataset, args)

        # evaluate on validation set
        acc1 = validate(test_seen_loader, texts, model, prompter, criterion, args, "seen")
        acc2 = validate(test_unseen_loader, texts, model, prompter, criterion, args, "unseen")

        h = 2 * acc1 * acc2 / (acc1 + acc2)

        # remember best acc@1 and save checkpoint
        is_best = h > best_acc1
        best_acc1 = max(h, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    wandb.run.finish()


def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, data_set, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    predicted, label = [], []
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        np_target = target.cpu().numpy()
        # gt = np.zeros(shape=(len(np_target), len(data_set.seenclasses) + len(data_set.unseenclasses)))
        gt = np.zeros(shape=(len(np_target), len(data_set.classnames_dict)))
        for j in range(len(np_target)):
            gt[j][np_target[j]] = 1
        # sampled_classes = data_set.seenclasses
        # sampled_classes, inverse_ind = np.unique(np_target, return_inverse=True)
        sampled_classes = data_set.labels
        gt = gt[:, sampled_classes]
        ground_truth = torch.from_numpy(gt).float().to(target.device)

        # with automatic mixed precision
        with autocast():
            prompted_images = prompter(images)
            output, _ = model(prompted_images, text_tokens)
            loss = criterion(output[:, sampled_classes], ground_truth)
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            predicted.extend(list(torch.max(output.data, 1)[1].cpu().numpy()))
            label.extend(list(target.cpu().numpy()))

        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = cal_acc_per_class(predicted, label)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg
                })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg


def validate(val_loader, texts, model, prompter, criterion, args, data_type):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original ' + data_type + ' Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt ' + data_type + ' Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()

    predicted_org, predicted_prompt, label = [], [], []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt, _ = model(prompted_images, text_tokens)
            output_org, _ = model(images, text_tokens)
            loss = criterion(output_prompt, target)

            predicted_org.extend(list(torch.max(output_org.data, 1)[1].cpu().numpy()))
            predicted_prompt.extend(list(torch.max(output_prompt.data, 1)[1].cpu().numpy()))
            label.extend(list(target.cpu().numpy()))

            # measure accuracy and record loss
            # acc1 = accuracy(output_prompt, target, topk=(1,))
            acc1 = cal_acc_per_class(predicted_prompt, label)
            losses.update(loss.item(), images.size(0))
            top1_prompt.update(acc1, images.size(0))

            # acc1 = accuracy(output_org, target, topk=(1,))
            acc1 = cal_acc_per_class(predicted_org, label)
            top1_org.update(acc1, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Prompt ' + data_type + ' Acc@1 {top1_prompt.avg:.3f} '.format(top1_prompt=top1_prompt),
              'Original ' + data_type + ' Acc@1 {top1_org.avg:.3f}'.format(top1_org=top1_org))

        if args.use_wandb:
            wandb.log({
                'val_loss': losses.avg,
                'val_acc_prompt': top1_prompt.avg,
                'val_acc_org': top1_org.avg,
            })

    return top1_prompt.avg


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


if __name__ == '__main__':
    main()
