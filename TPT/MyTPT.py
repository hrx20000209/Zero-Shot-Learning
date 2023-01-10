import os
import torch
import argparse
import clip
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from data.datautils import AugMixAugmenter

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# Prompter Class
class VisualPrompter(torch.nn.Module):
    def __init__(self, pad_h, pad_w, clip_model):
        super().__init__()
        self.mask = torch.ones((3, 224, 224))
        self.mask[:, pad_h: 224 - pad_h, pad_w: 224 - pad_w] = 0

        delta = torch.zeros((3, 224, 224))
        delta.require_grad = True

        self.perturbation = torch.nn.Parameter(
            delta.float(), requires_grad=True)
        self.model = clip_model

    def forward(self, images, text_inputs):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(text_inputs)
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # probs = (self.model.logit_scale.exp() * norm_image_features @ norm_text_features.T)
        probs = (100.0 * norm_image_features @ norm_text_features.T).softmax(dim=-1)

        return probs


# Utils Function

def top_k(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name in skip_list:
            print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}]


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').split('.')[1]
    return class_names


def select_confident_samples(logit, top):
    batch_entropy = -(logit.softmax(1) * logit.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logit[idx], idx


def avg_entropy(outputs):
    logit = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logit = outputs.log_softmax(dim=1) [N, 1000]
    avg_logit = logit.logsumexp(dim=0) - np.log(logit.shape[0])  # avg_logit = logit.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logit.dtype).min
    avg_logit = torch.clamp(avg_logit, min=min_real)
    return -(avg_logit * torch.exp(avg_logit)).sum(dim=-1)


# Parser
def add_parser():
    parser = argparse.ArgumentParser("Test Time Visual Prompting Tuning")

    # base
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # dataset
    parser.add_argument("--data_root", type=str, default='../../dataset', help="the directory of dataset")
    parser.add_argument("--dataset", type=str, default='CUB_200_2011', help="the name of dataset")
    parser.add_argument("--image_size", type=int, default=164, help="image size")

    # model
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help='clip model')
    parser.add_argument("--prompt_size", type=int, default=30, help="size for visual prompts")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="learning rate")
    parser.add_argument('--selection_threshold', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')

    return parser.parse_args()


def build_dataset(argus, data_transform):
    data_dir = os.path.join(argus.data_root, argus.dataset)
    if argus.dataset == 'CUB_200_2011':
        data_dir = os.path.join(data_dir, 'images')
    data_set = ImageFolder(data_dir, data_transform)
    return data_set


def test_time_prompt_tuning(data_loader, text, prompt, norm, optim):
    all_loss = []
    all_top1 = []
    lr = optim.param_groups[0]["lr"]

    for i, (images, labels) in enumerate(tqdm(data_loader)):
        # Pad the Image
        batch_images = images[0]
        for j in range(1, len(images)):
            batch_images = torch.cat((batch_images, images[j]), 0)
        batch_images = torch.nn.functional.pad(batch_images, pad_dim, "constant", value=0)
        batch_images = batch_images.to(device)

        # Normalize the image and noise
        noise = prompt.perturbation.to(device)
        noise = noise.repeat(batch_images.size(0), 1, 1, 1)
        noise.retain_grad()
        batch_images = norm(batch_images + noise)
        batch_images.require_grad = True

        # save_image(batch_images, './pic/test_2.jpg', normalize=True)
        probs = prompt(batch_images, text)

        # Confidence selection
        # selected_probs, selected_idx = select_confident_samples(probs, args.selection_threshold)
        ground_truth = torch.cat(tuple(labels.to(device) for _ in range(probs.size(0))), 0)
        loss = criterion(probs, ground_truth)
        loss.backward()

        # Update the perturbation
        grad_p_t = noise.grad
        grad_p_t = grad_p_t.mean(0).squeeze(0)
        g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
        scaled_g = grad_p_t / (g_norm + 1e-10)
        scaled_g_pad = scaled_g * prompt.mask.to(device)
        updated_pad = scaled_g_pad * lr
        prompt.perturbation.data = prompt.perturbation.data - updated_pad.detach().cpu()
        prompt.zero_grad()

        optim.step()

        all_loss.append(loss.detach().cpu().numpy())
        top1, top5 = top_k(probs, labels.to(device), ks=(1, 5))
        all_top1.extend(top1.cpu())
        print("Top1 accuracy: {:.4f}, Loss: {:.4f}".format(np.mean(all_top1), loss))

    return np.mean(all_loss), np.mean(all_top1)


if __name__ == '__main__':
    args = add_parser()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = clip.load(args.clip_model, device)
    normalization = clip_preprocess.transforms[-1]

    preprocess = [
        transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(args.image_size, args.image_size))
    ]
    for item in clip_preprocess.transforms[2:-1]:
        preprocess.append(item)
    preprocess = transforms.Compose(preprocess)

    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    data_augment = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1, augmix=True)

    # Build dataset
    print("evaluating: {}".format(args.dataset))
    val_dataset = build_dataset(args, data_augment)
    total_step = len(val_dataset)
    print("number of test samples: {}".format(total_step))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    classnames = val_dataset.classes
    classnames = refine_classname(classnames)
    text_inputs = torch.cat([clip.tokenize(f"this is a photo of a {c}") for c in classnames])
    text_inputs.requires_grad = False
    text_inputs = text_inputs.to(device)

    # Build visual prompter model
    prompter = VisualPrompter(args.prompt_size, args.prompt_size, clip_model)
    prompter.model.requires_grad_(False)
    param_groups = add_weight_decay(prompter, 0.0, skip_list=("perturbation"))
    print(param_groups)

    pad_length = int((224 - args.image_size) / 2)
    pad_dim = (pad_length, pad_length, pad_length, pad_length)

    # Create optimizer
    optimizer = torch.optim.SGD(param_groups, lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_top1 = test_time_prompt_tuning(val_loader, text_inputs, prompter, normalization, optimizer)
