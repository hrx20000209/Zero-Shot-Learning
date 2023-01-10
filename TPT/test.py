import torch
import clip
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').split('.')[1]
    return class_names


def top_k(output, target, ks=(1,)):
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = clip.load("ViT-B/32", device)
normalization = clip_preprocess.transforms[-1]

prompter = VisualPrompter(30, 30, clip_model)

preprocess = [
        transforms.Resize(164, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(164, 164))
    ]
for item in clip_preprocess.transforms[2:-1]:
    preprocess.append(item)
preprocess = transforms.Compose(preprocess)

data_root = "../../dataset/CUB_200_2011/images"

data_set = ImageFolder(data_root, preprocess)
classnames = refine_classname(data_set.classes)
text_inputs = torch.cat([clip.tokenize(f"this is a photo of a {c}") for c in classnames])
text_inputs.requires_grad = False
text_inputs = text_inputs.to(device)

data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True)

all_top1 = []
for i, (images, labels) in enumerate(tqdm(data_loader)):
    pad_length = int((224 - 164) / 2)
    pad_dim = (pad_length, pad_length, pad_length, pad_length)

    images = torch.nn.functional.pad(images, pad_dim, "constant", value=0)
    images = images.to(device)
    noise = prompter.perturbation.to(device)
    noise = noise.repeat(images.size(0), 1, 1, 1)
    noise.retain_grad()

    # Normalize the image and noise
    images = normalization(images + noise)
    # images = images + normalization(noise)
    images.require_grad = True

    save_image(images, './pic/test.jpg', normalize=True)
    probs = prompter(images, text_inputs)

    top1, top5 = top_k(probs, labels.to(device), ks=(1, 5))
    all_top1.extend(top1.cpu())
    print("Top1 accuracy: {:.4f}".format(np.mean(all_top1)))
