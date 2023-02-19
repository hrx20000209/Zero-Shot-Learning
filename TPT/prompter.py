import torch


class Pertubation(torch.nn.Module):
    def __init__(self, pad_h, pad_w, clip_model):
        super().__init__()
        self.mask = torch.ones((3, 224, 224))
        self.mask[:, pad_h: 224 - pad_h, pad_w: 224 - pad_w] = 0

        delta = torch.zeros((3, 224, 224))
        self.init_state = delta.detach().clone()
        delta.require_grad = True

        self.perturbation = torch.nn.Parameter(delta.float(), requires_grad=True)
        self.model = clip_model

    def forward(self, images, text_inputs):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(text_inputs)
        norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        probs = self.model.logit_scale.exp() * norm_image_features @ norm_text_features.T

        return probs

    def reset(self):
        tmp = self.init_state
        self.perturbation.copy_(tmp)

    def add_weight_decay(self, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if name in skip_list:
                print(name)
                no_decay.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.}]


class PadPrompter(torch.nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.base_size = image_size - pad_size * 2
        self.pad_up = torch.nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = torch.nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = torch.nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))
        self.pad_right = torch.nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))

        self.init_state_pad_up = self.pad_up.detach().clone()
        self.init_state_pad_down = self.pad_down.detach().clone()
        self.init_state_pad_left = self.pad_left.detach().clone()
        self.init_state_pad_right = self.pad_right.detach().clone()

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt

    def reset(self):
        tmp = self.init_state_pad_up
        self.pad_up.copy_(tmp)
        tmp = self.init_state_pad_down
        self.pad_down.copy_(tmp)
        tmp = self.init_state_pad_left
        self.pad_left.copy_(tmp)
        tmp = self.init_state_pad_right
        self.pad_right.copy_(tmp)
