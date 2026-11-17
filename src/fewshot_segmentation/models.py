"""ResNet-50 feature extractor — SSP paper (layer3) + 512-dim projection."""
import torch
import torch.nn as nn
from torchvision import models

from .config import FEATURE_DIM, device, PROJECT_ROOT


def freeze_bn(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        if module.weight is not None:
            module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False


class SSPResNet50(nn.Module):
    """ResNet-50 up to layer3 + 1x1 projection to FEATURE_DIM."""

    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=None)
        checkpoint_path = (
            PROJECT_ROOT / "checkpoints" / "pretrained" / "resnet50-11ad3fa6.pth"
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        resnet.load_state_dict(state_dict)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # 1024 → FEATURE_DIM (512) projection
        self.proj = nn.Sequential(
            nn.Conv2d(1024, FEATURE_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(FEATURE_DIM),
            nn.ReLU(inplace=True),
        )
        self._init_proj()

    def _init_proj(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)          # [B, 1024, H/16, W/16]
        return self.proj(f3)          # [B, 512, H/16, W/16]


def create_backbone():
    model = SSPResNet50()

    # Freeze stem + layer1 + layer2
    for module in [model.conv1, model.bn1, model.layer1, model.layer2]:
        for p in module.parameters():
            p.requires_grad = False

    # Train layer3 + projection
    for module in [model.layer3, model.proj]:
        for p in module.parameters():
            p.requires_grad = True

    model.apply(freeze_bn)
    return model.to(device)


def extract_feature(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feature = model(image_tensor)
    return feature.squeeze(0)