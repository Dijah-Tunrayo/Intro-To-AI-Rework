import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

PLANTVILLAGE_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/leaf_cnn.pt"))
CLASSES_PATH = Path(os.environ.get("CLASSES_PATH", "models/classes.json"))
IMAGE_SIZE = 128

preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class FINALCNN(nn.Module):
    """Must stay byte-for-byte identical to the FINALCNN trained in Section 4 -
    torch.load only restores weights, not architecture, so any mismatch here
    causes load_state_dict to fail (or worse, load into the wrong layers)."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int) -> nn.Module:
    return FINALCNN(num_classes=num_classes)


@lru_cache(maxsize=1)
def load_model() -> tuple[nn.Module, list[str], bool]:
    """Return (model, classes, is_trained)."""
    classes = PLANTVILLAGE_CLASSES
    if CLASSES_PATH.exists():
        classes = json.loads(CLASSES_PATH.read_text())

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(checkpoint, dict) and "classes" in checkpoint:
            classes = checkpoint["classes"]
        state = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        net = build_model(len(classes))
        net.load_state_dict(state)
        net.eval()
        return net, classes, True

    # No checkpoint yet - FINALCNN is trained from scratch (unlike MobileNetV2,
    # there's no ImageNet-pretrained version of this custom architecture to
    # fall back to), so this is an untrained model. trained=False downstream
    # tells app.py to flag this clearly in the response instead of pretending
    # it's a real prediction.

    net = build_model(len(classes))
    net.eval()
    return net, classes, False


@torch.inference_mode()
def predict(image: Image.Image, top_k: int = 3) -> tuple[list[tuple[str, float]], bool]:
    net, classes, is_trained = load_model()
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    probs = torch.softmax(net(tensor), dim=1)[0]
    k = min(top_k, probs.numel())
    scores, idx = torch.topk(probs, k)
    return [(classes[int(i)], float(s)) for s, i in zip(scores, idx)], is_trained
