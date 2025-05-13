import cv2
import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from settings import DEVICE, IMG_SIZE


# class EmbeddingModel(nn.Module):
#     def __init__(self, backbone='efficientnet_b0', n_layers_unfreeze=10):
#         super().__init__()
        
#         self.backbone = timm.create_model(backbone, pretrained=True)
#         self.backbone.classifier = nn.Identity()

#         for p in self.backbone.parameters():                            p.requires_grad = False
#         for p in list(self.backbone.parameters())[-n_layers_unfreeze:]: p.requires_grad = True

#         feat_dim = self.backbone.num_features
#         self.embedding_head = nn.Sequential(
#             nn.Linear(feat_dim, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
#             nn.Linear(512, 256)     , nn.SiLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
#             nn.Linear(256, 128)     , nn.SiLU(), nn.BatchNorm1d(128),
#             nn.Linear(128, 128)
#         )

#     def forward(self, x):
#         feat = self.backbone(x)
#         emb  = self.embedding_head(feat)
#         emb  = F.normalize(emb, p=2, dim=1)
#         return emb


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        out  = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out  = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out  = F.relu(out, inplace=True)
        out  = self.dropout(out)
        return out


class EmbeddingModel(nn.Module):
    def __init__(self, emb_dim=256, dropout_block=0.1, dropout_head=0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = self._make_layer( BasicBlock, in_planes=64,  planes=64,  num_blocks=2, stride=1, dropout=dropout_block)
        self.layer2 = self._make_layer( BasicBlock, in_planes=64,  planes=128, num_blocks=2, stride=2, dropout=dropout_block)
        self.layer3 = self._make_layer( BasicBlock, in_planes=128, planes=256, num_blocks=2, stride=2, dropout=dropout_block)
        self.layer4 = self._make_layer( BasicBlock, in_planes=256, planes=512, num_blocks=2, stride=2, dropout=dropout_block)
        self.layer5 = self._make_layer( BasicBlock, in_planes=512, planes=1024,num_blocks=2, stride=2, dropout=dropout_block)

        self.pool   = nn.AdaptiveAvgPool2d((1,1))

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * BasicBlock.expansion, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_head),
            nn.Linear(512, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

    def _make_layer(self, block, in_planes, planes, num_blocks, stride, dropout):
        layers = []
        layers.append(block(in_planes, planes, stride=stride, dropout_prob=dropout))
        for _ in range(1, num_blocks):
            layers.append(block(planes * block.expansion, planes, stride=1, dropout_prob=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)   # [B, 64, 56,56]
        x = self.layer1(x) # [B, 64, 56,56]
        x = self.layer2(x) # [B,128, 28,28]
        x = self.layer3(x) # [B,256, 14,14]
        x = self.layer4(x) # [B,512, 7,7]
        x = self.layer5(x) # [B,1024,3,3]  (112 → 56 →28→14→7→3)
        x = self.pool(x)   # [B,1024,1,1]
        emb = self.embedding_head(x)  # [B, emb_dim]
        emb = F.normalize(emb, p=2, dim=1)
        return emb
    
    
# def load_model(model, model_path, device):
#     state = torch.load(model_path, map_location=device)

#     model.backbone.load_state_dict(state['model_backbone'])
#     model.embedding_head.load_state_dict(state['model_embedding'])
#     model.to(device)
#     return model.eval()

def load_model(model, model_path, device):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state'])
    model.to(device)
    return model.eval()


def get_embedding(emb_model, img_input):
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, Image.Image):
        img = img_input.convert('RGB')
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
    elif isinstance(img_input, torch.Tensor):
        img = to_pil_image(img_input.cpu())
    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")
        
    base_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img = base_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = emb_model(img).cpu().numpy()
    return emb.reshape(1, -1).astype(np.float32)