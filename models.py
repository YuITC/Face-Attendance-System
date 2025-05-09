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


class EmbeddingModel(nn.Module):
    def __init__(self, backbone='efficientnet_b0', n_layers_unfreeze=10):
        super().__init__()
        
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.backbone.classifier = nn.Identity()

        for p in self.backbone.parameters():                            p.requires_grad = False
        for p in list(self.backbone.parameters())[-n_layers_unfreeze:]: p.requires_grad = True

        feat_dim = self.backbone.num_features
        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 256)     , nn.SiLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
            nn.Linear(256, 128)     , nn.SiLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        feat = self.backbone(x)
        emb  = self.embedding_head(feat)
        emb  = F.normalize(emb, p=2, dim=1)
        return emb
    
    
def load_model(model, model_path, device):
    state = torch.load(model_path, map_location=device)

    model.backbone.load_state_dict(state['model_backbone'])
    model.embedding_head.load_state_dict(state['model_embedding'])
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