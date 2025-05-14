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
    def __init__(self, emb_dim=128, backbone='resnet50d', unfreeze_blocks=['layer3', 'layer4']):
        super().__init__()
        
        self.backbone = timm.create_model(backbone, pretrained=True)
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()

        for p in self.backbone.parameters(): p.requires_grad = False
        for b in unfreeze_blocks:
            for p in getattr(self.backbone, b).parameters():
                p.requires_grad = True

        feat_dim = self.backbone.num_features
        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.PReLU(), nn.Dropout(0.5),
            nn.Linear(512, emb_dim),  nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)             # [B, feat_dim]
        emb  = self.embedding_head(feat)    # [B, emb_dim]
        emb  = F.normalize(emb, p=2, dim=1) # L2-norm
        return emb
    
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