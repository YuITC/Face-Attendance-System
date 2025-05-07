import os
import torch
from torchvision import transforms


DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMG_SIZE  = 112
EMB_DIM   = 128
THRESHOLD = 0.7
print(f'Using device: {DEVICE}')


TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


MODEL_PATH = 'results/models/best_model.pt'
INDEX_PATH = 'gallery/embedding/face_index.index'
INFO_PATH  = 'gallery/user/face_info.pkl'
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
os.makedirs(os.path.dirname(INFO_PATH), exist_ok=True)