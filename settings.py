import torch

DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMG_SIZE  = 112 # 
EMB_DIM   = 256
THRESHOLD = 0.9 #
# print(f'Using device: {DEVICE}')

IMG_PER_USER = 10
MODEL_PATH = 'results/models/best_model (2).pt'
INFO_PATH  = 'gallery/user/face_info.pkl'
INDEX_PATH = 'gallery/embedding/face_index.index'