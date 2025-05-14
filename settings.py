import torch

# Model settings
DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMG_SIZE  = 112
EMB_DIM   = 128

# Paths
IMG_PER_USER = 50
GALLERY_PATH = 'gallery/user'
INFO_PATH    = 'gallery/user/face_info.pkl'
INDEX_PATH   = 'gallery/embedding/face_index.index'
MODEL_PATH   = 'results/models/best_model.pt'

# Reference
THRESHOLD  = 0.6 # cosine threshold (calibrated)
WIN_SIZE   = 7   # number of consecutive frames
MIN_VOTES  = 5   # need ≥ 5 frames to meet threshold to be considered a match