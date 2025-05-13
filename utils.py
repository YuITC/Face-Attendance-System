import os
import faiss
import pickle
import numpy as np
from models import get_embedding
from settings import IMG_SIZE, EMB_DIM, INFO_PATH, INDEX_PATH


def build_db(emb_dim, index_path, info_path):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatIP(emb_dim)
        faiss.write_index(index, index_path)
        
    if os.path.exists(info_path):
        info = pickle.load(open(info_path,'rb'))
    else:
        info = []
        pickle.dump(info, open(info_path,'wb'))

    return index, info


def build_db_from_images(index, info, image_folder, emb_model):
    pid      = os.path.basename(image_folder)
    name     = f"User_{pid}"

    face_embs = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        face_embs.append(get_embedding(emb_model, img_path))
        
    for face_emb in face_embs:
        info.append({'pid': pid, 'name': name, 'embedding': face_emb})
    pickle.dump(info, open(INFO_PATH, 'wb'))
    index.add(np.concatenate(face_embs, axis=0))
    faiss.write_index(index, INDEX_PATH)
    print(f"Added person ID {pid} to index.")


def get_db_info(index, info):
    if index.ntotal == len(info):
        pids  = set([i['pid'] for i in info])
        names = set([i['name'] for i in info])
        print(f"Number of registered faces: {len(pids)}")
        for pid, name in zip(pids, names):
            print(f"- ID: {pid}, Name: {name}")
    else:
        print('Mismatch between index and info.')