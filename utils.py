import os
import faiss
import pickle
from models import get_embedding
from settings import EMB_DIM, INFO_PATH, INDEX_PATH, GALLERY_PATH


def build_db(emb_model):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(INFO_PATH), exist_ok=True)
    
    index = faiss.IndexFlatIP(EMB_DIM)
    info  = []
        
    if len(os.listdir(GALLERY_PATH)) > 0:
        for id_name in os.listdir(GALLERY_PATH):
            pdir = os.path.join(GALLERY_PATH, id_name)
            if not os.path.isdir(pdir): 
                continue
            
            for filename in os.listdir(pdir):
                img_path = os.path.join(pdir, filename)
                pid      = id_name.split('_')[0]
                name     = id_name.split('_')[1]
                
                index.add(get_embedding(emb_model, img_path))
                info.append({'pid': pid, 'name': name, 'img_path': img_path})
            print(f"Added {id_name} to index.")

    faiss.write_index(index, INDEX_PATH)
    pickle.dump(info, open(INFO_PATH, 'wb'))
    return index, info


def remove_db():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(INFO_PATH):
        os.remove(INFO_PATH)

def get_db_info(index, info):
    if index.ntotal == len(info):
        pid  = set([i['pid'] for i in info])
        name = set([i['name'] for i in info])
        print(f"Number of registered faces: {len(pid)}")
        for x, y in zip(pid, name):
            print(f"- ID: {x}, Name: {y}")
            
    elif index == None and info == None:
        print('No faces registered.')
    else:
        print('Mismatch between index and info.')