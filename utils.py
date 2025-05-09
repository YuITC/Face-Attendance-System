import os
import faiss
import pickle


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


def get_db_info(index, info):
    if index.ntotal == len(info):
        pids  = set([i['pid'] for i in info])
        names = set([i['name'] for i in info])
        print(f"Number of registered faces: {len(pids)}")
        for pid, name in zip(pids, names):
            print(f"- ID: {pid}, Name: {name}")
    else:
        print('Mismatch between index and info.')