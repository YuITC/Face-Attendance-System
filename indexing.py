import os
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import faiss



def build_index_and_info(emb_dim, index_path, info_path):
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


def get_index_and_info(index_path, info_path):
    index = faiss.read_index(index_path)
    info  = pickle.load(open(info_path,'rb'))
    return index, info