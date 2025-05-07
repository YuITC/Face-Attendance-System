import os
import time
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cv2
import faiss
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from settings import *
from embedding import EmbeddingModel, load_model, get_embedding
from indexing  import get_index_and_info, build_index_and_info


def detect_face(face_detector):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return

    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print('Error: Could not capture frame.')
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detect face
        bbox, prob = face_detector.detect(frame)
        draw_frame = frame.copy()
        
        if bbox is not None:
            for i, (box, confidence) in enumerate(zip(bbox, prob)):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_frame, f"{confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                  
        cv2.imshow('Face Detection ([ESC] to escape) (Test camera)', draw_frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
def add_face(pid, name, face_detector, emb_model):
    index, info = get_index_and_info(INDEX_PATH, INFO_PATH)

    if pid in [i['pid'] for i in info]:
        print(f"Person ID {pid} already exists. Please use a different ID.")
        return
    
    else:
        face_cnt  = 0
        face_embs = []
        pdir      = os.path.join('gallery/user', pid)
        os.makedirs(pdir, exist_ok=True)
        
        
        cap = cv2.VideoCapture(0)
        while face_cnt < 10:
            ret, frame = cap.read()
            frame      = cv2.flip(frame, 1)
            if not ret: break
            
            bbox, _  = face_detector.detect(frame)
            if bbox is not None and len(bbox) == 1:
                x1, y1, x2, y2 = [int(v) for v in bbox[0]]
                draw_frame     = frame.copy()
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img   = Image.fromarray(rgb_frame) 
                face_img  = face_detector.extract(pil_img, bbox, save_path=os.path.join(pdir, f"{os.path.basename(pdir)}_{face_cnt:03d}.jpg"))
                face_cnt += 1
                
                if face_img is not None:
                    print(f"Extracted face {face_cnt} from {pid}.")
                    face_embs.append(get_embedding(emb_model, face_img))
            
            time.sleep(0.05)
            
            cv2.imshow('Face Registration', draw_frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        
        for face_emb in face_embs:
            info.append({'pid': pid, 'name': name, 'embedding': face_emb})
        pickle.dump(info, open(INFO_PATH, 'wb'))
        
        face_embs = np.concatenate(face_embs, axis=0)
        index.add(face_embs)
        faiss.write_index(index, INDEX_PATH)
        
        print(f"Added person ID {pid} to index.")
        # return info, index
    
    
def remove_face(pid):
    index, info = get_index_and_info(INDEX_PATH, INFO_PATH)
    
    if pid not in [i['pid'] for i in info]:
        print(f"Person ID {pid} does not exist.")
        return
    
    else:
        pdir = os.path.join('gallery/user', pid)
        if os.path.exists(pdir):
            for file in os.listdir(pdir):
                os.remove(os.path.join(pdir, file))
            os.rmdir(pdir)
        
        info  = [i for i in info if i['pid'] != pid]
        index = faiss.IndexFlatIP(128)
        for i in info:
            index.add(i['embedding'].reshape(1, -1))
        faiss.write_index(index, INDEX_PATH)
        pickle.dump(info, open(INFO_PATH, 'wb'))
        
        print(f"Removed person ID {pid} from index.")
        # return info, index
    
    
def search_face(face_detector, emb_model):
    index, info = get_index_and_info(INDEX_PATH, INFO_PATH)
    cap         = cv2.VideoCapture(0)
    if index.ntotal == 0:
        print("No faces registered in the system.")
        return
    
    while True:
        ret, frame = cap.read()
        frame      = cv2.flip(frame, 1)
        if not ret: break

        bbox, _    = face_detector.detect(frame)
        draw_frame = frame.copy()
        if bbox is not None and len(bbox) == 1:
            x1, y1, x2, y2 = [int(v) for v in bbox[0]]
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img    = Image.fromarray(rgb_frame) 
            face_img   = face_detector.extract(pil_img, bbox, save_path=None)
            
            face_emb   = get_embedding(emb_model, face_img)
            D, I       = index.search(face_emb, 1)
            score, idx = D[0][0], I[0][0]
            
            if score >= THRESHOLD:
                name = info[idx]['name']
                cv2.putText(draw_frame, f"{name} ({score:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(draw_frame, 'Unknown', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        time.sleep(0.05)
        cv2.imshow('Face Recognition', draw_frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()