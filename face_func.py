import os
import time
import pickle
import argparse

import cv2
import numpy as np
from PIL import Image
import faiss
from facenet_pytorch import MTCNN

from settings import INFO_PATH, INDEX_PATH, THRESHOLD, DEVICE, IMG_SIZE, MODEL_PATH, EMB_DIM, IMG_PER_USER
from models   import EmbeddingModel, load_model, get_embedding
from utils    import build_db, get_db_info


def detect_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print('Error: Could not open webcam.'); return
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        frame      = cv2.flip(frame, 1)
        if not ret: print('Error: Could not capture frame.'); break

        cur_time = time.time()
        fps      = 1 / (cur_time - prev_time) if prev_time > 0 else 0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        bbox, _    = detector.detect(frame)
        draw_frame = frame.copy()
        if bbox is not None and len(bbox) == 1:
            x1, y1, x2, y2 = [int(v) for v in bbox[0]]
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(draw_frame, 'Face Detected', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        time.sleep(0.03)
        cv2.imshow('Face Detection ([ESC] for escape)', draw_frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def add_face(pid, name, index, info):
    if pid in [i['pid'] for i in info]: print(f"ID {pid} already exists."); return
    else:
        face_cnt  = 0
        face_embs = []
        pdir      = os.path.join('gallery/user', pid)
        os.makedirs(pdir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        while face_cnt < IMG_PER_USER:
            ret, frame = cap.read()
            frame      = cv2.flip(frame, 1)
            if not ret: break

            bbox, _    = detector.detect(frame)
            draw_frame = frame.copy()
            if bbox is not None and len(bbox) == 1:
                x1, y1, x2, y2 = [int(v) for v in bbox[0]] # bbox[0] = [x1 y1 x2 y2]
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_frame, f"Registering {pid} ({face_cnt+1}/{IMG_PER_USER})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Face Registration ([Space] for capture, [ESC] for escape)', draw_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 32: # Space key
                if bbox is not None and len(bbox) == 1:
                    face_img = detector.extract(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), bbox,
                        save_path=os.path.join(pdir, f"{os.path.basename(pdir)}_{face_cnt:04d}.jpg")
                    )
                    if face_img is not None:
                        print(f"Extracted face {face_cnt+1} from user {pid}.")
                        face_embs.append(get_embedding(embedder, face_img))
                        face_cnt += 1
                    else:
                        print(f"Failed to extract face {face_cnt+1} from user {pid}.")
                else:
                    print('No face detected.')
            elif key == 27:
                break
            
        cap.release()
        cv2.destroyAllWindows()

        for face_emb in face_embs:
            info.append({'pid': pid, 'name': name, 'embedding': face_emb})
        pickle.dump(info, open(INFO_PATH, 'wb'))
        index.add(np.concatenate(face_embs, axis=0))
        faiss.write_index(index, INDEX_PATH)
        print(f"Added person ID {pid} to index.")


def remove_face(pid, index, info):
    if pid not in [i['pid'] for i in info]: print(f"ID {pid} does not exist."); return
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
    
    
def search_face(index, info, thresh=THRESHOLD):
    cap = cv2.VideoCapture(0)
    if index.ntotal == 0: print('No faces registered in the system.'); return
    
    while True:
        ret, frame = cap.read()
        frame      = cv2.flip(frame, 1)
        if not ret: break

        bbox, _    = detector.detect(frame)
        draw_frame = frame.copy()
        if bbox is not None and len(bbox) == 1:
            x1, y1, x2, y2 = [int(v) for v in bbox[0]]
            
            face_img = detector.extract(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), bbox,
                save_path=None
            )
            if face_img is not None:
                face_emb   = get_embedding(embedder, face_img)
                D, I       = index.search(face_emb, 1)
                score, idx = D[0][0], I[0][0]
                
                if score >= thresh:
                    name = info[idx]['name']
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(draw_frame, f"{name} ({score:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(draw_frame, 'Unknown', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        time.sleep(0.03)
        cv2.imshow('Face Recognition', draw_frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    index, info = build_db(EMB_DIM, INDEX_PATH, INFO_PATH)
    embedder    = load_model(EmbeddingModel(), MODEL_PATH, DEVICE)
    detector    = MTCNN(image_size=IMG_SIZE, margin=10, keep_all=False, post_process=False, device=DEVICE)
    
    
    parser = argparse.ArgumentParser(description='Face Attendance System')
    parser.add_argument('--mode'  , type=str, choices=['detect', 'add', 'remove', 'recognize', 'overview'], required=True, help='Mode: detect, add, remove, recognize face, or view database information')
    parser.add_argument('--pid'   , type=str, help='Person ID')
    parser.add_argument('--name'  , type=str, help='Person name')
    parser.add_argument('--thresh', type=float, default=THRESHOLD, help='Threshold for face recognition')
    args = parser.parse_args()
 
    
    if args.mode == 'detect':
        detect_face()
    
    elif args.mode == 'add':
        if args.pid is None or args.name is None:
            print('Please provide both --pid and --name for adding a face.')
        else:
            add_face(args.pid, args.name, index, info)
            
    elif args.mode == 'remove':
        if args.pid is None:
            print('Please provide --pid for removing a face.')
        else:
            remove_face(args.pid, index, info)
            
    elif args.mode == 'recognize':
        search_face(index, info, args.thresh)
        
    elif args.mode == 'overview':
        index, info = build_db(EMB_DIM, INDEX_PATH, INFO_PATH)
        get_db_info(index, info)