import os
import time
import argparse
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image
import faiss
from facenet_pytorch import MTCNN

from settings import IMG_PER_USER, MODEL_PATH, DEVICE, IMG_SIZE, THRESHOLD, WIN_SIZE, MIN_VOTES
from models   import EmbeddingModel, load_model, get_embedding
from utils    import build_db, get_db_info, remove_db



def detect_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print('Error: Could not open webcam.'); return
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        frame      = cv2.flip(frame, 1)
        if not ret: print('Error: Could not capture frame.'); break

        cur_time  = time.time()
        fps       = 1 / (cur_time - prev_time) if prev_time > 0 else 0
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


def add_face(pid, name, index, info, emb_model):
    if pid in [i['pid'] for i in info]: print(f"ID {pid} already exists."); return
    else:
        face_cnt = 0
        pdir     = os.path.join('gallery/user', f"{pid}_{name}")
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
                cv2.putText(draw_frame, f"Registering {pid} ({face_cnt}/{IMG_PER_USER})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Face Registration ([Space] for capture, [ESC] for escape)', draw_frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 32: # Space key
                if bbox is not None and len(bbox) == 1:
                    face_img = detector.extract(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), bbox,
                        save_path=os.path.join(pdir, f"{os.path.basename(pdir)}_{face_cnt:04d}.jpg")
                    )
                    if face_img is not None: print(f"Extracted face {face_cnt} from user {pid}."); face_cnt += 1
                    else:                    print(f"Failed to extract face {face_cnt} from user {pid}.")
                else:
                    print('No face detected.')
            elif key == 27:
                break
            
        cap.release()
        cv2.destroyAllWindows()
        print(f"Added person ID {pid} to gallery.")
        index, info = build_db(emb_model)


def remove_face(pid, index, info, emb_model):
    if pid not in [i['pid'] for i in info]: print(f"ID {pid} does not exist."); return
    else:
        name = [i['name'] for i in info if i['pid'] == pid][0]
        pdir = os.path.join('gallery/user', f"{pid}_{name}")
        if os.path.exists(pdir):
            for file in os.listdir(pdir):
                os.remove(os.path.join(pdir, file))
            os.rmdir(pdir)
        print(f"Removed person ID {pid} from gallery.")
    
    index, info = build_db(emb_model)


def search_face(index, info, thresh=THRESHOLD):
    if index.ntotal == 0: print('No faces registered in the system.'); return
    
    votes = deque(maxlen=WIN_SIZE)
    cap   = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame      = cv2.flip(frame, 1)
        if not ret: break

        bbox, _    = detector.detect(frame)
        draw_frame = frame.copy()
        if bbox is not None and len(bbox) == 1:
            x1, y1, x2, y2 = [int(v) for v in bbox[0]]
            
            # Embedding phase
            face_img = detector.extract(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), bbox, save_path=None)
            if face_img is not None:
                face_emb   = get_embedding(embedder, face_img)
                D, I       = index.search(face_emb, 1)
                score, idx = D[0][0], I[0][0]

                if score >= thresh:
                    name = info[idx]['name']
                    votes.append((name, score))
                else:
                    votes.append(('Unknown', score))
                
                # Voting phase
                label, color = 'Unknown', (0, 0, 255)
                if len(votes) == WIN_SIZE:
                    names = [v[0] for v in votes if v[0] != 'Unknown']
                    if names:
                        most_common, cnt = Counter(names).most_common(1)[0]
                        if cnt >= MIN_VOTES:
                            avg_score = np.mean([s for n, s in votes if n == most_common])
                            label     = f'{most_common} ({avg_score:.2f})'
                            color     = (0, 255, 0)
                            
                # Rendering phase
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        else:
            # If no face detected → reset vote to "Unknown"
            votes.append(('Unknown', 0.0))

        time.sleep(0.03)
        cv2.imshow('Face Recognition', draw_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    embedder    = load_model(EmbeddingModel(), MODEL_PATH, DEVICE)
    detector    = MTCNN(image_size=IMG_SIZE, margin=10, keep_all=False, post_process=False, device=DEVICE)
    index, info = build_db(embedder)
    
    parser = argparse.ArgumentParser(description='Face Attendance System')
    parser.add_argument('--mode'  , type=str, choices=['build_db', 'detect', 'add', 'remove', 'search', 'db_info', 'remove_db'], required=True, help='Mode of operation')
    parser.add_argument('--pid'   , type=str, help='Person ID')
    parser.add_argument('--name'  , type=str, help='Person name')
    parser.add_argument('--thresh', type=float, default=THRESHOLD, help='Threshold for face recognition')
    args = parser.parse_args()
 
 
    if   args.mode == 'build_db' : index, info = build_db(embedder)
    elif args.mode == 'db_info'  : get_db_info(index, info)
    elif args.mode == 'remove_db': remove_db()
    elif args.mode == 'detect'   : detect_face()
    elif args.mode == 'add'      : add_face(args.pid, args.name, index, info, embedder)
    elif args.mode == 'remove'   : remove_face(args.pid, index, info, embedder)
    elif args.mode == 'search'   : search_face(index, info, args.thresh)
