import os
import streamlit as st
from facenet_pytorch import MTCNN

from settings  import IMG_SIZE, DEVICE, MODEL_PATH, INDEX_PATH, INFO_PATH
from embedding import EmbeddingModel, load_model
from indexing  import build_index_and_info, get_index_and_info
from functions import add_face, remove_face, search_face, detect_face

emb_model     = load_model(EmbeddingModel(), MODEL_PATH, DEVICE)
face_detector = MTCNN(image_size=IMG_SIZE, margin=10, keep_all=False, post_process=False, device=DEVICE)


# ===== Page Configuration =====
st.set_page_config(page_title='Real-time Face Recognition', layout='wide')
st.title('🎥 Real-time Face Recognition System')


# ===== Information =====
build_index_and_info(emb_dim=128, index_path=INDEX_PATH, info_path=INFO_PATH)
index, info = get_index_and_info(INDEX_PATH, INFO_PATH)
with st.container(border=True):
    st.write('**Number of registered users:**', len(set([i['pid'] for i in info])))


# ===== Functionality =====
menu = st.sidebar.radio('**Select Function**', [
    'Face Detection',
    'Add Face (Face Registration)',
    'Remove Face (Face Deletion)',
    'Face Recognition',
])


if menu == 'Face Detection':
    with st.container(border=True):
        st.write(
            """
            ### Face Detection
            This function allows you to test the camera and detect faces in real-time. Click the button below to open the webcam and start face detection.
            """
        )
        if st.button('Start Face Detection'):
            detect_face(face_detector)
            st.success('Face detection completed!')
            

elif menu == 'Add Face (Face Registration)':
    with st.container(border=True):
        st.write(
            """
            ### Add Face (Face Registration)
            This function allows you to register a new user by capturing their face. Enter the user ID and name, then click the button below to start the webcam and capture the face images.
            """
        )
        pid  = st.text_input('User ID (PID)', value='admin')
        name = st.text_input('User Name', value='admin')
        if st.button('Start Face Registration'):
            if pid and name:
                add_face(pid, name, face_detector, emb_model)
                st.success(f'Face registration completed for {name} (PID: {pid})!')
            else:
                st.error('Please enter a valid User ID and Name.')


elif menu == 'Remove Face (Face Deletion)':
    with st.container(border=True):
        st.write(
            """
            ### Remove Face (Face Deletion)
            This function allows you to remove a registered user from the system. Enter the user ID and click the button below to remove the face images.
            """
        )
        pid = st.text_input('User ID (PID)', value='admin')
        if st.button('Remove Face'):
            if pid:
                remove_face(pid)
                st.success(f'Face removed for User ID: {pid}!')
            else:
                st.error('Please enter a valid User ID.')
                
                
elif menu == 'Face Recognition':
    with st.container(border=True):
        st.write(
            """
            ### Face Recognition
            This function allows you to recognize faces in real-time. Click the button below to start the webcam and perform face recognition.
            """
        )
        if st.button('Start Face Recognition'):
            search_face(face_detector, emb_model)
            st.success('Face recognition completed!')