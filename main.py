import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

import tempfile

movie_path = 'movie/'

def sidebar_parm():
    col1, col2 = st.sidebar.columns(2)
    button_run = col1.button('スタート')
    button_stop = col2.button('ストップ')
    mode = st.sidebar.selectbox('モードの選択', ['動画ファイル選択', 'WEBカメラ選択'])
    fps_val = st.sidebar.slider('フレームレート', 1, 100, 50)

    uploaded_mv_file = None
    if mode == '動画ファイル選択':
        uploaded_mv_file = st.sidebar.file_uploader("動画ファイルアップロード", type='mp4')
        if uploaded_mv_file is not None:
            st.sidebar.video(uploaded_mv_file)

    return button_run, button_stop, mode, fps_val, uploaded_mv_file


def create_virtual_bg(button_stop, mode, fps_val, movie_path, uploaded_mv_file):
    #################################################
    mv_file_path = None
    cap_file = None

    if mode == '動画ファイル選択':
        # mv_file_path = movie_path + uploaded_mv_file.name
        # cap_file = cv2.VideoCapture(mv_file_path)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_mv_file.read())

        cap_file = cv2.VideoCapture(tfile.name)
    else:
        cap_file = cv2.VideoCapture(1)

    #################################################
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))
    mark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))

    # Stremlitの動画貼り付けポイント
    image_container = st.empty()

    # 骨格推定処理
    with mp_holistic.Holistic(min_detection_confidence=0.5, static_image_mode=False) as holistic_detection:

        while cap_file.isOpened():
            success, image = cap_file.read()
            if not success:
                break
            if button_stop == True:
                break

            # 動画ファイル処理
            if mode == '動画ファイル選択':
                # image = cv2.resize(image , dsize=None, fx=0.2, fy=0.2)
                image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
            else:
                image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = holistic_detection.process(rgb_image)

            mp_drawing.draw_landmarks(
                image=rgb_image,
                landmark_list=results.face_landmarks,
                connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mesh_drawing_spec
                )
            mp_drawing.draw_landmarks(
                image=rgb_image,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mark_drawing_spec,
                connection_drawing_spec=mesh_drawing_spec
                )
            mp_drawing.draw_landmarks(
                image=rgb_image,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mark_drawing_spec,
                connection_drawing_spec=mesh_drawing_spec
                )
            mp_drawing.draw_landmarks(
                image=rgb_image,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mark_drawing_spec,
                connection_drawing_spec=mesh_drawing_spec
                )

            # 出力
            time.sleep(1/fps_val)
            # image_container.image(image)
            image_container.image(rgb_image)

    cap_file.release()
    return 0


if __name__ == "__main__":

    st.sidebar.title('動画を選択して、スタートボタンを教えてね！')
    button_run, button_stop, mode, fps_val, uploaded_mv_file = sidebar_parm()

    st.title('VTuberモーション骨格推定アプリ')
    if button_run == True:
        if mode == '動画ファイル選択' and uploaded_mv_file is None:
            st.text('動画ファイルをアップロードしてください')
        else:
            # cap_file = read_img_movie(movie_path, uploaded_mv_file, mode)
            create_virtual_bg(button_stop, mode, fps_val, movie_path, uploaded_mv_file)
