import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# ================== Custom Streamlit Styling ==================
st.set_page_config(page_title="Emotion Based Music Recommender", page_icon="🎵", layout="centered")

st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #89f7fe 0%, #66a6ff 100%);
            color: #222;
        }
        .stApp {
            background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        }
        h1, h2, h3 {
            text-align: center;
            color: #2b2d42;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 2px solid #4361ee;
            background-color: #f0f8ff;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-size: 1em;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #2575fc 0%, #6a11cb 100%);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# ================== Header ==================
st.markdown("<h1>🎧 Emotion Based Music Recommender</h1>", unsafe_allow_html=True)
st.markdown("<h3>Let your emotions pick your next song!</h3>", unsafe_allow_html=True)

# ================== Model and Setup ==================
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not(emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# ================== Emotion Processing ==================
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 100), 3)
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# ================== User Inputs ==================
st.write("### 🎼 Enter Your Preferences Below")
lang = st.text_input("🎵 Preferred Language")
singer = st.text_input("🎤 Favorite Singer")

# ================== Webcam and Button ==================
if lang and singer and st.session_state["run"] != "false":
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("🎥 **Turn on your webcam to detect your emotion:**")
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

st.markdown("<br>", unsafe_allow_html=True)
btn = st.button("🎶 Recommend Me Songs")

if btn:
    if not(emotion):
        st.warning("⚠️ Please let me capture your emotion first!")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
