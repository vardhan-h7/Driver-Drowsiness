import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tensorflow as tf
import os
import av

# Load TFLite model
model_path = 'cnnCat2.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Haar cascades
face = cv2.CascadeClassifier(r'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_righteye_2splits.xml')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
lbl = ['Close', 'Open']

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.score = 0
        self.thicc = 2
        self.rpred = [99]
        self.lpred = [99]

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        height, width = frm.shape[:2]
        cv2.rectangle(frm, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frm[y:y + h, x:x + w]
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255.0
            r_eye = r_eye.astype(np.float32)
            r_eye = r_eye.reshape((1, 24, 24, 1))  # Adjust input shape
            interpreter.set_tensor(input_details[0]['index'], r_eye)
            interpreter.invoke()
            self.rpred = interpreter.get_tensor(output_details[0]['index'])
            self.rpred = np.argmax(self.rpred, axis=1)

            if self.rpred[0] == 1:
                lbl = 'Open'
            if self.rpred[0] == 0:
                lbl = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frm[y:y + h, x:x + w]
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255.0
            l_eye = l_eye.astype(np.float32)
            l_eye = l_eye.reshape((1, 24, 24, 1))  # Adjust input shape
            interpreter.set_tensor(input_details[0]['index'], l_eye)
            interpreter.invoke()
            self.lpred = interpreter.get_tensor(output_details[0]['index'])
            self.lpred = np.argmax(self.lpred, axis=1)

            if self.lpred[0] == 1:
                lbl = 'Open'
            if self.lpred[0] == 0:
                lbl = 'Closed'
            break

        if self.rpred[0] == 0 and self.lpred[0] == 0:
            self.score = self.score + 1
            cv2.putText(frm, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            self.score = self.score - 1
            cv2.putText(frm, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if self.score < 0:
            self.score = 0
        cv2.putText(frm, 'Score:' + str(self.score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if self.score > 15:
            # person is feeling sleepy so we beep the alarm
            st.session_state['sleepy'] = True
            if self.thicc < 16:
                self.thicc = self.thicc + 2
            else:
                self.thicc = self.thicc - 2
                if self.thicc < 2:
                    self.thicc = 2
            cv2.rectangle(frm, (0, 0), (width, height), (0, 0, 255), self.thicc)
        else:
            st.session_state['sleepy'] = False

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

def play_alarm():
    if st.session_state.get('sleepy', False):
        sound_html = """
        <audio autoplay>
        <source src="data:audio/wav;base64,{}" type="audio/wav">
        Your browser does not support the audio element.
        </audio>
        """.format(load_alarm())
        st.components.v1.html(sound_html)

def load_alarm():
    import base64
    with open("alarm.wav", "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode()

st.title("Drowsiness Detection System")
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

# Alarm sound playing section
play_alarm()
