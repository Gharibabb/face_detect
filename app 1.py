import streamlit as st
import os
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def prediction_img(img, detect_model, classNames):
    uploaded_image = Image.open(img)
    frame = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_img, 1.1, 4)
    for x, y, w, h in faces:
        roi_gray_img = gray_img[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        facess = face_detect.detectMultiScale(roi_gray_img)
        if len(facess) == 0:
            print("Visage non détecté")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey + eh, ex:ex + ew]

    final_img = cv2.resize(face_roi, (224, 224))
    final_img = np.expand_dims(final_img, axis=0)  # besoin de la 4e dimension
    final_img = final_img / 255  # normalisation

    prediction = detect_model.predict(final_img)
    pred = np.argmax(prediction[0])
    return classNames[pred]

def main():
    st.title("Reconnaissance d'Émotions Faciales")
    st.sidebar.header("Paramètres")

    uploaded_file = st.file_uploader("Choisissez une image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Image téléchargée.", use_column_width=True)
        st.write("")

        if st.button("Classer l'émotion"):
            st.write("Classification en cours...")
            classNames = []
            classFile = 'label.names'
            with open(classFile, 'rt') as f:
                classNames = f.read().rstrip('\n').split('\n')

            detect_model = tf.keras.models.load_model("face_emotion_rec_v2.h5")

            result = prediction_img(uploaded_file, detect_model, classNames)

            st.success(f"L'émotion prédite est : {result}")

if __name__ == "__main__":
    main()
