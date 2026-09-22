import tensorflow as tf
import numpy as np
import cv2
import tensorflow.keras.applications.mobilenet_v2 as mobilenet_v2

model = tf.keras.models.load_model("models/trash_classifier.keras")
cam = cv2.VideoCapture(0)

width_frame = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

class_names = ["Compost", "Garbage", "None", "Recyclable"]

while True:
    ret, frame = cam.read()

    if not ret:
        break

    display_frame = frame.copy()

    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = mobilenet_v2.preprocess_input(frame)

    prediction = model.predict(frame, verbose = 0)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    text = f"Predicted Class: {predicted_class}"
    cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera', display_frame)

    if cv2.waitKey(1) == ord('q'):
        break   