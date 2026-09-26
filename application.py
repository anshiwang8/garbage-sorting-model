import tensorflow as tf
import numpy as np
import cv2
import tensorflow.keras.applications.mobilenet_v2 as mobilenet_v2
from ultralytics import YOLO

model = tf.keras.models.load_model("models/trash_classifier.keras")
yolo_model = YOLO(
    r"C:\Users\anshi\Desktop\garbage sorting\runs\detect\train-3\weights\best.pt"
)
cam = cv2.VideoCapture(0)   

width_frame = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

class_names = ["Compost", "Garbage", "None", "Recyclable"]

YOLO_CONF_THRESHOLD = 0.5

while True:
    ret, frame = cam.read()

    if not ret:
        break

    display_frame = frame.copy()

    # conf= drops any YOLO box below the threshold, so only confident detections remain
    detections = yolo_model.predict(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    object_frame = None

    if len(detections.boxes) > 0:
        best_box = max(
            detections.boxes,
            key=lambda box: float(box.conf)
        )

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        if x2 > x1 and y2 > y1:
            object_frame = frame[y1:y2, x1:x2]
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    if object_frame is not None:
        # MobileNet only runs when YOLO is at least YOLO_CONF_THRESHOLD certain
        frame = cv2.resize(object_frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = mobilenet_v2.preprocess_input(frame)

        prediction = model.predict(frame, verbose = 0)
        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]

        text = f"Predicted Class: {predicted_class}"

        # Show how much the model thinks the object belongs to each category
        for i, (name, prob) in enumerate(zip(class_names, prediction[0])):
            color = (0, 255, 0) if i == predicted_index else (255, 255, 255)
            cv2.putText(display_frame, f"{name}: {float(prob):.0%}", (20, 75 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        text = "No confident detection"

    cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera', display_frame)

    if cv2.waitKey(1) == ord('q'):
        break   
cam.release()
cv2.destroyAllWindows()