from flask import Flask, Response, jsonify, render_template
import cv2
import numpy as np
import pyttsx3
import threading

app = Flask(__name__)
latest_objects = []

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Load YOLOv4 model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

last_announcement = ""

def speak(text):
    def run_speech():
        local_engine = pyttsx3.init()
        local_engine.setProperty("rate", 150)
        local_engine.say(text)
        local_engine.runAndWait()

    threading.Thread(target=run_speech, daemon=True).start()

def generate_frames():
    global latest_objects, last_announcement
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(out_layers)

        boxes, confidences, class_ids = [], [], []
        detected_objects = set()

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    detected_objects.add(classes[class_id])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if detected_objects:
            announcement = "I see " + ", ".join(detected_objects)
            if announcement != last_announcement:
                speak(announcement)
                last_announcement = announcement

        latest_objects = list(detected_objects)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_objects')
def get_detected_objects():
    return jsonify({"objects": latest_objects})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/objects')
def alias_detected_objects():
    return get_detected_objects()

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, Response, jsonify
# from flask import Flask, render_template
# import cv2
# import numpy as np
# import pyttsx3
#
# app = Flask(__name__)
# latest_objects = []
#
# # Initialize Text-to-Speech engine
# engine = pyttsx3.init()
# engine.setProperty("rate", 150)
#
# # Load YOLOv4 model
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#
# layer_names = net.getLayerNames()
# out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
#
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()
#
# last_announcement = ""
#
# def generate_frames():
#     global latest_objects, last_announcement
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         height, width, _ = frame.shape
#         blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#         net.setInput(blob)
#         detections = net.forward(out_layers)
#
#         boxes, confidences, class_ids = [], [], []
#         detected_objects = set()
#
#         for output in detections:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#
#                 if confidence > 0.5:
#                     center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)
#
#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
#                     detected_objects.add(classes[class_id])
#
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
#
#         if len(indices) > 0:
#             for i in indices:
#                 i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
#                 x, y, w, h = boxes[i]
#                 label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         import threading
#
#         def speak(text):
#             def run_speech():
#                 engine = pyttsx3.init()
#                 engine.setProperty("rate", 150)
#                 engine.say(text)
#                 engine.runAndWait()
#
#             threading.Thread(target=run_speech, daemon=True).start()
#
#         # Inside generate_frames:
#         if detected_objects:
#             announcement = "I see " + ", ".join(detected_objects)
#             if announcement != last_announcement:
#                 speak(announcement)
#                 last_announcement = announcement
#
#         latest_objects = list(detected_objects)
#
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# @app.route('/detected_objects')
# def get_detected_objects():
#     return jsonify({"objects": latest_objects})
#
# @app.route('/')
# def home():
#     return '''
#     <h2>YOLOv4 Object Detection Flask App</h2>
#     <p>Video stream: <a href="/video_feed">Click here</a></p>
#     <p>Detected objects: <a href="/detected_objects">Click here</a></p>
#     '''
# @app.route('/objects')
# def alias_detected_objects():
#     return get_detected_objects()
#
# @app.route('/')
# def index():  # changed from 'home' to 'index'
#     return render_template('index.html')
#
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

# import cv2
# import numpy as np
# import pyttsx3 #text to speech
# #import tensorflow as tf
#
# # Initialize Text-to-Speech engine
# engine = pyttsx3.init()
# engine.setProperty("rate", 150)  # Adjust speech speed
#
# # Load YOLOv4 model
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
#
# # Optional: Use GPU if available (uncomment if you have CUDA installed)
# # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#
# # Load class names
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#
# layer_names = net.getLayerNames()
# out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
#
# # Open camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()
#
# last_announcement = ""
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     height, width, _ = frame.shape
#
#     # Convert frame for YOLO
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     detections = net.forward(out_layers)
#
#     boxes, confidences, class_ids = [], [], []
#     detected_objects = set()
#
#     for output in detections:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#
#             if confidence > 0.5:
#                 center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#                 detected_objects.add(classes[class_id])
#
#     # Apply Non-Maximum Suppression
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
#
#     if len(indices) > 0:
#         for i in indices:
#             i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
#             x, y, w, h = boxes[i]
#             label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     # Speak detected objects only if changed
#     if detected_objects:
#         announcement = "I see " + ", ".join(detected_objects)
#         if announcement != last_announcement:
#             print(announcement)
#             engine.say(announcement)
#             engine.runAndWait()
#             last_announcement = announcement
#
#     # Show frame (remove this if integrating with Flask web app)
#     cv2.imshow("YOLOv4 Object Detection", frame)
#
#     # Break on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
