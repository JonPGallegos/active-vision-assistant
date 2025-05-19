import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Load pre-trained DNN (MobileNet SSD) from OpenCV
net = cv2.dnn.readNetFromCaffe(
    "C:/Users/Judah/Documents/Projects/Active Vision Assistant/1. Active Vision Assistant/models/ssd/MobileNetSSD_deploy.prototxt", # The location has changed, YOU NEED TO UP DATE THIS LINE FOR THE NEW LOCATION
    "C:/Users/Judah/Documents/Projects/Active Vision Assistant/1. Active Vision Assistant/models/ssd/MobileNetSSD_deploy.caffemodel" # The location has changed, YOU NEED TO UP DATE THIS LINE FOR THE NEW LOCATION
)



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face + mesh
    result = face_mesh.process(rgb_frame)

    gaze_point = None  # To mark gaze location
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get iris center (landmarks 474–475 for right eye)
            iris = face_landmarks.landmark[474]
            ix, iy = int(iris.x * w), int(iris.y * h)
            gaze_point = (ix, iy)
            cv2.circle(frame, gaze_point, 5, (255, 0, 255), -1)
            break

    # Prepare frame for DNN object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Object detection logic
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            class_name = CLASSES[class_id]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Highlight object only if gaze is within the box
            if gaze_point and x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {int(confidence * 100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Active Vision Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
